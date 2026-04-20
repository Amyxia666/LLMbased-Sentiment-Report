import streamlit as st
import json
import re
import io
import os
import tempfile
from collections import Counter
import requests
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib import rcParams

# ===== 字体配置 =====
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'PingFang SC', 'Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False

BG = "#0f0f13"
GRAY = "#aaaaaa"

# ==============================================================
# 页面配置
# ==============================================================
st.set_page_config(
    page_title="评论情感分析系统",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 用户评论情感分析系统")
st.caption("基于 LLM 的评论情感分析工具 · 支持批量处理、差评聚类、可视化报告")

# ==============================================================
# 侧边栏：配置区
# ==============================================================
with st.sidebar:
    st.header("⚙️ 配置")

    api_key = st.text_input(
        "Deepseek API Key",
        type="password",
        placeholder="sk-xxxxxxxxxxxxxxxx",
        help="你的 Deepseek API Key，不会被存储"
    )

    batch_size = st.slider("每批处理评论数", min_value=5, max_value=20, value=10,
                           help="每次 API 调用处理的评论数量，建议 8-10")

    auth_threshold = st.slider("真实性过滤阈值", min_value=1, max_value=9, value=5,
                                help="可信度低于此分值的评论将被标记为可疑")

    enable_competitor = st.checkbox("开启竞品对比", value=False)

    st.divider()
    st.markdown("**📌 数据格式说明**")
    st.markdown("""
上传 JSON 文件，结构需满足：
```json
{
  "result": {
    "items": [
      { "content": "评论正文" }
    ]
  }
}
```
没有数据？[下载示例文件](https://github.com/yourhandle/yourrepo/raw/main/data/sample_reviews.json)
""")

# ==============================================================
# 工具函数
# ==============================================================
def ask_model(prompt, api_key):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post("https://api.deepseek.com/v1/chat/completions",
                             headers=headers, json=data)
    return response.json()["choices"][0]["message"]["content"]

def clean_keywords(keywords):
    stopwords = {"非常棒","非常差","很好","很差","不错","一般","好","差",
                 "满意","不满意","喜欢","讨厌","棒","赞","烂","糟","感觉","觉得","还好","还行","可以"}
    cleaned, seen = [], set()
    for k in keywords:
        k = k.strip()
        if k and k not in stopwords and len(k) > 1 and k not in seen:
            cleaned.append(k)
            seen.add(k)
    return cleaned[:5]

def read_json_file(uploaded_file):
    data = json.load(uploaded_file)
    reviews = []
    items = data.get("result", {}).get("items", [])
    for item in items:
        content = item.get("content", "")
        if content and len(content.strip()) >= 5:
            reviews.append(content.strip())
    return reviews

"""新增"""
def read_multiple_json_files(uploaded_files):
    all_reviews = []
    seen = set()
    for f in uploaded_files:
        try:
            reviews = read_json_file(f)
            for r in reviews:
                if r not in seen:
                    all_reviews.append(r)
                    seen.add(r)
        except Exception as e:
            st.warning(f"文件 {f.name} 读取失败：{e}，已跳过")
    return all_reviews
                
def parse_single(raw):
    try:
        clean = re.sub(r"```json|```", "", raw).strip()
        item = json.loads(clean)
        return {
            "sentiment": item.get("sentiment", "unknown"),
            "reason": item.get("reason", ""),
            "keywords": clean_keywords(item.get("keywords", []))
        }
    except Exception:
        return {"sentiment": "unknown", "reason": "解析失败", "keywords": []}

# ==============================================================
# 批量真实性过滤
# ==============================================================
def filter_authentic(reviews, api_key, batch_size, threshold, progress_bar, status_text):
    authentic, suspicious = [], []
    total_batches = (len(reviews) + batch_size - 1) // batch_size

    for i in range(0, len(reviews), batch_size):
        batch = reviews[i: i + batch_size]
        batch_num = i // batch_size + 1
        status_text.text(f"🔍 真实性过滤中... 第 {batch_num}/{total_batches} 批")
        progress_bar.progress(batch_num / total_batches / 2)  # 占总进度前50%

        numbered = "\n".join([f"{j+1}. {r}" for j, r in enumerate(batch)])
        prompt = f"""你是反刷评专家。请判断以下 {len(batch)} 条评论是否真实，只输出 JSON 数组：
[
  {{
    "id": 1,
    "authenticity_score": 0到10的整数,
    "is_suspicious": true或false,
    "reason": "一句话"
  }}
]
评论：
{numbered}
只输出 JSON 数组，不要任何解释。"""

        try:
            raw = ask_model(prompt, api_key)
            clean = re.sub(r"```json|```", "", raw).strip()
            items = json.loads(clean)
            for j, item in enumerate(items):
                if j >= len(batch):
                    break
                review = batch[j]
                score = item.get("authenticity_score", 5)
                is_susp = item.get("is_suspicious", False)
                reason = item.get("reason", "")
                if is_susp or score < threshold:
                    suspicious.append((review, {"score": score, "reason": reason}))
                else:
                    authentic.append(review)
        except Exception:
            authentic.extend(batch)

    return authentic, suspicious

# ==============================================================
# 批量情感分析
# ==============================================================
def analyze_batch(reviews, api_key, batch_size, progress_bar, status_text):
    results = []
    total_batches = (len(reviews) + batch_size - 1) // batch_size

    for i in range(0, len(reviews), batch_size):
        batch = reviews[i: i + batch_size]
        batch_num = i // batch_size + 1
        status_text.text(f"🧠 情感分析中... 第 {batch_num}/{total_batches} 批")
        progress_bar.progress(0.5 + batch_num / total_batches / 2)  # 占总进度后50%

        numbered = "\n".join([f"{j+1}. {r}" for j, r in enumerate(batch)])
        prompt = f"""请分析以下 {len(batch)} 条评论的情感，只输出 JSON 数组：
[
  {{
    "id": 1,
    "sentiment": "positive 或 neutral 或 negative",
    "reason": "一句话说明主要情感原因",
    "keywords": ["关键词1", "关键词2", "关键词3"]
  }}
]
关键词规则：提取 3-5 个描述具体问题或优点的名词/动词短语，禁止纯情感词，禁止语义重复。
评论：
{numbered}
只输出 JSON 数组，不要任何解释。"""

        try:
            raw = ask_model(prompt, api_key)
            clean = re.sub(r"```json|```", "", raw).strip()
            parsed_items = json.loads(clean)
            if len(parsed_items) != len(batch):
                raise ValueError("数量不匹配")
            for item in parsed_items:
                results.append({
                    "sentiment": item.get("sentiment", "unknown"),
                    "reason": item.get("reason", ""),
                    "keywords": clean_keywords(item.get("keywords", []))
                })
        except Exception:
            # fallback：逐条处理
            status_text.text(f"⚠️ 第 {batch_num} 批解析异常，逐条重试...")
            for r in batch:
                prompt_single = f"""分析这条评论的情感，只输出JSON：
{{"sentiment": "positive/neutral/negative", "reason": "一句话", "keywords": ["词1","词2","词3"]}}
评论：{r}
只输出JSON。"""
                try:
                    raw = ask_model(prompt_single, api_key)
                    results.append(parse_single(raw))
                except Exception:
                    results.append({"sentiment": "unknown", "reason": "解析失败", "keywords": []})

    return results

# ==============================================================
# 总结 & 差评归类
# ==============================================================
def summarize_and_classify(reviews_with_results, api_key):
    pos_reviews = [r for r, res in reviews_with_results if res["sentiment"] == "positive"]
    neg_reviews = [r for r, res in reviews_with_results if res["sentiment"] == "negative"]

    def summarize(reviews, label):
        if not reviews:
            return f"没有{label}数据"
        text = "\n".join(reviews[:20])
        prompt = f"以下是用户评论（{label}）：\n{text}\n请总结最主要的3-5个原因，每条一行，格式：- 原因描述"
        return ask_model(prompt, api_key)

    pos_summary = summarize(pos_reviews, "好评")
    neg_summary = summarize(neg_reviews, "差评")

    # 差评归类
    reason_counts = {}
    if neg_reviews:
        text = "\n".join(neg_reviews[:30])
        prompt = f"""已知差评原因：\n{neg_summary}\n差评内容：\n{text}\n请统计每个原因对应数量。只输出JSON：{{"原因1": 数量}}"""
        try:
            raw = ask_model(prompt, api_key)
            clean = re.sub(r"```json|```", "", raw).strip()
            start, end = clean.find("{"), clean.rfind("}") + 1
            reason_counts = json.loads(clean[start:end])
        except Exception:
            pass

    return pos_summary, neg_summary, reason_counts

# ==============================================================
# 可视化
# ==============================================================
def make_pie_chart(pos, neu, neg):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.spines[:].set_visible(False)

    total = pos + neu + neg
    pie_data = [x for x in [pos, neu, neg] if x > 0]
    pie_labels = [l for l, x in zip(["正向", "中性", "负向"], [pos, neu, neg]) if x > 0]
    pie_colors = [c for c, x in zip(["#4ade80", "#facc15", "#f87171"], [pos, neu, neg]) if x > 0]

    wedges, texts, autotexts = ax.pie(
        pie_data, labels=pie_labels, colors=pie_colors,
        autopct="%1.1f%%", startangle=140, pctdistance=0.78,
        labeldistance=1.15,
        wedgeprops={"linewidth": 2.5, "edgecolor": BG},
        textprops={"fontsize": 12},
    )
    for t in texts:
        t.set_color("white")
    for at in autotexts:
        at.set_color(BG)
        at.set_fontweight("bold")

    ax.text(0, 0, f"{total}\n条评论", ha="center", va="center",
            fontsize=14, color="white", fontweight="bold", linespacing=1.6)
    ax.set_title("情感分布总览", color="white", fontsize=15, pad=20, fontweight="bold")
    fig.tight_layout(pad=2.5)
    return fig

def make_bar_chart(reason_counts):
    if not reason_counts:
        return None
    sorted_items = sorted(reason_counts.items(), key=lambda x: x[1])
    labels_bar = [item[0] for item in sorted_items]
    values_bar = [item[1] for item in sorted_items]
    n = len(labels_bar)

    fig_h = max(5, n * 0.7 + 2)
    fig, ax = plt.subplots(figsize=(9, fig_h))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.spines[:].set_visible(False)

    y_pos = np.arange(n)
    gradient_colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.75, n))
    bar_height = min(0.55, 0.9 / max(n, 1))
    bars = ax.barh(y_pos, values_bar, color=gradient_colors, height=bar_height, edgecolor="none")

    for bar, val in zip(bars, values_bar):
        ax.text(val + max(values_bar) * 0.02, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", ha="left", color="white", fontsize=10, fontweight="bold")

    wrapped = ["\n".join([lb[i:i+8] for i in range(0, len(lb), 8)]) for lb in labels_bar]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(wrapped, color="white", fontsize=10)
    ax.set_xlabel("评论数量", color=GRAY, fontsize=11, labelpad=8)
    ax.set_title("差评原因分布", color="white", fontsize=15, pad=20, fontweight="bold")
    ax.tick_params(axis="x", colors=GRAY, labelsize=10)
    ax.set_xlim(0, max(values_bar) * 1.35)
    ax.xaxis.grid(True, color="#1e1e2e", linewidth=0.8, linestyle="--")
    ax.set_axisbelow(True)
    fig.tight_layout(pad=2.5)
    return fig

# ==============================================================
# 主界面：文件上传区
# ==============================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📂 主产品评论")
    main_files = st.file_uploader("上传 JSON 文件", type="json", key="main",accept_multiple_files=True)

with col2:
    if enable_competitor:
        st.subheader("📂 竞品评论")
        comp_files = st.file_uploader("上传竞品 JSON 文件", type="json", key="comp",accept_multiple_files=True)
        comp_name = st.text_input("竞品名称", value="竞品A")
        my_name = st.text_input("我方名称", value="我方产品")

# ==============================================================
# 运行按钮
# ==============================================================
st.divider()

run_btn = st.button("🚀 开始分析", type="primary", use_container_width=True,
                    disabled=(not api_key or not main_files))

if not api_key:
    st.info("👈 请在左侧填入 API Key")
elif not main_files:
    st.info("👆 请先上传评论 JSON 文件")

# ==============================================================
# 分析流程
# ==============================================================
if run_btn:
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Step 1: 读取文件
    status_text.text("📖 读取评论文件...")
    try:
        reviews = read_multiple_json_files(main_files)
        st.success(f"✅ 读取完成：共 {len(reviews)} 条有效评论(来自{len(main_files)}个文件，已去重)")
    except Exception as e:
        st.error(f"文件读取失败：{e}")
        st.stop()

    # Step 2: 真实性过滤
    authentic, suspicious = filter_authentic(
        reviews, api_key, batch_size, auth_threshold, progress_bar, status_text
    )

    if suspicious:
        with st.expander(f"⚠️ 过滤掉 {len(suspicious)} 条可疑评论（点击展开）"):
            for rev, info in suspicious:
                st.markdown(f"- **[{info['score']}/10]** {rev[:60]}... → {info['reason']}")

    st.info(f"🔍 真实性过滤完成：保留 **{len(authentic)}** 条，剔除 **{len(suspicious)}** 条")

    # Step 3: 情感分析
    results_raw = analyze_batch(authentic, api_key, batch_size, progress_bar, status_text)
    reviews_with_results = list(zip(authentic, results_raw))

    progress_bar.progress(1.0)
    status_text.text("✅ 分析完成！")

    # Step 4: 总结
    with st.spinner("📝 生成总结与改进建议..."):
        pos_summary, neg_summary, reason_counts = summarize_and_classify(reviews_with_results, api_key)

    # ==============================================================
    # 展示结果
    # ==============================================================
    st.divider()
    st.header("📊 分析结果")

    # 情感统计
    sentiments = [r["sentiment"] for r in results_raw]
    counts = Counter(sentiments)
    pos, neu, neg = counts.get("positive", 0), counts.get("neutral", 0), counts.get("negative", 0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("总评论数", len(authentic))
    m2.metric("😊 好评", pos, f"{pos/len(authentic)*100:.1f}%" if authentic else "")
    m3.metric("😐 中性", neu, f"{neu/len(authentic)*100:.1f}%" if authentic else "")
    m4.metric("😞 差评", neg, f"{neg/len(authentic)*100:.1f}%" if authentic else "")

    # 图表
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig_pie = make_pie_chart(pos, neu, neg)
        st.pyplot(fig_pie)

    with chart_col2:
        if reason_counts:
            fig_bar = make_bar_chart(reason_counts)
            if fig_bar:
                st.pyplot(fig_bar)
        else:
            st.info("差评数量不足，无法生成原因分布图")

    # 总结文字
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.subheader("✅ 好评原因")
        st.markdown(pos_summary)
    with res_col2:
        st.subheader("❌ 差评原因")
        st.markdown(neg_summary)

    # 逐条明细（可折叠）
    st.divider()
    with st.expander("📋 逐条分析明细", expanded=False):
        sentiment_emoji = {"positive": "😊", "neutral": "😐", "negative": "😞", "unknown": "❓"}
        for i, (review, res) in enumerate(reviews_with_results, 1):
            emoji = sentiment_emoji.get(res["sentiment"], "❓")
            st.markdown(f"**{i}. {emoji} {review[:80]}{'...' if len(review) > 80 else ''}**")
            st.caption(f"原因：{res['reason']}　|　关键词：{'、'.join(res['keywords']) if res['keywords'] else '无'}")
            st.divider()

    # 保存图表供下载
    buf_pie = io.BytesIO()
    fig_pie.savefig(buf_pie, format="png", dpi=160, bbox_inches="tight", facecolor=BG)
    buf_pie.seek(0)

    st.download_button(
        label="📥 下载情感分布图",
        data=buf_pie,
        file_name="chart_sentiment_pie.png",
        mime="image/png"
    )

    if reason_counts:
        buf_bar = io.BytesIO()
        fig_bar.savefig(buf_bar, format="png", dpi=160, bbox_inches="tight", facecolor=BG)
        buf_bar.seek(0)
        st.download_button(
            label="📥 下载差评原因图",
            data=buf_bar,
            file_name="chart_neg_reasons.png",
            mime="image/png"
        )
