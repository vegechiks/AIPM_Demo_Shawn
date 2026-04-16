import pandas as pd
import plotly.express as px
import streamlit as st

from backend.topic_model import run_topic_analysis_sync
from backend.utils import (
    render_sidebar_config,
    load_df,
    save_df,
    enrich_province_column,
    generate_wordcloud_image,
    ai_name_topics,
)
from backend.config import DATA_DIR

st.set_page_config(page_title="主题分析", page_icon="🔬", layout="wide")
render_sidebar_config()

st.title("🔬 主题分析")
st.caption("使用 BTM（Biterm Topic Model）挖掘评论热点话题")

# ─────────────────────────────────────────
# 图表/结果渲染函数
# ─────────────────────────────────────────

def render_topic_results(result: dict):
    topic_words_df: pd.DataFrame = result["topic_words_df"]
    doc_topic_df: pd.DataFrame = result["doc_topic_df"]
    word_freq: dict = result["word_freq"]
    n_docs: int = result["n_docs"]

    st.divider()
    st.subheader("📊 主题分析结果")

    tab1, tab2, tab3 = st.tabs(["☁️ 词云", "📋 主题关键词", "📈 主题分布"])

    # ── Tab1: 词云 ──
    with tab1:
        st.markdown("**全量词频词云**（基于所有评论分词结果）")
        if word_freq:
            img_bytes = generate_wordcloud_image(word_freq)
            if img_bytes:
                st.image(img_bytes, use_container_width=True)
            else:
                st.warning(
                    "⚠️ 未检测到中文字体，无法生成词云。\n\n"
                    "本地运行请确保系统安装了 SimHei/微软雅黑字体；"
                    "云端部署请在 packages.txt 中添加 `fonts-wqy-zenhei`。"
                )
                # 降级展示词频 Top 30
                freq_df = (
                    pd.DataFrame(list(word_freq.items()), columns=["词语", "频次"])
                    .sort_values("频次", ascending=False)
                    .head(30)
                )
                fig = px.bar(
                    freq_df, x="频次", y="词语", orientation="h",
                    title="词频 Top 30",
                    color="频次", color_continuous_scale="Blues",
                )
                fig.update_layout(
                    height=600,
                    yaxis={"categoryorder": "total ascending"},
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("暂无词频数据")

    # ── Tab2: 主题关键词 ──
    with tab2:
        st.markdown(f"**BTM 共识别 {len(topic_words_df)} 个主题，每个主题展示 Top 10 关键词**")

        # 判断是否已经过 AI 命名
        has_ai_name = (
            "topic_description" in topic_words_df.columns
            and topic_words_df["topic_description"].notna().any()
            and (topic_words_df["topic_description"] != "").any()
        )

        # AI 命名按钮
        api_key = st.session_state.get("deepseek_key", "").strip()
        if not has_ai_name:
            if api_key:
                if st.button("✨ AI 分析：为每个主题自动命名", type="secondary"):
                    with st.spinner("DeepSeek 正在分析主题关键词，请稍候..."):
                        try:
                            named_df = ai_name_topics(topic_words_df, api_key)
                            # 更新 session_state 中的结果
                            result["topic_words_df"] = named_df
                            st.session_state["topic_result"] = result
                            topic_words_df = named_df
                            has_ai_name = True
                            st.success("✅ AI 命名完成！")
                            st.rerun()
                        except Exception as e:
                            st.error(f"AI 命名失败：{e}")
            else:
                st.info("配置 DeepSeek API Key 后可使用 AI 自动为主题命名")

        # 展示每个主题
        num_topics = len(topic_words_df)
        cols_per_row = min(3, num_topics)
        rows = (num_topics + cols_per_row - 1) // cols_per_row

        for r in range(rows):
            cols = st.columns(cols_per_row)
            for c in range(cols_per_row):
                idx = r * cols_per_row + c
                if idx >= num_topics:
                    break
                row = topic_words_df.iloc[idx]
                topic_name = row.get("topic_name") or f"主题 {int(row['topic_id']) + 1}"
                topic_desc = row.get("topic_description") or ""
                keywords = row.get("keywords") or ""

                with cols[c]:
                    with st.container(border=True):
                        st.markdown(f"**{topic_name}**")
                        if topic_desc:
                            st.caption(topic_desc)
                        st.markdown(
                            " ".join(
                                f"`{w}`"
                                for w in keywords.split("、")
                                if w.strip()
                            )
                        )

        # 完整 DataFrame
        with st.expander("查看完整主题关键词表格"):
            display_cols = [c for c in ["topic_id", "topic_name", "topic_description", "keywords"] if c in topic_words_df.columns]
            st.dataframe(topic_words_df[display_cols], use_container_width=True)

    # ── Tab3: 主题分布 ──
    with tab3:
        if "dominant_topic" in doc_topic_df.columns:
            # 合并主题名称
            name_map = {
                int(row["topic_id"]): (row.get("topic_name") or f"主题 {int(row['topic_id'])+1}")
                for _, row in topic_words_df.iterrows()
            }
            doc_topic_df = doc_topic_df.copy()
            doc_topic_df["主题名称"] = doc_topic_df["dominant_topic"].map(name_map)

            dist = doc_topic_df["主题名称"].value_counts().reset_index()
            dist.columns = ["主题", "评论数"]
            dist["占比"] = (dist["评论数"] / dist["评论数"].sum() * 100).round(1)

            col_a, col_b = st.columns(2)
            with col_a:
                fig_pie = px.pie(
                    dist, names="主题", values="评论数",
                    title="各主题评论占比",
                    hole=0.4,
                )
                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                fig_pie.update_layout(height=380)
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_b:
                fig_bar = px.bar(
                    dist, x="主题", y="评论数",
                    title="各主题评论数量",
                    text="评论数",
                    color="评论数",
                    color_continuous_scale="Blues",
                )
                fig_bar.update_traces(textposition="outside")
                fig_bar.update_layout(
                    height=380,
                    xaxis_title="",
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # 主题 × 情感交叉分析（若有情感数据）
            if "sentiment" in doc_topic_df.columns:
                st.markdown("**主题 × 情感交叉分析**")
                LABEL_MAP = {"positive": "积极", "neutral": "中性", "negative": "消极"}
                COLOR_MAP = {"积极": "#4CAF50", "中性": "#9E9E9E", "消极": "#F44336"}
                doc_topic_df["情感"] = doc_topic_df["sentiment"].map(LABEL_MAP).fillna(doc_topic_df["sentiment"])
                cross = (
                    doc_topic_df.groupby(["主题名称", "情感"])
                    .size()
                    .reset_index(name="数量")
                )
                fig_cross = px.bar(
                    cross, x="主题名称", y="数量", color="情感",
                    barmode="stack",
                    color_discrete_map=COLOR_MAP,
                    title="各主题情感构成",
                )
                fig_cross.update_layout(height=420, xaxis_title="")
                st.plotly_chart(fig_cross, use_container_width=True)

        else:
            st.info("暂无文档-主题分配数据")

    # ── 下载 ──
    st.divider()
    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "⬇️ 下载主题关键词 CSV",
            data=topic_words_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
            file_name=f"topic_words_{st.session_state.get('current_bvid', 'result')}.csv",
            mime="text/csv",
        )
    with dl2:
        if "dominant_topic" in doc_topic_df.columns:
            st.download_button(
                "⬇️ 下载评论-主题分配 CSV",
                data=doc_topic_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                file_name=f"topics_{st.session_state.get('current_bvid', 'result')}.csv",
                mime="text/csv",
            )


# ─────────────────────────────────────────
# 页面主体
# ─────────────────────────────────────────

data_file = st.session_state.get("data_file")
if not data_file:
    st.warning("⚠️ 尚未爬取数据，请先前往「📥 数据爬取」页面完成数据采集。")
    st.stop()

# 优先读取情感分析结果（含 sentiment 列）；没有则读原始数据
sentiment_file = st.session_state.get("sentiment_file")
if sentiment_file:
    source_df = load_df(sentiment_file)
    data_source_label = "情感分析结果（含情感标签）"
else:
    source_df = load_df(data_file)
    data_source_label = "原始爬取数据"

if source_df is None or source_df.empty:
    st.error("数据文件读取失败，请重新爬取。")
    st.stop()

source_df = enrich_province_column(source_df)
vtitle = source_df["video_title"].iloc[0][:30] if "video_title" in source_df.columns else ""
st.info(f"当前数据来源：**{data_source_label}** | {len(source_df)} 条 | 视频：{vtitle}")

if not sentiment_file:
    st.warning("💡 建议先完成「💬 情感分析」后再进行主题分析，可获得主题 × 情感交叉视图")

st.divider()

# ── 主题参数 ──
st.subheader("⚙️ 主题分析参数")
st.info(
    "停用词是在分词和主题建模中被过滤的常见词，会直接影响词云、主题关键词和主题分布。"
    "请在左侧「系统配置」中统一维护全局停用词；结合当前视频补充无意义口头词、梗词或平台词后，主题分析效果通常更好。"
)

p1, p2 = st.columns(2)
with p1:
    num_topics = st.slider(
        "主题数量（K）",
        min_value=2,
        max_value=10,
        value=5,
        help="建议根据评论数量选择：< 100 条 → K=3，100-500 条 → K=5，> 500 条 → K=6-8",
    )
with p2:
    iterations = st.slider(
        "迭代次数",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
        help="迭代次数越多，结果越稳定，但耗时越长",
    )

st.divider()

if st.button("🚀 开始主题分析", type="primary"):
    with st.status("正在进行主题分析...", expanded=True) as status:
        st.write("正在预处理评论文本...")
        try:
            result = run_topic_analysis_sync(
                source_df,
                num_topics=num_topics,
                extra_stopwords_str=st.session_state.get("custom_stopwords_text", ""),
                iterations=iterations,
            )
            st.write(f"✅ BTM 模型训练完成，覆盖 {result['n_docs']} 条有效文档")
            status.update(label="主题分析完成！", state="complete")
        except Exception as e:
            status.update(label="分析失败", state="error")
            st.error(f"❌ {e}")
            st.stop()

    st.session_state["topic_result"] = result
    st.success(f"✅ 主题分析完成！共发现 **{num_topics} 个主题**，覆盖 **{result['n_docs']} 条**评论")
    render_topic_results(result)

else:
    topic_result = st.session_state.get("topic_result")
    if topic_result:
        st.success(f"展示已有主题分析结果。点击上方按钮可重新分析。")
        render_topic_results(topic_result)
