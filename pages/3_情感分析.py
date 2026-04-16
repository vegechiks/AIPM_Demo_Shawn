from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit.components.v1 import html

from backend.config import DATA_DIR
from backend.sentiment import PROMPT_TEMPLATES, generate_event_prompt, run_sentiment_analysis
from backend.sentiment_insights import (
    build_china_map_rows,
    generate_ai_sentiment_report,
    report_cache_key,
    render_china_sentiment_map,
    top_words_html,
    with_china_province,
)
from backend.stopwords import parse_stopwords
from backend.utils import render_sidebar_config, load_df, save_df, enrich_province_column

st.set_page_config(page_title="情感分析", page_icon="💬", layout="wide")
render_sidebar_config()

st.title("💬 情感分析")
st.caption("使用 DeepSeek 大模型对评论进行情感倾向分类")

# ─────────────────────────────────────────
# 图表渲染函数（定义在顶部）
# ─────────────────────────────────────────

def render_charts(df: pd.DataFrame):
    if "sentiment" not in df.columns:
        st.error("数据中缺少 sentiment 列")
        return

    LABEL_MAP = {"positive": "积极", "neutral": "中性", "negative": "消极"}
    COLOR_MAP = {"积极": "#4CAF50", "中性": "#9E9E9E", "消极": "#F44336"}

    df = df.copy()
    df["情感"] = df["sentiment"].map(LABEL_MAP).fillna(df["sentiment"])
    stopwords = parse_stopwords(st.session_state.get("custom_stopwords_text", ""))

    st.divider()
    st.subheader("📊 情感分析结果")

    tab1, tab2, tab3, tab4 = st.tabs(["🥧 整体分布", "👥 性别差异", "🗺️ 地区分布", "🤖 AI分析"])

    # ── Tab1: 整体分布 ──
    with tab1:
        counts = df["情感"].value_counts().reset_index()
        counts.columns = ["情感", "数量"]
        counts["top_words"] = [
            top_words_html(df[df["情感"] == label], stopwords=stopwords)
            for label in counts["情感"].tolist()
        ]

        col_pie, col_bar = st.columns(2)
        with col_pie:
            fig_pie = px.pie(
                counts, names="情感", values="数量",
                color="情感", color_discrete_map=COLOR_MAP,
                title="情感分布",
                hole=0.4,
                custom_data=["top_words"],
            )
            fig_pie.update_traces(
                hovertemplate="情感=%{label}<br>数量=%{value}<br>占比=%{percent}<br><br>高频词 Top10:<br>%{customdata[0]}<extra></extra>",
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(height=380)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_bar:
            fig_bar = px.bar(
                counts, x="情感", y="数量",
                color="情感", color_discrete_map=COLOR_MAP,
                title="情感数量",
                text="数量",
                custom_data=["top_words"],
            )
            fig_bar.update_traces(
                hovertemplate="情感=%{x}<br>数量=%{y}<br><br>高频词 Top10:<br>%{customdata[0]}<extra></extra>",
            )
            fig_bar.update_traces(textposition="outside")
            fig_bar.update_layout(showlegend=False, height=380, xaxis_title="", yaxis_title="评论数")
            st.plotly_chart(fig_bar, use_container_width=True)

        total = len(df)
        c1, c2, c3 = st.columns(3)
        pos = (df["情感"] == "积极").sum()
        neu = (df["情感"] == "中性").sum()
        neg = (df["情感"] == "消极").sum()
        c1.metric("积极评论", f"{pos} 条", f"{pos/total*100:.1f}%")
        c2.metric("中性评论", f"{neu} 条", f"{neu/total*100:.1f}%")
        c3.metric("消极评论", f"{neg} 条", f"{neg/total*100:.1f}%")

    # ── Tab2: 性别差异 ──
    with tab2:
        if "gender" not in df.columns:
            st.info("数据中缺少性别字段")
        else:
            gender_map = {"男": "男性", "女": "女性", "保密": "未知"}
            df_g = df.copy()
            df_g["性别"] = df_g["gender"].map(gender_map).fillna("未知")

            gender_sentiment = (
                df_g.groupby(["性别", "情感"])
                .size()
                .reset_index(name="数量")
            )
            valid_genders = df_g["性别"].value_counts()
            valid_genders = valid_genders[valid_genders >= 3].index.tolist()
            gender_sentiment = gender_sentiment[gender_sentiment["性别"].isin(valid_genders)]
            gender_sentiment["top_words"] = [
                    top_words_html(
                        df_g[(df_g["性别"] == row["性别"]) & (df_g["情感"] == row["情感"])],
                        stopwords=stopwords,
                    )
                for _, row in gender_sentiment.iterrows()
            ]

            if gender_sentiment.empty:
                st.info("性别数据样本不足，无法展示差异分析（每组需至少 3 条）")
            else:
                fig_g = px.bar(
                    gender_sentiment,
                    x="性别", y="数量", color="情感",
                    barmode="group",
                    color_discrete_map=COLOR_MAP,
                    title="不同性别的情感分布（分组柱状图）",
                    text="数量",
                    custom_data=["top_words"],
                )
                fig_g.update_traces(
                    hovertemplate="性别=%{x}<br>情感=%{fullData.name}<br>数量=%{y}<br><br>高频词 Top10:<br>%{customdata[0]}<extra></extra>",
                )
                fig_g.update_traces(textposition="outside")
                fig_g.update_layout(height=420, xaxis_title="", yaxis_title="评论数")
                st.plotly_chart(fig_g, use_container_width=True)

                pct = gender_sentiment.copy()
                total_by_gender = pct.groupby("性别")["数量"].transform("sum")
                pct["占比"] = (pct["数量"] / total_by_gender * 100).round(1)
                fig_pct = px.bar(
                    pct,
                    x="性别", y="占比", color="情感",
                    barmode="stack",
                    color_discrete_map=COLOR_MAP,
                    title="不同性别的情感占比（堆叠图）",
                    text="占比",
                    custom_data=["top_words"],
                )
                fig_pct.update_traces(
                    hovertemplate="性别=%{x}<br>情感=%{fullData.name}<br>占比=%{y:.1f}%<br><br>高频词 Top10:<br>%{customdata[0]}<extra></extra>",
                )
                fig_pct.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
                fig_pct.update_layout(height=380, xaxis_title="", yaxis_title="占比 (%)")
                st.plotly_chart(fig_pct, use_container_width=True)

    # ── Tab3: 地区分布 ──
    with tab3:
        if "ip_province" not in df.columns:
            st.info("数据中缺少地区字段")
        else:
            china_df = with_china_province(df)
            china_valid = china_df[china_df["china_province"].notna()].copy()
            map_rows = build_china_map_rows(df, stopwords=stopwords)
            if map_rows:
                st.markdown("#### 中国地区积极倾向热图")
                html(render_china_sentiment_map(map_rows), height=660)
            else:
                st.info("暂无可用于地图展示的中国省份正负向评论数据。")

            province_counts = (
                china_valid["china_province"]
                .value_counts()
                .reset_index()
            )
            province_counts.columns = ["省份/地区", "评论数"]
            province_counts = province_counts.head(20)
            province_counts["top_words"] = [
                top_words_html(china_valid[china_valid["china_province"] == region], stopwords=stopwords)
                for region in province_counts["省份/地区"].tolist()
            ]

            if province_counts.empty:
                st.info("暂无有效地区数据")
            else:
                fig_map = px.bar(
                    province_counts,
                    x="评论数", y="省份/地区",
                    orientation="h",
                    title=f"评论地区分布 Top {len(province_counts)}",
                    color="评论数",
                    color_continuous_scale="Blues",
                    text="评论数",
                    custom_data=["top_words"],
                )
                fig_map.update_traces(
                    hovertemplate="省份/地区=%{y}<br>评论数=%{x}<br><br>高频词 Top10:<br>%{customdata[0]}<extra></extra>",
                )
                fig_map.update_traces(textposition="outside")
                fig_map.update_layout(
                    height=max(400, len(province_counts) * 28),
                    yaxis={"categoryorder": "total ascending"},
                    coloraxis_showscale=False,
                    xaxis_title="评论数",
                    yaxis_title="",
                )
                st.plotly_chart(fig_map, use_container_width=True)

                top8 = province_counts.head(8)["省份/地区"].tolist()
                df_top8 = china_valid[china_valid["china_province"].isin(top8)]
                region_sent = (
                    df_top8.groupby(["china_province", "情感"])
                    .size()
                    .reset_index(name="数量")
                    .rename(columns={"china_province": "地区"})
                )
                region_sent["top_words"] = [
                    top_words_html(
                        df_top8[(df_top8["china_province"] == row["地区"]) & (df_top8["情感"] == row["情感"])],
                        stopwords=stopwords,
                    )
                    for _, row in region_sent.iterrows()
                ]
                if not region_sent.empty:
                    fig_rs = px.bar(
                        region_sent,
                        x="地区", y="数量", color="情感",
                        barmode="stack",
                        color_discrete_map=COLOR_MAP,
                        title="Top 8 省份的情感构成",
                        custom_data=["top_words"],
                    )
                    fig_rs.update_traces(
                        hovertemplate="地区=%{x}<br>情感=%{fullData.name}<br>数量=%{y}<br><br>高频词 Top10:<br>%{customdata[0]}<extra></extra>",
                    )
                    fig_rs.update_layout(height=400, xaxis_title="", yaxis_title="评论数")
                    st.plotly_chart(fig_rs, use_container_width=True)

    with tab4:
        st.markdown("#### AI 情感差异分析")
        api_key = st.session_state.get("deepseek_key", "").strip()
        stopwords_text = st.session_state.get("custom_stopwords_text", "")
        cache_key = report_cache_key(df, stopwords_text=stopwords_text)
        cached_key = st.session_state.get("sentiment_ai_report_key")
        cached_report = st.session_state.get("sentiment_ai_report", "")
        if cached_report and cached_key == cache_key:
            st.markdown(cached_report)
        if not api_key:
            st.warning("请先在左侧系统配置中填入 DeepSeek API Key。")
        if st.button("生成 AI 分析报告", disabled=not api_key):
            with st.spinner("正在生成整体、性别与地区差异分析..."):
                try:
                    report = generate_ai_sentiment_report(df, api_key, stopwords=stopwords)
                except Exception as e:
                    st.error(f"AI 分析生成失败：{e}")
                else:
                    st.session_state["sentiment_ai_report_key"] = cache_key
                    st.session_state["sentiment_ai_report"] = report
                    st.markdown(report)

    # ── 下载 ──
    st.divider()
    sentiment_file = st.session_state.get("sentiment_file")
    if sentiment_file:
        result_csv = load_df(sentiment_file)
        if result_csv is not None:
            st.download_button(
                "⬇️ 下载情感分析结果 CSV",
                data=result_csv.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                file_name=f"sentiment_{st.session_state.get('current_bvid', 'result')}.csv",
                mime="text/csv",
            )


# ─────────────────────────────────────────
# 页面主体
# ─────────────────────────────────────────

data_file = st.session_state.get("data_file")
if not data_file:
    st.warning("⚠️ 尚未爬取数据，请先前往「📥 数据爬取」页面完成数据采集。")
    st.stop()

raw_df = load_df(data_file)
if raw_df is None or raw_df.empty:
    st.error("数据文件读取失败，请重新爬取。")
    st.stop()

vtitle = raw_df["video_title"].iloc[0][:30] if "video_title" in raw_df.columns else ""
st.info(f"当前数据：**{len(raw_df)} 条**评论 | 视频：{vtitle}")
st.divider()

video_meta_cols = [
    "video_url", "bvid", "aid", "video_title", "video_desc", "video_pubdate",
    "video_duration", "video_tname", "up_name", "up_mid", "view_count",
    "like_count_video", "coin_count", "favorite_count", "share_count", "reply_count",
]
video_meta = {
    col: raw_df[col].iloc[0]
    for col in video_meta_cols
    if col in raw_df.columns
}
if video_meta:
    st.session_state["video_meta"] = video_meta

# ── Prompt 配置 ──
st.subheader("📝 Prompt 配置")

template_name = st.selectbox(
    "选择内置 Prompt 模板",
    options=list(PROMPT_TEMPLATES.keys()),
    help=(
        "事件级情感分析：围绕视频讨论的具体事件、争议对象或公共议题判断评论立场，"
        "适合热点事件、产品发布、社会争议等语境。"
        "通用情感分析：只根据评论文本本身判断积极、中性或消极，"
        "适合日常话题、泛娱乐内容或无明确事件对象的评论。"
    ),
)

if (
    st.session_state.get("sentiment_prompt_template") != template_name
    or "sentiment_custom_prompt" not in st.session_state
):
    st.session_state["sentiment_prompt_template"] = template_name
    st.session_state["sentiment_custom_prompt"] = PROMPT_TEMPLATES[template_name]

api_key = st.session_state.get("deepseek_key", "").strip()
is_event_template = template_name.startswith("事件级情感分析")
if is_event_template:
    video_summary = st.session_state.get("subtitle_summary", "").strip()
    if video_summary:
        st.success("已检测到视频总结，将用于生成更贴合当前视频事件的 Prompt。")
    else:
        st.info("建议先在「视频总结」页面生成视频总结。AI 会结合视频总结识别核心事件，生成的事件级 Prompt 会更精准。")

    sample_comments = (
        raw_df["content"]
        .dropna()
        .astype(str)
        .head(40)
        .tolist()
        if "content" in raw_df.columns
        else []
    )
    gen_disabled = not api_key
    if st.button("AI 生成 Prompt", disabled=gen_disabled):
        with st.spinner("正在根据视频上下文生成事件级 Prompt..."):
            try:
                generated_prompt = generate_event_prompt(
                    video_meta=video_meta or st.session_state.get("video_meta", {}),
                    video_summary=video_summary,
                    sample_comments=sample_comments,
                    api_key=api_key,
                    base_template=PROMPT_TEMPLATES[template_name],
                )
            except Exception as e:
                st.error(f"Prompt 生成失败：{e}")
            else:
                if generated_prompt:
                    st.session_state["sentiment_custom_prompt"] = generated_prompt
                    st.success("Prompt 已生成并填入下方文本框，可继续手动编辑后再开始分析。")
                    st.rerun()
                else:
                    st.error("未生成有效 Prompt，请检查 DeepSeek API Key 或稍后重试。")
    if gen_disabled:
        st.caption("填写 DeepSeek API Key 后可使用 AI 生成 Prompt。")

custom_prompt = st.text_area(
    "Prompt 内容（可直接编辑）",
    key="sentiment_custom_prompt",
    height=250,
)

# ── 分析参数 ──
st.subheader("⚙️ 分析参数")
p1, p2, p3 = st.columns(3)
with p1:
    max_comments = st.slider(
        "最大分析条数",
        min_value=20,
        max_value=min(500, len(raw_df)),
        value=min(200, len(raw_df)),
        step=20,
    )
with p2:
    st.metric("预计 DeepSeek API 调用次数", max_comments)
with p3:
    max_workers = st.select_slider(
        "并发数",
        options=[1, 2, 3, 4, 6, 8],
        value=4,
        help="并发越高速度越快，但更容易触发 API 限流；如果失败或限流较多，请调低并发数。",
    )

if not api_key:
    st.warning("⚠️ 未配置 DeepSeek API Key，请在左侧 **⚙️ 系统配置** 中填入。")

st.divider()

# ── 开始分析 ──
can_analyze = bool(api_key) and bool(custom_prompt.strip())

if st.button("🚀 开始情感分析", type="primary", disabled=not can_analyze):
    progress_bar = st.progress(0.0, text="准备中...")
    status_text = st.empty()
    labels = []

    try:
        for progress, message, current_labels in run_sentiment_analysis(
            raw_df,
            custom_prompt.strip(),
            api_key,
            max_comments=max_comments,
            max_workers=max_workers,
        ):
            progress_bar.progress(min(progress, 1.0), text=message)
            status_text.caption(message)
            if current_labels:
                labels = current_labels

        if labels:
            result_df = raw_df.head(len(labels)).copy()
            result_df["sentiment"] = labels
            result_df = enrich_province_column(result_df)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            bvid = st.session_state.get("current_bvid", "unknown")
            filepath = DATA_DIR / f"sentiment_{bvid}_{ts}.csv"
            save_df(result_df, filepath)
            st.session_state["sentiment_file"] = str(filepath)

            progress_bar.progress(1.0, text="分析完成！")
            status_text.empty()
            st.success(f"✅ 情感分析完成！共分析 **{len(labels)} 条**评论")
            render_charts(result_df)
        else:
            st.error("分析未返回结果，请检查 API Key 是否有效。")

    except Exception as e:
        st.error(f"❌ 分析过程中发生错误：{e}")

else:
    # 展示已有结果
    sentiment_file = st.session_state.get("sentiment_file")
    if sentiment_file:
        sent_df = load_df(sentiment_file)
        if sent_df is not None and not sent_df.empty and "sentiment" in sent_df.columns:
            sent_df = enrich_province_column(sent_df)
            st.success(f"展示已有分析结果（{len(sent_df)} 条）。点击上方按钮可重新分析。")
            render_charts(sent_df)
