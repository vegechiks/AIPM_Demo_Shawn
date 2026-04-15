from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from backend.config import DATA_DIR
from backend.sentiment import PROMPT_TEMPLATES, run_sentiment_analysis
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

    st.divider()
    st.subheader("📊 情感分析结果")

    tab1, tab2, tab3 = st.tabs(["🥧 整体分布", "👥 性别差异", "🗺️ 地区分布"])

    # ── Tab1: 整体分布 ──
    with tab1:
        counts = df["情感"].value_counts().reset_index()
        counts.columns = ["情感", "数量"]

        col_pie, col_bar = st.columns(2)
        with col_pie:
            fig_pie = px.pie(
                counts, names="情感", values="数量",
                color="情感", color_discrete_map=COLOR_MAP,
                title="情感分布",
                hole=0.4,
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
                )
                fig_pct.update_traces(texttemplate="%{text:.1f}%", textposition="inside")
                fig_pct.update_layout(height=380, xaxis_title="", yaxis_title="占比 (%)")
                st.plotly_chart(fig_pct, use_container_width=True)

    # ── Tab3: 地区分布 ──
    with tab3:
        if "ip_province" not in df.columns:
            st.info("数据中缺少地区字段")
        else:
            province_counts = (
                df["ip_province"]
                .value_counts()
                .reset_index()
            )
            province_counts.columns = ["省份/地区", "评论数"]
            province_counts = province_counts[province_counts["省份/地区"] != "未知"].head(20)

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
                df_top8 = df[df["ip_province"].isin(top8)]
                region_sent = (
                    df_top8.groupby(["ip_province", "情感"])
                    .size()
                    .reset_index(name="数量")
                    .rename(columns={"ip_province": "地区"})
                )
                if not region_sent.empty:
                    fig_rs = px.bar(
                        region_sent,
                        x="地区", y="数量", color="情感",
                        barmode="stack",
                        color_discrete_map=COLOR_MAP,
                        title="Top 8 省份的情感构成",
                    )
                    fig_rs.update_layout(height=400, xaxis_title="", yaxis_title="评论数")
                    st.plotly_chart(fig_rs, use_container_width=True)

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

# ── Prompt 配置 ──
st.subheader("📝 Prompt 配置")

template_name = st.selectbox(
    "选择内置 Prompt 模板",
    options=list(PROMPT_TEMPLATES.keys()),
    help="「事件级情感分析」为本研究论文方法；「通用情感分析」适用于普通话题",
)

custom_prompt = st.text_area(
    "Prompt 内容（可直接编辑）",
    value=PROMPT_TEMPLATES[template_name],
    height=250,
)

# ── 分析参数 ──
st.subheader("⚙️ 分析参数")
p1, p2 = st.columns(2)
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

api_key = st.session_state.get("deepseek_key", "").strip()
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
            raw_df, custom_prompt.strip(), api_key, max_comments=max_comments
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
