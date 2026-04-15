import html

import pandas as pd
import streamlit as st

from backend.utils import render_sidebar_config, load_df, enrich_province_column

st.set_page_config(page_title="数据展示", page_icon="📋", layout="wide")
render_sidebar_config()

st.title("📋 数据展示")
st.caption("查看爬取到的原始评论数据")

# ── 数据检查 ──
data_file = st.session_state.get("data_file")
if not data_file:
    st.warning("⚠️ 尚未爬取数据，请先前往「📥 数据爬取」页面完成数据采集。")
    st.stop()

df = load_df(data_file)
if df is None or df.empty:
    st.error("数据文件读取失败或为空，请重新爬取。")
    st.session_state["data_file"] = None
    st.stop()

df = enrich_province_column(df)

# ── 顶部统计卡片 ──
st.divider()
st.markdown(
    """
    <style>
    .metric-tooltip {
        min-width: 0;
    }
    .metric-tooltip__label {
        color: rgba(49, 51, 63, 0.72);
        font-size: 0.875rem;
        line-height: 1.25;
        margin-bottom: 0.25rem;
    }
    .metric-tooltip__value {
        color: rgb(49, 51, 63);
        font-size: 2.25rem;
        line-height: 1.2;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        cursor: default;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_metric(label: str, value: object, tooltip: object | None = None) -> None:
    value_text = str(value)
    tooltip_text = str(tooltip if tooltip is not None else value_text)
    st.markdown(
        f"""
        <div class="metric-tooltip" title="{html.escape(tooltip_text, quote=True)}">
            <div class="metric-tooltip__label">{html.escape(label)}</div>
            <div class="metric-tooltip__value">{html.escape(value_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


video_title = df["video_title"].iloc[0] if "video_title" in df.columns else "—"
bvid = df["bvid"].iloc[0] if "bvid" in df.columns else "—"

m1, m2, m3, m4, m5 = st.columns(5)
with m1:
    render_metric("总评论数", f"{len(df):,}")
with m2:
    render_metric("视频标题", video_title, video_title)
with m3:
    render_metric("BV 号", bvid, bvid)
male = (df["gender"] == "男").sum() if "gender" in df.columns else 0
female = (df["gender"] == "女").sum() if "gender" in df.columns else 0
with m4:
    render_metric("男 / 女", f"{male} / {female}")
provinces = df["ip_province"].nunique() if "ip_province" in df.columns else 0
with m5:
    render_metric("涉及省份/地区数", provinces)

st.divider()

# ── 筛选控制栏 ──
filter_col1, filter_col2, filter_col3 = st.columns([2, 1, 1])

with filter_col1:
    search_kw = st.text_input("🔍 关键词搜索（评论内容）", placeholder="输入关键词筛选...")

with filter_col2:
    gender_options = ["全部"] + sorted(df["gender"].dropna().unique().tolist()) if "gender" in df.columns else ["全部"]
    gender_filter = st.selectbox("性别筛选", gender_options)

with filter_col3:
    sort_by = st.selectbox("排序方式", ["默认（评论时间）", "点赞量从高到低", "点赞量从低到高"])

# ── 应用筛选 ──
filtered_df = df.copy()

if search_kw.strip():
    filtered_df = filtered_df[
        filtered_df["content"].astype(str).str.contains(search_kw.strip(), case=False, na=False)
    ]

if gender_filter != "全部" and "gender" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["gender"] == gender_filter]

if sort_by == "点赞量从高到低" and "like_count" in filtered_df.columns:
    filtered_df = filtered_df.sort_values("like_count", ascending=False)
elif sort_by == "点赞量从低到高" and "like_count" in filtered_df.columns:
    filtered_df = filtered_df.sort_values("like_count", ascending=True)

st.caption(f"共 {len(filtered_df)} 条记录（总计 {len(df)} 条）")

# ── 数据表格 ──
display_cols_map = {
    "username": "用户名",
    "content": "评论内容",
    "like_count": "点赞量",
    "gender": "性别",
    "ip_province": "IP属地",
    "comment_time": "评论时间",
}
available_display_cols = [c for c in display_cols_map if c in filtered_df.columns]
show_df = filtered_df[available_display_cols].rename(columns=display_cols_map)

st.dataframe(
    show_df,
    use_container_width=True,
    height=500,
    column_config={
        "评论内容": st.column_config.TextColumn(width="large"),
        "点赞量": st.column_config.NumberColumn(format="%d"),
    },
)

# ── 下载按钮 ──
csv_bytes = filtered_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
st.download_button(
    label="⬇️ 下载当前筛选结果 CSV",
    data=csv_bytes,
    file_name=f"comments_{st.session_state.get('current_bvid', 'export')}.csv",
    mime="text/csv",
)
