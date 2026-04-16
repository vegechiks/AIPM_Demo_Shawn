import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from backend.bilibili_crawler import extract_bvid, validate_bilibili_input, crawl_bilibili
from backend.config import DATA_DIR
from backend.utils import render_sidebar_config, save_df

st.set_page_config(page_title="数据爬取", page_icon="📥", layout="wide")
render_sidebar_config()

st.title("📥 数据爬取")
st.caption("当前版本支持：哔哩哔哩（B站）")

st.divider()

# ── 平台选择 ──
col_platform, col_blank = st.columns([1, 3])
with col_platform:
    platform = st.selectbox(
        "选择平台",
        options=["哔哩哔哩（B站）", "抖音（开发中）", "TikTok（开发中）", "YouTube（开发中）"],
    )

disabled = platform != "哔哩哔哩（B站）"
if disabled:
    st.warning("当前版本仅支持哔哩哔哩，其他平台正在开发中。")

# ── 视频链接输入 ──
st.subheader("视频链接")
url_col, hint_col = st.columns([3, 2])
with url_col:
    video_url = st.text_input(
        "输入 B 站视频链接或 BV 号",
        placeholder="https://www.bilibili.com/video/BV1xx411c7mD  或  BV1xx411c7mD",
        disabled=disabled,
    )

# 实时格式校验
with hint_col:
    st.write("")  # 占位
    if video_url.strip():
        valid, err_msg = validate_bilibili_input(video_url)
        if valid:
            bvid = extract_bvid(video_url)
            st.success(f"✅ 识别到 BV 号：`{bvid}`")
        else:
            st.error(f"❌ {err_msg}")
    else:
        st.caption("请输入视频链接")

# ── 爬取参数 ──
st.subheader("爬取参数")
param_col1, param_col2 = st.columns(2)
with param_col1:
    max_pages = st.slider(
        "最大爬取页数",
        min_value=5,
        max_value=100,
        value=30,
        step=5,
        help="B 站每页约 20 条评论，30 页约 600 条",
        disabled=disabled,
    )
with param_col2:
    st.metric("预计最大评论数", f"约 {max_pages * 20} 条")
    st.caption("当前仅统计一级评论，不含楼中楼回复；评论默认按热度优先获取，通常会优先覆盖点赞较高的热评。")

# ── Cookie 提醒 ──
cookie = st.session_state.get("bili_cookie", "").strip()
if not cookie:
    st.warning("⚠️ 未配置 Bilibili Cookie。请先在左侧 **⚙️ 系统配置** 中填入 Cookie，否则可能无法爬取或数据不完整。")

st.divider()

# ── 开始爬取按钮 ──
can_crawl = (
    not disabled
    and video_url.strip()
    and validate_bilibili_input(video_url)[0]
)

if st.button("🚀 开始爬取", type="primary", disabled=not can_crawl):
    bvid = extract_bvid(video_url)
    cookie = st.session_state.get("bili_cookie", "").strip()

    st.divider()
    progress_bar = st.progress(0.0, text="准备中...")
    status_text = st.empty()
    result_placeholder = st.empty()

    all_comments = []
    final_message = ""

    try:
        for progress, message, comments in crawl_bilibili(bvid, cookie, max_pages=max_pages):
            progress_bar.progress(min(progress, 1.0), text=message)
            status_text.caption(message)
            if comments:
                all_comments = comments
            if progress >= 1.0:
                final_message = message

        if all_comments:
            df = pd.DataFrame(all_comments)

            # 保存 CSV
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comments_{bvid}_{ts}.csv"
            filepath = DATA_DIR / filename
            save_df(df, filepath)

            # 更新 session_state
            st.session_state["data_file"] = str(filepath)
            st.session_state["current_bvid"] = bvid
            st.session_state["video_title"] = df["video_title"].iloc[0] if "video_title" in df.columns else bvid
            meta_cols = [
                "video_url", "bvid", "aid", "video_title", "video_desc", "video_pubdate",
                "video_duration", "video_tname", "up_name", "up_mid", "view_count",
                "like_count_video", "coin_count", "favorite_count", "share_count", "reply_count",
            ]
            st.session_state["video_meta"] = {
                col: df[col].iloc[0]
                for col in meta_cols
                if col in df.columns
            }
            # 新的爬取任务清空旧的分析结果
            st.session_state["sentiment_file"] = None
            st.session_state["topic_result"] = None

            progress_bar.progress(1.0, text="爬取完成！")
            status_text.empty()

            result_placeholder.success(
                f"✅ 爬取成功！共获取 **{len(df)} 条**评论，已保存至 `{filename}`"
            )

            # 展示数据预览
            with st.expander("📋 数据预览（前 10 条）", expanded=True):
                preview_cols = ["username", "content", "like_count", "gender", "ip_location", "comment_time"]
                show_cols = [c for c in preview_cols if c in df.columns]
                st.dataframe(df[show_cols].head(10), use_container_width=True)

            st.info("👉 点击左侧「数据展示」查看完整数据，或继续进行「情感分析」")

        else:
            error_msg = final_message or "未获取到评论数据"
            result_placeholder.error(error_msg)

    except Exception as e:
        st.error(f"❌ 爬取过程中发生错误：{e}")
