import html

import streamlit as st

from backend.asr import download_bilibili_audio, transcribe_audio_openai
from backend.bilibili_crawler import extract_bvid, validate_bilibili_input
from backend.bilibili_subtitle import (
    fetch_subtitle_json,
    get_video_subtitle_options,
    run_subtitle_summary,
    subtitle_body_to_text,
)
from backend.utils import render_sidebar_config

st.set_page_config(page_title="视频总结", page_icon="📝", layout="wide")
render_sidebar_config()

st.title("📝 视频总结")
st.caption("根据 B 站视频字幕生成结构化内容总结与关键时间线")

st.session_state.setdefault("subtitle_video_info", None)
st.session_state.setdefault("subtitle_options", [])
st.session_state.setdefault("subtitle_text", "")
st.session_state.setdefault("subtitle_summary", "")
st.session_state.setdefault("subtitle_summary_done", False)
st.session_state.setdefault("subtitle_bvid", "")
st.session_state.setdefault("subtitle_source", None)
st.session_state.setdefault("subtitle_summary_source", None)
st.session_state.setdefault("subtitle_status", "idle")


def reset_loaded_subtitle() -> None:
    st.session_state["subtitle_text"] = ""
    st.session_state["subtitle_summary"] = ""
    st.session_state["subtitle_summary_done"] = False
    st.session_state["subtitle_source"] = None
    st.session_state["subtitle_summary_source"] = None
    st.session_state["subtitle_status"] = "idle"


def reset_subtitle_query() -> None:
    st.session_state["subtitle_video_info"] = None
    st.session_state["subtitle_options"] = []
    st.session_state["subtitle_bvid"] = ""
    reset_loaded_subtitle()


def build_subtitle_source(video: dict, subtitle: dict) -> dict:
    return {
        "type": "bilibili",
        "bvid": video.get("bvid") or "",
        "aid": video.get("aid"),
        "cid": video.get("cid"),
        "subtitle_id": subtitle.get("id") or "",
        "lan": subtitle.get("lan") or "",
    }


def build_asr_source(video: dict) -> dict:
    return {
        "type": "asr",
        "bvid": video.get("bvid") or "",
        "aid": video.get("aid"),
        "cid": video.get("cid"),
    }


def source_matches_video(source: dict | None, video: dict | None) -> bool:
    if not source or not video:
        return False
    return (
        str(source.get("bvid") or "") == str(video.get("bvid") or "")
        and str(source.get("aid") or "") == str(video.get("aid") or "")
        and str(source.get("cid") or "") == str(video.get("cid") or "")
    )


st.markdown(
    """
    <style>
    .metric-tooltip {
        min-width: 0;
        position: relative;
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
    .metric-tooltip:hover::after {
        content: attr(data-tooltip);
        position: absolute;
        left: 0;
        top: calc(100% + 0.5rem);
        z-index: 9999;
        width: max-content;
        max-width: min(36rem, 70vw);
        padding: 0.55rem 0.7rem;
        border-radius: 6px;
        background: rgba(17, 24, 39, 0.96);
        color: #fff;
        font-size: 0.9rem;
        line-height: 1.45;
        white-space: normal;
        overflow-wrap: anywhere;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
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
        <div class="metric-tooltip" data-tooltip="{html.escape(tooltip_text, quote=True)}">
            <div class="metric-tooltip__label">{html.escape(label)}</div>
            <div class="metric-tooltip__value">{html.escape(value_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

default_bvid = st.session_state.get("current_bvid") or st.session_state.get("subtitle_bvid") or ""

st.divider()
st.subheader("视频信息")

input_col, action_col = st.columns([3, 1])
with input_col:
    video_input = st.text_input(
        "输入 B 站视频链接或 BV 号",
        value=default_bvid,
        placeholder="https://www.bilibili.com/video/BV1xx411c7mD  或  BV1xx411c7mD",
    )

with action_col:
    st.write("")
    fetch_clicked = st.button("获取字幕", type="primary", use_container_width=True)

cookie = st.session_state.get("bili_cookie", "").strip()
api_key = st.session_state.get("deepseek_key", "").strip()
asr_api_key = st.session_state.get("openai_asr_key", "").strip()

input_bvid = extract_bvid(video_input) if video_input.strip() else None
loaded_video = st.session_state.get("subtitle_video_info") or {}
loaded_bvid = loaded_video.get("bvid")
if input_bvid and loaded_bvid and input_bvid != loaded_bvid:
    reset_subtitle_query()
    st.info("已切换视频，请点击“获取字幕”重新查询当前视频字幕。")

if not cookie:
    st.info("部分视频字幕需要登录态。若获取失败，请先在左侧系统配置中填入 Bilibili Cookie。")

if fetch_clicked:
    valid, err_msg = validate_bilibili_input(video_input)
    if not valid:
        st.error(err_msg)
    else:
        bvid = extract_bvid(video_input)
        with st.spinner("正在查询视频字幕..."):
            try:
                video_info, subtitle_options = get_video_subtitle_options(bvid, cookie)
            except Exception as e:
                st.error(f"字幕查询失败：{e}")
            else:
                st.session_state["subtitle_video_info"] = video_info
                st.session_state["subtitle_options"] = subtitle_options
                st.session_state["subtitle_bvid"] = video_info["bvid"]
                st.session_state["current_bvid"] = video_info["bvid"]
                st.session_state["video_title"] = video_info["title"]
                reset_loaded_subtitle()
                if subtitle_options:
                    st.success(f"已找到 {len(subtitle_options)} 个字幕轨道。")
                else:
                    st.warning("该视频暂无可用字幕，或当前 Cookie 无权访问字幕。")

video_info = st.session_state.get("subtitle_video_info")
subtitle_options = st.session_state.get("subtitle_options") or []

if video_info:
    if st.session_state.get("subtitle_text") and not source_matches_video(
        st.session_state.get("subtitle_source"),
        video_info,
    ):
        reset_loaded_subtitle()
        st.warning("检测到旧字幕不属于当前视频，已自动清除。请重新加载字幕。")

    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        render_metric("视频标题", video_info.get("title", "—"), video_info.get("title", "—"))
    with info_col2:
        render_metric("BV 号", video_info.get("bvid", "—"), video_info.get("bvid", "—"))
    with info_col3:
        render_metric(
            "视频分段数",
            video_info.get("page_count", 1),
            "视频分段数：B站视频可能包含多个分段（常称 P1、P2、P3）。当前功能默认读取第 1 个分段的字幕；如果视频有多个分段，后续可扩展为选择指定分段。",
        )

st.divider()
st.subheader("字幕内容")

if subtitle_options:
    st.warning(
        "由于 B 站字幕接口不稳定，字幕内容出错率较高。请人工检查下方字幕预览；"
        "如果发现字幕不属于当前视频，可以多点几次“获取字幕/加载字幕”重试，"
        "或使用 ASR 音频转写获取字幕（需要配置 OpenAI API Key）。"
    )
    option_labels = [
        f"{item.get('lan_doc', '字幕')} ({item.get('lan', '-')})"
        for item in subtitle_options
    ]
    selected_label = st.selectbox("选择字幕轨道", option_labels)
    selected_index = option_labels.index(selected_label)
    selected_subtitle = subtitle_options[selected_index]

    load_col, meta_col = st.columns([1, 3])
    with load_col:
        load_clicked = st.button("加载字幕", use_container_width=True)
    with meta_col:
        st.caption(f"字幕来源：{selected_subtitle.get('lan_doc', '字幕')} | URL 为临时链接，建议每次实时获取。")

    if load_clicked:
        with st.spinner("正在下载字幕..."):
            try:
                subtitle_json = fetch_subtitle_json(
                    selected_subtitle["subtitle_url"],
                    cookie=cookie,
                    bvid=video_info.get("bvid"),
                )
                subtitle_text = subtitle_body_to_text(subtitle_json.get("body") or [])
            except Exception as e:
                st.error(f"字幕下载失败：{e}")
            else:
                st.session_state["subtitle_text"] = subtitle_text
                st.session_state["subtitle_summary"] = ""
                st.session_state["subtitle_summary_done"] = False
                st.session_state["subtitle_source"] = build_subtitle_source(
                    video_info,
                    selected_subtitle,
                )
                st.session_state["subtitle_summary_source"] = None
                st.session_state["subtitle_status"] = "bilibili_loaded"
                if subtitle_text:
                    st.success(f"B站字幕已加载，共 {len(subtitle_text.splitlines())} 行。请先人工检查字幕是否对应当前视频。")
                else:
                    st.warning("字幕为空，建议重试或使用 ASR 音频转写。")

subtitle_source = st.session_state.get("subtitle_source")
subtitle_text = st.session_state.get("subtitle_text") or ""
subtitle_is_current = bool(subtitle_text) and source_matches_video(subtitle_source, video_info)
if subtitle_text and not subtitle_is_current:
    reset_loaded_subtitle()
    subtitle_text = ""
    subtitle_is_current = False

if subtitle_text:
    source_type = (subtitle_source or {}).get("type")
    if source_type == "asr":
        st.success(f"ASR 转写完成，共 {len(subtitle_text.splitlines())} 行。")
    elif source_type == "bilibili":
        st.warning(
            f"B站字幕已加载，共 {len(subtitle_text.splitlines())} 行。"
            "由于接口不稳定，请人工确认字幕内容是否对应当前视频。"
        )
    with st.expander("字幕预览", expanded=True):
        preview_lines = subtitle_text.splitlines()[:30]
        st.text("\n".join(preview_lines))
        if len(subtitle_text.splitlines()) > 30:
            st.caption("仅展示前 30 行字幕。")

    st.download_button(
        "下载字幕 TXT",
        data=subtitle_text.encode("utf-8"),
        file_name=f"subtitle_{st.session_state.get('subtitle_bvid', 'video')}.txt",
        mime="text/plain",
    )

show_asr = bool(video_info)
if show_asr:
    st.divider()
    st.subheader("ASR 音频转写")
    st.caption("当 B站字幕缺失、明显错误或不稳定时，下载该视频音频并调用 OpenAI 云端语音识别生成转写文本。")
    if not asr_api_key:
        st.warning("请先在左侧系统配置中填入 OpenAI ASR API Key。")
    if st.button("使用 ASR 转写音频", disabled=not asr_api_key or not video_info):
        progress_bar = st.progress(0.0, text="准备下载音频...")
        try:
            progress_bar.progress(0.25, text="正在下载视频音频...")
            audio_path = download_bilibili_audio(video_info.get("bvid"), cookie=cookie)
            progress_bar.progress(0.65, text="正在调用云端 ASR 转写...")
            asr_text = transcribe_audio_openai(audio_path, asr_api_key)
            if not asr_text.strip():
                st.error("ASR 未返回有效文本。")
            else:
                st.session_state["subtitle_text"] = asr_text
                st.session_state["subtitle_summary"] = ""
                st.session_state["subtitle_summary_done"] = False
                st.session_state["subtitle_source"] = build_asr_source(video_info)
                st.session_state["subtitle_summary_source"] = None
                st.session_state["subtitle_status"] = "asr"
                progress_bar.progress(1.0, text="ASR 转写完成。")
                st.success(f"ASR 转写完成，共 {len(asr_text.splitlines())} 行。")
                st.rerun()
        except Exception as e:
            progress_bar.empty()
            st.error(f"ASR 转写失败：{e}")

st.divider()
st.subheader("AI 视频总结")

if not api_key:
    st.warning("请先在左侧系统配置中填入 DeepSeek API Key。")

can_summarize = bool(subtitle_text.strip()) and bool(api_key)
if st.button("生成视频总结", type="primary", disabled=not can_summarize):
    progress_bar = st.progress(0.0, text="准备生成总结...")
    status_text = st.empty()
    try:
        final_summary = ""
        for progress, message, summary in run_subtitle_summary(subtitle_text, api_key):
            progress_bar.progress(min(progress, 1.0), text=message)
            status_text.caption(message)
            if summary:
                final_summary = summary
        if final_summary:
            st.session_state["subtitle_summary"] = final_summary
            st.session_state["subtitle_summary_done"] = True
            st.session_state["subtitle_summary_source"] = subtitle_source
            progress_bar.progress(1.0, text="视频总结生成完成。")
            status_text.empty()
            st.success("视频总结生成完成。")
        else:
            st.error("未生成有效总结，请检查字幕内容或 API Key。")
    except Exception as e:
        st.error(f"生成视频总结失败：{e}")

summary_source = st.session_state.get("subtitle_summary_source")
summary_text = st.session_state.get("subtitle_summary") or ""
if summary_text and not source_matches_video(summary_source, video_info):
    st.session_state["subtitle_summary"] = ""
    st.session_state["subtitle_summary_done"] = False
    st.session_state["subtitle_summary_source"] = None
    summary_text = ""

if summary_text:
    st.markdown(summary_text)
    st.download_button(
        "下载视频总结 Markdown",
        data=summary_text.encode("utf-8"),
        file_name=f"summary_{st.session_state.get('subtitle_bvid', 'video')}.md",
        mime="text/markdown",
    )
