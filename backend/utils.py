"""
共享工具函数：字体检测、IP省份解析、词云生成、AI主题命名、侧边栏配置渲染
"""
from __future__ import annotations

import io
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import OpenAI

from backend.config import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


# ─────────────────────────────────────────
# 中文字体检测（词云需要）
# ─────────────────────────────────────────

_FONT_CANDIDATES = [
    # Windows
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/SimHei.ttf",
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
    # Linux / Streamlit Cloud
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
]


def get_chinese_font() -> str | None:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    return None


# ─────────────────────────────────────────
# IP 省份解析
# ─────────────────────────────────────────

def parse_ip_province(ip_location: str) -> str:
    if not ip_location:
        return "未知"
    s = str(ip_location).strip()
    # 去除 "IP属地：" 前缀
    if "：" in s:
        return s.split("：")[-1].strip()
    if ":" in s:
        return s.split(":")[-1].strip()
    return s or "未知"


def enrich_province_column(df: pd.DataFrame) -> pd.DataFrame:
    """在 df 上添加 ip_province 列"""
    df = df.copy()
    df["ip_province"] = df["ip_location"].apply(parse_ip_province)
    return df


# ─────────────────────────────────────────
# 词云生成
# ─────────────────────────────────────────

def generate_wordcloud_image(
    word_freq: dict[str, int],
    width: int = 900,
    height: int = 450,
) -> bytes | None:
    """
    生成词云并返回 PNG bytes。
    需要 wordcloud 和 matplotlib。
    若没有中文字体则返回 None。
    """
    try:
        from wordcloud import WordCloud
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    font_path = get_chinese_font()
    if not font_path:
        return None

    wc = WordCloud(
        font_path=font_path,
        width=width,
        height=height,
        background_color="white",
        max_words=120,
        colormap="Blues",
        prefer_horizontal=0.9,
        min_font_size=10,
        max_font_size=100,
        random_state=42,
    )
    wc.generate_from_frequencies(word_freq)

    buf = io.BytesIO()
    wc.to_image().save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


# ─────────────────────────────────────────
# AI 主题命名（DeepSeek）
# ─────────────────────────────────────────

def ai_name_topics(
    topic_words_df: pd.DataFrame,
    api_key: str,
) -> pd.DataFrame:
    """
    为每个主题的 top 词调用 DeepSeek 起名和描述。
    返回更新了 topic_name 和 topic_description 列的新 DataFrame。
    """
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    df = topic_words_df.copy()
    names = []
    descriptions = []

    for _, row in df.iterrows():
        keywords = row.get("keywords") or ""
        prompt = (
            f"以下是从短视频评论中通过BTM主题模型挖掘出的一组主题关键词：\n"
            f"关键词：{keywords}\n\n"
            "请根据这些词语的语义关联，完成两件事：\n"
            "1. 给这个主题起一个简洁的中文名称（4-8个汉字）\n"
            "2. 用一句话（15-25字）描述该主题的核心内容\n\n"
            "请严格按照以下JSON格式返回，不要包含任何其他内容：\n"
            '{"name": "主题名称", "description": "一句话描述"}'
        )

        name, desc = f"主题 {int(row['topic_id']) + 1}", ""
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": "你是一个专业的文本分析助手，擅长归纳主题。"},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                    max_tokens=100,
                )
                raw = resp.choices[0].message.content or ""
                # 去掉 code fence
                raw = raw.strip()
                if raw.startswith("```"):
                    lines = raw.splitlines()
                    raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).strip()
                data = json.loads(raw)
                name = str(data.get("name") or name).strip()
                desc = str(data.get("description") or "").strip()
                break
            except Exception:
                if attempt < 2:
                    time.sleep(1.5)

        names.append(name)
        descriptions.append(desc)
        time.sleep(0.5)

    df["topic_name"] = names
    df["topic_description"] = descriptions
    return df


# ─────────────────────────────────────────
# 侧边栏配置渲染（每个页面调用）
# ─────────────────────────────────────────

def render_sidebar_config():
    """在侧边栏渲染 Cookie 和 API Key 配置框，同步到 session_state"""
    st.session_state.setdefault("bili_cookie", "")
    st.session_state.setdefault("deepseek_key", "")
    st.session_state.setdefault("data_file", None)
    st.session_state.setdefault("sentiment_file", None)
    st.session_state.setdefault("topic_result", None)
    st.session_state.setdefault("current_bvid", "")
    st.session_state.setdefault("video_title", "")

    with st.sidebar:
        st.markdown("---")
        with st.expander("⚙️ 系统配置", expanded=False):
            cookie_val = st.text_area(
                "Bilibili Cookie",
                value=st.session_state["bili_cookie"],
                height=80,
                placeholder="粘贴您的 Bilibili Cookie（登录后从浏览器复制）",
            )
            st.session_state["bili_cookie"] = cookie_val

            key_val = st.text_input(
                "DeepSeek API Key",
                value=st.session_state["deepseek_key"],
                type="password",
                placeholder="sk-...",
            )
            st.session_state["deepseek_key"] = key_val

        st.markdown("---")
        st.caption("📊 数据流程状态")
        raw_ok = st.session_state.get("data_file") is not None
        sent_ok = st.session_state.get("sentiment_file") is not None
        topic_ok = st.session_state.get("topic_result") is not None

        bvid = st.session_state.get("current_bvid") or ""
        vtitle = st.session_state.get("video_title") or ""
        label = f"《{vtitle}》" if vtitle else (bvid or "—")

        st.markdown(
            f"{'✅' if raw_ok else '⏳'} 数据爬取：{label if raw_ok else '未开始'}\n\n"
            f"{'✅' if sent_ok else '⏳'} 情感分析：{'已完成' if sent_ok else '未开始'}\n\n"
            f"{'✅' if topic_ok else '⏳'} 主题分析：{'已完成' if topic_ok else '未开始'}"
        )


# ─────────────────────────────────────────
# CSV 保存 / 读取
# ─────────────────────────────────────────

def save_df(df: pd.DataFrame, path: str | Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def load_df(path: str | Path) -> pd.DataFrame | None:
    p = Path(path)
    if not p.exists():
        return None
    return pd.read_csv(p, encoding="utf-8-sig")
