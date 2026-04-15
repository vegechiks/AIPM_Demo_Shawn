"""
基于 DeepSeek 的情感分析 —— 参考毕设 sentiment_analysis/experiments/run_llm_api.py
改动：移除评估/指标逻辑，支持外部自定义 prompt，接受 DataFrame 输入，yield 进度。
"""
from __future__ import annotations

import json
import re
import time
from typing import Generator

import pandas as pd
from openai import OpenAI

from backend.config import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

# ─────────────────────────────────────────
# 内置 Prompt 模板
# ─────────────────────────────────────────

PROMPT_EVENT_LEVEL = """Task:
Classify the sentiment polarity of a short-video comment about the public opinion event surrounding the rapid rise of DeepSeek R1.

Important:
This is NOT ordinary product review sentiment classification.
You must judge the comment's dominant stance toward the event-level discourse of DeepSeek R1, including model capability, cost efficiency, authenticity, open-source significance, market competition, geopolitical implications, and social impact.

Label definitions:
1. positive — The comment overall expresses support, approval, recognition, optimism, or admiration toward the DeepSeek R1 event.
2. negative — The comment overall expresses criticism, doubt, denial, pessimism, worry, ridicule, or hostility toward the DeepSeek R1 event.
3. neutral — The comment does NOT express a clear stance toward the DeepSeek R1 event (factual statements, pure questions, off-topic content, etc.).

Critical rule: Anti-A does NOT equal Pro-DeepSeek. If the comment only criticizes another country/company without supporting DeepSeek, label it neutral.
Special rule: If the comment recognizes technical strength but mainly expresses social anxiety or risk concern, label it negative.
Emoji rule: Judge emoji sentiment — 👍❤️🔥💯👏 are positive; 🤡💀🙄🤦🤮 are negative; 😐🤔🤷 are neutral.

Output: Return JSON only. Example: {"label":"positive"}
Do not output markdown, code fences, or extra explanation."""

PROMPT_GENERAL = """Task:
Classify the sentiment polarity of the following comment.

Label definitions:
1. positive — The comment expresses positive, supportive, happy, or approving emotions.
2. negative — The comment expresses negative, critical, sad, angry, or disapproving emotions.
3. neutral — The comment is factual, objective, or does not have a clear emotional stance.

Output: Return JSON only. Example: {"label":"positive"}
Do not output markdown, code fences, or extra explanation."""

PROMPT_TEMPLATES = {
    "事件级情感分析（论文方法）": PROMPT_EVENT_LEVEL,
    "通用情感分析": PROMPT_GENERAL,
}

VALID_LABELS = {"positive", "negative", "neutral"}
MAX_RETRIES = 3
RETRY_SLEEP = 2.0
REQUEST_SLEEP = 0.3


# ─────────────────────────────────────────
# API 调用
# ─────────────────────────────────────────

def _create_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def _strip_code_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = lines[1:] if len(lines) >= 2 else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _parse_label(raw: str) -> str | None:
    raw = _strip_code_fence(raw)
    if not raw:
        return None
    try:
        data = json.loads(raw)
        label = str(data.get("label") or "").strip().lower()
        return label if label in VALID_LABELS else None
    except Exception:
        # 直接搜索标签关键字作为兜底
        for label in VALID_LABELS:
            if label in raw.lower():
                return label
        return None


def _call_once(client: OpenAI, system_prompt: str, content: str) -> str | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f'Classify the following comment and return json only.\nComment: "{content}"'},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=32,
            )
            raw = resp.choices[0].message.content or ""
            label = _parse_label(raw)
            if label:
                return label
        except Exception:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP)
    return "neutral"  # 最终兜底


# ─────────────────────────────────────────
# 批量分析 Generator
# ─────────────────────────────────────────

SentimentYield = tuple[float, str, list[str]]
"""(progress, message, labels_so_far)"""


def run_sentiment_analysis(
    df: pd.DataFrame,
    system_prompt: str,
    api_key: str,
    max_comments: int = 200,
) -> Generator[SentimentYield, None, None]:
    """
    Generator，每处理一条评论 yield (progress, message, labels_so_far)。
    df 需包含 'content' 列。
    返回的 labels_so_far 与 df 前 max_comments 行一一对应。
    """
    client = _create_client(api_key)
    comments = df["content"].astype(str).tolist()[:max_comments]
    total = len(comments)
    labels: list[str] = []

    yield 0.0, f"开始分析，共 {total} 条评论...", []

    for i, text in enumerate(comments):
        label = _call_once(client, system_prompt, text)
        labels.append(label)
        progress = (i + 1) / total
        yield progress, f"正在分析第 {i+1}/{total} 条...", labels.copy()
        time.sleep(REQUEST_SLEEP)

    yield 1.0, f"✅ 情感分析完成，共分析 {total} 条评论", labels
