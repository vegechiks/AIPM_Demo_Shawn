"""
基于 DeepSeek 的情感分析 —— 参考毕设 sentiment_analysis/experiments/run_llm_api.py
改动：移除评估/指标逻辑，支持外部自定义 prompt，接受 DataFrame 输入，yield 进度。
"""
from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    "事件级情感分析": PROMPT_EVENT_LEVEL,
    "通用情感分析": PROMPT_GENERAL,
}

VALID_LABELS = {"positive", "negative", "neutral"}
MAX_RETRIES = 3
RETRY_SLEEP = 2.0
REQUEST_SLEEP = 0.3
DEFAULT_MAX_WORKERS = 4
MAX_WORKERS_LIMIT = 8


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


def _classify_one(api_key: str, system_prompt: str, index: int, content: str) -> tuple[int, str]:
    client = _create_client(api_key)
    label = _call_once(client, system_prompt, content)
    return index, label or "neutral"


def generate_event_prompt(
    video_meta: dict,
    video_summary: str,
    sample_comments: list[str],
    api_key: str,
    base_template: str = PROMPT_EVENT_LEVEL,
) -> str:
    """
    Generate an event-specific sentiment prompt from video context.
    The returned text is meant to be edited by the user before batch analysis.
    """
    client = _create_client(api_key)

    meta_lines = []
    meta_fields = [
        ("视频标题", video_meta.get("video_title")),
        ("BV号", video_meta.get("bvid")),
        ("UP主", video_meta.get("up_name")),
        ("发布时间", video_meta.get("video_pubdate")),
        ("分区", video_meta.get("video_tname")),
        ("视频简介", video_meta.get("video_desc")),
        ("播放量", video_meta.get("view_count")),
        ("点赞数", video_meta.get("like_count_video")),
        ("评论数", video_meta.get("reply_count")),
    ]
    for label, value in meta_fields:
        if value not in (None, ""):
            meta_lines.append(f"{label}: {value}")

    comments_text = "\n".join(
        f"{idx}. {comment.strip()}"
        for idx, comment in enumerate(sample_comments, start=1)
        if str(comment).strip()
    )

    prompt = f"""你正在为短视频评论分析系统生成“事件级情感分析”Prompt。

目标：
1. 根据视频上下文识别该视频讨论的核心事件。
2. 明确评论情感应面向哪个事件/对象判断。
3. 模仿给定模板，生成可直接用于逐条评论分类的 system prompt。
4. 输出的 prompt 必须要求模型只返回 JSON：{{"label":"positive"}} / {{"label":"neutral"}} / {{"label":"negative"}}。

约束：
- 不要输出解释、Markdown 标题或代码块，只输出最终 system prompt。
- 保留 positive / neutral / negative 三分类。
- 不要把普通语气强弱直接等同于事件立场，必须围绕核心事件判断。
- 需要写清楚容易误判的情况，例如玩梗、反讽、只批评第三方、纯提问、无关评论。
- 如果视频总结缺失，就根据视频信息和评论样本谨慎归纳，不要编造具体事件细节。

【视频基础信息】
{chr(10).join(meta_lines) if meta_lines else "暂无"}

【视频总结】
{video_summary.strip() if video_summary.strip() else "暂无。建议用户先在视频总结页面生成总结，以便更精准识别事件。"}

【评论样本】
{comments_text or "暂无"}

【参考模板】
{base_template}
"""

    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": "你是严谨的短视频舆情分析方法专家，擅长为事件级情感分类设计高质量提示词。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1500,
    )
    return _strip_code_fence(resp.choices[0].message.content or "").strip()


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
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> Generator[SentimentYield, None, None]:
    """
    Generator，每完成一条评论 yield (progress, message, labels_so_far)。
    df 需包含 'content' 列。
    最终返回的 labels 与 df 前 max_comments 行一一对应。
    """
    comments = df["content"].astype(str).tolist()[:max_comments]
    total = len(comments)
    if total == 0:
        yield 1.0, "没有可分析的评论。", []
        return

    worker_count = max(1, min(int(max_workers or 1), MAX_WORKERS_LIMIT, total))
    labels: list[str | None] = [None] * total
    completed = 0

    yield 0.0, f"开始并发分析，共 {total} 条评论，并发数 {worker_count}...", []

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_classify_one, api_key, system_prompt, idx, text): idx
            for idx, text in enumerate(comments)
        }
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result_idx, label = future.result()
            except Exception:
                result_idx, label = idx, "neutral"
            labels[result_idx] = label
            completed += 1
            progress = completed / total
            partial = [item for item in labels if item is not None]
            yield (
                progress,
                f"正在并发分析评论：已完成 {completed}/{total} 条，并发数 {worker_count}",
                partial,
            )

    final_labels = [label or "neutral" for label in labels]
    yield 1.0, f"✅ 情感分析完成，共分析 {total} 条评论", final_labels
