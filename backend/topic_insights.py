"""
AI report helpers for topic analysis results.
"""
from __future__ import annotations

import json
from hashlib import md5

import pandas as pd
from openai import OpenAI

from backend.config import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL


def _split_keywords(value: object) -> list[str]:
    return [word.strip() for word in str(value or "").split("、") if word.strip()]


def _topic_name_map(topic_words_df: pd.DataFrame) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for _, row in topic_words_df.iterrows():
        tid = int(row["topic_id"])
        mapping[tid] = str(row.get("topic_name") or f"主题 {tid + 1}")
    return mapping


def _representative_comments(group: pd.DataFrame, limit: int = 5) -> list[str]:
    if "topic_score" in group.columns:
        group = group.sort_values("topic_score", ascending=False)
    comments = []
    for text in group.get("content", pd.Series(dtype=str)).dropna().astype(str).head(limit):
        text = text.strip()
        if text:
            comments.append(text[:140])
    return comments


def build_topic_report_payload(result: dict, video_meta: dict | None = None) -> dict:
    topic_words_df: pd.DataFrame = result["topic_words_df"]
    doc_topic_df: pd.DataFrame = result["doc_topic_df"]
    word_freq: dict = result.get("word_freq") or {}
    n_docs = int(result.get("n_docs") or len(doc_topic_df))

    name_map = _topic_name_map(topic_words_df)
    topic_counts = (
        doc_topic_df["dominant_topic"].value_counts().to_dict()
        if "dominant_topic" in doc_topic_df.columns
        else {}
    )

    payload = {
        "video": video_meta or {},
        "n_docs": n_docs,
        "top_words": sorted(word_freq.items(), key=lambda item: item[1], reverse=True)[:20],
        "has_sentiment": "sentiment" in doc_topic_df.columns,
        "topics": [],
    }

    for _, row in topic_words_df.iterrows():
        tid = int(row["topic_id"])
        topic_group = (
            doc_topic_df[doc_topic_df["dominant_topic"] == tid]
            if "dominant_topic" in doc_topic_df.columns
            else pd.DataFrame()
        )
        count = int(topic_counts.get(tid, 0))
        topic_payload = {
            "topic_id": tid,
            "topic_name": name_map.get(tid, f"主题 {tid + 1}"),
            "topic_description": str(row.get("topic_description") or ""),
            "keywords": _split_keywords(row.get("keywords")),
            "count": count,
            "ratio": round(count / n_docs, 4) if n_docs else 0,
            "representative_comments": _representative_comments(topic_group),
        }

        if "sentiment" in topic_group.columns and not topic_group.empty:
            counts = topic_group["sentiment"].value_counts().to_dict()
            topic_payload["sentiment"] = {
                "positive": int(counts.get("positive", 0)),
                "neutral": int(counts.get("neutral", 0)),
                "negative": int(counts.get("negative", 0)),
            }
        else:
            topic_payload["sentiment"] = None

        payload["topics"].append(topic_payload)

    payload["topics"].sort(key=lambda item: item["count"], reverse=True)
    return payload


def topic_report_cache_key(result: dict, video_meta: dict | None = None) -> str:
    payload = build_topic_report_payload(result, video_meta=video_meta)
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return md5(raw.encode("utf-8")).hexdigest()


def generate_topic_ai_report(result: dict, api_key: str, video_meta: dict | None = None) -> str:
    payload = build_topic_report_payload(result, video_meta=video_meta)
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    prompt = f"""请基于以下短视频评论主题建模结果，生成中文主题分析报告。

要求：
1. 输出 Markdown。
2. 结构必须包括：主题概览、核心主题解读、主题热度排序、主题之间的关系、主题 × 情感分析、后续分析建议。
3. 如果 has_sentiment=false，主题 × 情感分析部分请说明尚未完成情感分析，无法解释各主题情感倾向。
4. 不要编造数据之外的信息；样本数少于 3 的主题只能作为参考，不要过度解读。
5. 解释主题时结合关键词、主题描述、占比和代表性评论。
6. 语言客观、简洁，适合放在数据分析报告中。

主题分析数据 JSON：
{json.dumps(payload, ensure_ascii=False)}
"""
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": "你是严谨的短视频评论主题分析师，擅长解释主题模型、主题热度和主题情感交叉结果。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1800,
    )
    return (resp.choices[0].message.content or "").strip()
