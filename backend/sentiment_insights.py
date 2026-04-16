"""
Sentiment visualization helpers: region normalization, word frequency,
map data and AI report generation.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from hashlib import md5

import jieba
import pandas as pd
from openai import OpenAI
from pyecharts import options as opts
from pyecharts.charts import Map
from pyecharts.commons.utils import JsCode

from backend.config import DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
from backend.stopwords import merge_stopwords

CHINA_PROVINCES = {
    "北京", "天津", "上海", "重庆",
    "河北", "山西", "辽宁", "吉林", "黑龙江",
    "江苏", "浙江", "安徽", "福建", "江西", "山东",
    "河南", "湖北", "湖南", "广东", "海南",
    "四川", "贵州", "云南", "陕西", "甘肃", "青海",
    "内蒙古", "广西", "西藏", "宁夏", "新疆",
    "香港", "澳门", "台湾",
}

PROVINCE_ALIASES = {
    "北京市": "北京", "天津市": "天津", "上海市": "上海", "重庆市": "重庆",
    "河北省": "河北", "山西省": "山西", "辽宁省": "辽宁", "吉林省": "吉林", "黑龙江省": "黑龙江",
    "江苏省": "江苏", "浙江省": "浙江", "安徽省": "安徽", "福建省": "福建", "江西省": "江西", "山东省": "山东",
    "河南省": "河南", "湖北省": "湖北", "湖南省": "湖南", "广东省": "广东", "海南省": "海南",
    "四川省": "四川", "贵州省": "贵州", "云南省": "云南", "陕西省": "陕西", "甘肃省": "甘肃", "青海省": "青海",
    "内蒙古自治区": "内蒙古", "广西壮族自治区": "广西", "西藏自治区": "西藏",
    "宁夏回族自治区": "宁夏", "新疆维吾尔自治区": "新疆",
    "香港特别行政区": "香港", "澳门特别行政区": "澳门",
    "台湾省": "台湾",
}

def normalize_china_province(value: str) -> str | None:
    text = str(value or "").strip()
    if not text or text == "未知":
        return None
    text = text.replace("IP属地：", "").replace("IP属地:", "").strip()
    if text in PROVINCE_ALIASES:
        return PROVINCE_ALIASES[text]
    if text in CHINA_PROVINCES:
        return text
    for province in sorted(CHINA_PROVINCES, key=len, reverse=True):
        if text.startswith(province):
            return province
    return None


def with_china_province(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ip_province" not in out.columns:
        out["china_province"] = None
        return out
    out["china_province"] = out["ip_province"].apply(normalize_china_province)
    return out


def top_words_from_texts(texts, topn: int = 10, stopwords: set[str] | None = None) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    stopword_set = stopwords if stopwords is not None else merge_stopwords()
    for text in texts:
        for word in jieba.lcut(str(text or "")):
            word = word.strip().lower()
            if (
                len(word) < 2
                or word in stopword_set
                or re.fullmatch(r"[\W_]+", word)
                or re.fullmatch(r"\d+(\.\d+)?", word)
            ):
                continue
            counter[word] += 1
    return counter.most_common(topn)


def format_top_words(words: list[tuple[str, int]]) -> str:
    if not words:
        return "暂无"
    return "<br>".join(f"{word}: {count}" for word, count in words)


def top_words_html(df: pd.DataFrame, topn: int = 10, stopwords: set[str] | None = None) -> str:
    if df.empty or "content" not in df.columns:
        return "暂无"
    return format_top_words(top_words_from_texts(df["content"].astype(str).tolist(), topn=topn, stopwords=stopwords))


def sentiment_counts(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in df.columns or "sentiment" not in df.columns:
        return pd.DataFrame()
    counts = (
        df.groupby([group_col, "sentiment"])
        .size()
        .reset_index(name="count")
    )
    return counts


def build_china_map_rows(df: pd.DataFrame, stopwords: set[str] | None = None) -> list[dict]:
    if "sentiment" not in df.columns:
        return []
    china_df = with_china_province(df)
    china_df = china_df[china_df["china_province"].notna()].copy()
    rows: list[dict] = []
    for province, group in china_df.groupby("china_province"):
        positive = int((group["sentiment"] == "positive").sum())
        negative = int((group["sentiment"] == "negative").sum())
        neutral = int((group["sentiment"] == "neutral").sum())
        denom = positive + negative
        if denom <= 0:
            continue
        ratio = positive / denom
        rows.append(
            {
                "province": province,
                "positive": positive,
                "negative": negative,
                "neutral": neutral,
                "total": int(len(group)),
                "ratio": ratio,
                "top_words": top_words_html(group, stopwords=stopwords),
            }
        )
    rows.sort(key=lambda item: item["ratio"], reverse=True)
    return rows


def render_china_sentiment_map(rows: list[dict]) -> str:
    data_pair = [
        {
            "name": row["province"],
            "value": [
                round(row["ratio"], 4),
                row["positive"],
                row["negative"],
                row["neutral"],
                row["total"],
                row["top_words"],
            ],
        }
        for row in rows
    ]
    tooltip_formatter = JsCode(
        """
        function (params) {
            if (!params.value || !Array.isArray(params.value)) {
                return params.name + '<br/>暂无正负向评论数据';
            }
            const ratio = (params.value[0] * 100).toFixed(1) + '%';
            return params.name
                + '<br/>积极倾向: ' + ratio
                + '<br/>积极: ' + params.value[1]
                + '<br/>消极: ' + params.value[2]
                + '<br/>中性: ' + params.value[3]
                + '<br/>总评论: ' + params.value[4]
                + '<br/><br/>高频词 Top10:<br/>' + params.value[5];
        }
        """
    )
    chart = (
        Map(init_opts=opts.InitOpts(width="100%", height="620px"))
        .add(
            "积极倾向",
            data_pair,
            "china",
            is_map_symbol_show=False,
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            title_opts=opts.TitleOpts(
                title="中国地区积极倾向热图",
                subtitle="热度 = 积极评论数 / (积极评论数 + 消极评论数)，保留港澳台，排除境外与未知 IP",
            ),
            visualmap_opts=opts.VisualMapOpts(
                min_=0,
                max_=1,
                dimension=0,
                range_text=["积极占比高", "积极占比低"],
                is_calculable=True,
                range_color=["#fff7bc", "#fec44f", "#f03b20", "#bd0026"],
            ),
            tooltip_opts=opts.TooltipOpts(formatter=tooltip_formatter),
        )
    )
    return chart.render_embed()


def build_ai_report_payload(df: pd.DataFrame, stopwords: set[str] | None = None) -> dict:
    label_map = {"positive": "积极", "neutral": "中性", "negative": "消极"}
    payload: dict = {
        "overall": {},
        "gender": [],
        "regions": [],
    }

    overall_counts = df["sentiment"].value_counts().to_dict() if "sentiment" in df.columns else {}
    payload["overall"] = {
        label_map.get(k, k): int(v)
        for k, v in overall_counts.items()
    }
    payload["overall_top_words"] = top_words_from_texts(
        df.get("content", pd.Series(dtype=str)).astype(str).tolist(),
        topn=10,
        stopwords=stopwords,
    )

    if "gender" in df.columns:
        gender_map = {"男": "男性", "女": "女性", "保密": "未知"}
        tmp = df.copy()
        tmp["gender_group"] = tmp["gender"].map(gender_map).fillna("未知")
        for gender, group in tmp.groupby("gender_group"):
            counts = group["sentiment"].value_counts().to_dict()
            payload["gender"].append(
                {
                    "性别": gender,
                    "样本数": int(len(group)),
                    "积极": int(counts.get("positive", 0)),
                    "中性": int(counts.get("neutral", 0)),
                    "消极": int(counts.get("negative", 0)),
                    "高频词": top_words_from_texts(group["content"].astype(str).tolist(), topn=10, stopwords=stopwords),
                }
            )

    for row in build_china_map_rows(df, stopwords=stopwords):
        payload["regions"].append(
            {
                "省份": row["province"],
                "样本数": row["total"],
                "积极": row["positive"],
                "中性": row["neutral"],
                "消极": row["negative"],
                "积极倾向": round(row["ratio"], 4),
                "高频词": row["top_words"].replace("<br>", "；"),
            }
        )
    payload["regions"] = payload["regions"][:20]
    return payload


def report_cache_key(df: pd.DataFrame, stopwords_text: str = "") -> str:
    cols = [col for col in ["comment_id", "content", "sentiment", "gender", "ip_province"] if col in df.columns]
    raw = df[cols].to_json(force_ascii=False, orient="records") + "\n" + str(stopwords_text or "")
    return md5(raw.encode("utf-8")).hexdigest()


def generate_ai_sentiment_report(df: pd.DataFrame, api_key: str, stopwords: set[str] | None = None) -> str:
    payload = build_ai_report_payload(df, stopwords=stopwords)
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    prompt = f"""请基于以下短视频评论情感分析统计数据，生成中文分析报告。

要求：
1. 结构包括：整体情感概况、性别差异、地区差异、主要关注点、分析注意事项。
2. 不要过度解读小样本；样本数少于 3 的群体只能作为参考。
3. 地区积极倾向的定义是：积极评论数 / (积极评论数 + 消极评论数)，中性评论不进入分母。
4. 语言简洁、客观，适合放在数据分析报告中。

统计数据 JSON：
{json.dumps(payload, ensure_ascii=False)}
"""
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": "你是严谨的短视频舆情数据分析师，擅长解释情感分布、群体差异和区域差异。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1400,
    )
    return (resp.choices[0].message.content or "").strip()
