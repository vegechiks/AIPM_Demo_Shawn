"""
BTM 主题建模 —— 参考毕设 topic_modeling_project/run_btm_model.py + preprocess.py
改动：去掉文件 I/O 依赖，接受 DataFrame 输入，返回结果；专注中文（Bilibili）。
"""
from __future__ import annotations

import re
from typing import Generator

import numpy as np
import pandas as pd
import jieba
import bitermplus as btm


# ─────────────────────────────────────────
# 默认中文停用词（内置基础集）
# ─────────────────────────────────────────

DEFAULT_STOPWORDS_ZH = {
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一", "一个",
    "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好",
    "自己", "这", "那", "啊", "吧", "呢", "嗯", "哦", "哈", "呀", "啥", "咋", "嘛",
    "对", "为", "以", "就是", "什么", "这个", "那个", "然后", "但是", "还是", "因为",
    "所以", "如果", "可以", "没", "让", "用", "来", "大", "中", "他", "她", "它",
    "这样", "那样", "已经", "现在", "时候", "知道", "觉得", "感觉", "一下", "一样",
    "可能", "应该", "还有", "只是", "而且", "不是", "真的", "一直", "一些", "关于",
    "更", "被", "把", "比", "从", "这里", "那里", "只有", "之前", "之后", "最",
    "多", "少", "新", "小", "大", "个", "年", "月", "日", "点", "分", "秒",
}

# 专有名词保护（避免被 jieba 切分）
CUSTOM_WORDS = [
    "deepseek", "DeepSeek", "r1", "R1", "gpt4", "GPT4", "gpt-4", "GPT-4",
    "yyds", "666", "emo", "破防", "芯片", "开源", "闭源", "大模型",
    "人工智能", "机器学习", "深度学习", "训练", "推理",
]


# ─────────────────────────────────────────
# 文本预处理（参考原 preprocess.py 的中文逻辑）
# ─────────────────────────────────────────

def _add_custom_words():
    for w in CUSTOM_WORDS:
        jieba.add_word(w)


def _clean_zh(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"\[[^\]]+\]", " ", text)  # 去平台表情 [笑哭]
    text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize_zh(text: str, stopwords: set[str], min_len: int = 2) -> list[str]:
    return [
        tok
        for tok in jieba.lcut(text)
        if tok.strip() and tok.strip() not in stopwords and len(tok.strip()) >= min_len
    ]


def preprocess_comments(
    df: pd.DataFrame,
    extra_stopwords: set[str] | None = None,
    min_doc_len: int = 2,
) -> tuple[list[list[str]], pd.DataFrame]:
    """
    返回 (texts, filtered_df)
    texts: 每个文档的 token list
    filtered_df: 过滤掉太短文档后的 df（index 已 reset）
    """
    _add_custom_words()
    stopwords = DEFAULT_STOPWORDS_ZH.copy()
    if extra_stopwords:
        stopwords.update(extra_stopwords)

    df = df.dropna(subset=["content"]).copy()
    df["_clean"] = df["content"].apply(_clean_zh)
    df["_tokens"] = df["_clean"].apply(lambda x: _tokenize_zh(x, stopwords))
    df["_token_len"] = df["_tokens"].apply(len)
    df = df[df["_token_len"] >= min_doc_len].reset_index(drop=True)

    texts = df["_tokens"].tolist()
    return texts, df


# ─────────────────────────────────────────
# BTM 训练
# ─────────────────────────────────────────

def train_btm(
    texts: list[list[str]],
    num_topics: int,
    iterations: int = 200,
    top_n_words: int = 10,
) -> tuple[btm.BTM, list[list[str]], np.ndarray]:
    """
    返回 (model, topic_words_list, p_zd)
    topic_words_list: 每个主题的 top_n 词列表
    p_zd: 文档-主题分布矩阵 shape=(n_docs, num_topics)
    """
    texts_joined = [" ".join(doc) for doc in texts]

    X, vocabulary, vocab_dict = btm.get_words_freqs(
        texts_joined,
        lowercase=False,
        token_pattern=r"(?u)\b\S+\b",
    )
    docs_vec = btm.get_vectorized_docs(texts_joined, vocabulary)
    biterms = btm.get_biterms(docs_vec)

    alpha = 50.0 / num_topics
    model = btm.BTM(
        X, vocabulary,
        T=num_topics,
        M=20,
        alpha=alpha,
        beta=0.01,
        seed=42,
        win=15,
        has_background=False,
    )
    model.fit(biterms, iterations=iterations, verbose=False)

    top_words_df = btm.get_top_topic_words(model, words_num=top_n_words)
    topic_words = []
    for t in range(model.topics_num_):
        col = f"topic{t}"
        words = list(top_words_df[col].dropna().astype(str))
        topic_words.append(words)

    p_zd = model.transform(docs_vec)  # shape: (n_docs, num_topics)
    return model, topic_words, p_zd


# ─────────────────────────────────────────
# 词频统计（用于词云）
# ─────────────────────────────────────────

def get_word_frequencies(texts: list[list[str]]) -> dict[str, int]:
    freq: dict[str, int] = {}
    for tokens in texts:
        for tok in tokens:
            freq[tok] = freq.get(tok, 0) + 1
    return freq


# ─────────────────────────────────────────
# 主题分析 Generator（供 Streamlit 驱动进度条）
# ─────────────────────────────────────────

TopicYield = tuple[float, str]
"""(progress, message)"""


def run_topic_analysis(
    df: pd.DataFrame,
    num_topics: int,
    extra_stopwords_str: str = "",
    iterations: int = 200,
) -> Generator[TopicYield, None, None]:
    """
    Generator，yield (progress, message)
    最终结果通过 result 字典写入（调用方传入可变对象）。
    实际上由于 generator 无法返回值，我们最后 yield 一个特殊 flag。
    调用方需要在 progress==1.0 后读取结果。
    """
    yield 0.05, "正在解析停用词..."
    extra_sw = set()
    if extra_stopwords_str.strip():
        for word in re.split(r"[,，\s\n]+", extra_stopwords_str.strip()):
            w = word.strip()
            if w:
                extra_sw.add(w)

    yield 0.15, "正在对评论进行分词和预处理..."
    texts, filtered_df = preprocess_comments(df, extra_stopwords=extra_sw)

    if len(texts) < num_topics * 2:
        yield 1.0, f"❌ 有效评论数量不足（{len(texts)} 条），无法训练 {num_topics} 个主题。请减少主题数量或爬取更多评论。"
        return

    yield 0.35, f"预处理完成，有效文档 {len(texts)} 条，开始训练 BTM 模型（{num_topics} 个主题）..."

    try:
        model, topic_words, p_zd = train_btm(
            texts, num_topics=num_topics, iterations=iterations
        )
    except Exception as e:
        yield 1.0, f"❌ BTM 训练失败：{e}"
        return

    yield 0.85, "模型训练完成，正在整理结果..."

    word_freq = get_word_frequencies(texts)

    # 构建文档-主题分配结果
    dominant_topics = np.argmax(p_zd, axis=1).tolist()
    dominant_scores = np.max(p_zd, axis=1).tolist()
    filtered_df = filtered_df.copy()
    filtered_df["dominant_topic"] = dominant_topics
    filtered_df["topic_score"] = [round(s, 4) for s in dominant_scores]

    # 构建主题关键词 DataFrame
    rows = []
    for tid, words in enumerate(topic_words):
        rows.append({
            "topic_id": tid,
            "topic_name": f"主题 {tid + 1}",
            "keywords": "、".join(words),
            **{f"word_{i+1}": w for i, w in enumerate(words)},
        })
    topic_words_df = pd.DataFrame(rows)

    yield 1.0, f"✅ 主题分析完成！共发现 {num_topics} 个主题，覆盖 {len(filtered_df)} 条评论"

    # 将结果附在最后一次 yield 的 message 里是不够的
    # 使用特殊约定：最后返回结果对象给调用方
    # （实际由调用方在 progress==1.0 后调用 run_topic_analysis_sync 获取结果）


def run_topic_analysis_sync(
    df: pd.DataFrame,
    num_topics: int,
    extra_stopwords_str: str = "",
    iterations: int = 200,
) -> dict:
    """
    同步版本，直接返回结果字典。
    供 Streamlit 在 generator 走完后调用。
    返回:
      {
        "topic_words_df": DataFrame,
        "doc_topic_df": DataFrame (含 dominant_topic, topic_score 列),
        "word_freq": dict,
        "n_docs": int,
      }
    """
    extra_sw = set()
    if extra_stopwords_str.strip():
        for word in re.split(r"[,，\s\n]+", extra_stopwords_str.strip()):
            w = word.strip()
            if w:
                extra_sw.add(w)

    texts, filtered_df = preprocess_comments(df, extra_stopwords=extra_sw)

    if len(texts) < num_topics * 2:
        raise ValueError(
            f"有效评论数量不足（{len(texts)} 条），无法训练 {num_topics} 个主题。"
            "请减少主题数量或爬取更多评论。"
        )

    model, topic_words, p_zd = train_btm(texts, num_topics=num_topics, iterations=iterations)
    word_freq = get_word_frequencies(texts)

    dominant_topics = np.argmax(p_zd, axis=1).tolist()
    dominant_scores = np.max(p_zd, axis=1).tolist()
    filtered_df = filtered_df.copy()
    filtered_df["dominant_topic"] = dominant_topics
    filtered_df["topic_score"] = [round(s, 4) for s in dominant_scores]

    rows = []
    for tid, words in enumerate(topic_words):
        rows.append({
            "topic_id": tid,
            "topic_name": f"主题 {tid + 1}",
            "keywords": "、".join(words),
            **{f"word_{i+1}": w for i, w in enumerate(words)},
        })
    topic_words_df = pd.DataFrame(rows)

    return {
        "topic_words_df": topic_words_df,
        "doc_topic_df": filtered_df,
        "word_freq": word_freq,
        "n_docs": len(filtered_df),
    }
