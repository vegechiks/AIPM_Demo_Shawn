"""
Shared stopword configuration for word frequency and topic analysis.
"""
from __future__ import annotations

import re

DEFAULT_STOPWORDS = {
    "的", "了", "是", "在", "我", "你", "他", "她", "它", "也", "就", "都", "很", "和", "啊", "吗", "吧", "呢",
    "有", "没有", "一个", "这个", "那个", "什么", "不是", "就是", "还是", "可以", "真的", "感觉", "觉得",
    "视频", "评论", "大家", "自己", "这么", "那么", "因为", "所以", "但是", "如果", "一下", "一样",
    "不", "人", "一", "上", "到", "说", "要", "去", "会", "着", "看", "好", "这", "那",
    "嗯", "哦", "哈", "呀", "啥", "咋", "嘛", "对", "为", "以", "然后", "没", "让", "用", "来",
    "大", "中", "这样", "那样", "已经", "现在", "时候", "知道", "可能", "应该", "还有", "只是",
    "而且", "一直", "一些", "关于", "更", "被", "把", "比", "从", "这里", "那里", "只有",
    "之前", "之后", "最", "多", "少", "新", "小", "个", "年", "月", "日", "点", "分", "秒",
}


def default_stopwords_text() -> str:
    return "\n".join(sorted(DEFAULT_STOPWORDS))


def parse_stopwords(text: str | None) -> set[str]:
    if text is None or not str(text).strip():
        return DEFAULT_STOPWORDS.copy()
    words = {
        word.strip()
        for word in re.split(r"[,，\s\n]+", str(text))
        if word.strip()
    }
    return words


def merge_stopwords(extra_text: str | None = None) -> set[str]:
    words = DEFAULT_STOPWORDS.copy()
    if extra_text and str(extra_text).strip():
        words.update(parse_stopwords(extra_text))
    return words
