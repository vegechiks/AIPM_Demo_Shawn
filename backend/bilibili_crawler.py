"""
B站评论爬虫 —— 参考毕设 crawler/bilibili_comment_crawler/src/crawler_bili.py
改动：去掉 MySQL 依赖，改为纯内存返回；使用 generator 推送进度给 Streamlit。
"""
from __future__ import annotations

import re
import time
import random
from datetime import datetime
from typing import Generator

import requests

PAGE_MAX_RETRIES = 3
PAGE_RETRY_SLEEP = 0.8
PAGE_INTERVAL_RANGE = (0.25, 0.6)


# ─────────────────────────────────────────
# URL / BV 解析与校验
# ─────────────────────────────────────────

def extract_bvid(url_or_bvid: str) -> str | None:
    s = url_or_bvid.strip()
    if re.match(r"^BV[0-9A-Za-z]{10,}$", s):
        return s
    m = re.search(r"(BV[0-9A-Za-z]{10,})", s)
    return m.group(1) if m else None


def validate_bilibili_input(text: str) -> tuple[bool, str]:
    """返回 (is_valid, error_msg)"""
    if not text.strip():
        return False, "链接不能为空"
    bvid = extract_bvid(text)
    if not bvid:
        return False, "无法识别 BV 号，请输入正确的 B 站视频链接或 BV 号（如 BV1xx411c7mD）"
    return True, ""


# ─────────────────────────────────────────
# 请求头构造
# ─────────────────────────────────────────

def _build_headers(cookie: str, bvid: str) -> dict:
    return {
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Origin": "https://www.bilibili.com",
        "Referer": f"https://www.bilibili.com/video/{bvid}",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Cookie": cookie,
    }


def _format_ts(ts: int | float | None) -> str:
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


def _parse_video_meta(video: dict, bvid: str, video_url: str) -> dict:
    owner = video.get("owner") or {}
    stat = video.get("stat") or {}
    return {
        "video_url": video_url,
        "bvid": video.get("bvid") or bvid,
        "aid": video.get("aid") or "",
        "video_title": video.get("title") or bvid,
        "video_desc": video.get("desc") or "",
        "video_pubdate": _format_ts(video.get("pubdate")),
        "video_duration": int(video.get("duration") or 0),
        "video_tname": video.get("tname") or "",
        "up_name": owner.get("name") or "",
        "up_mid": str(owner.get("mid") or ""),
        "view_count": int(stat.get("view") or 0),
        "like_count_video": int(stat.get("like") or 0),
        "coin_count": int(stat.get("coin") or 0),
        "favorite_count": int(stat.get("favorite") or 0),
        "share_count": int(stat.get("share") or 0),
        "reply_count": int(stat.get("reply") or 0),
    }


# ─────────────────────────────────────────
# 评论解析
# ─────────────────────────────────────────

def _parse_reply(rp: dict, video_meta: dict) -> dict | None:
    content = (rp.get("content") or {}).get("message") or ""
    if not content.strip():
        return None

    like_count = int(rp.get("like") or 0)
    ctime = rp.get("ctime")
    comment_time = (
        datetime.fromtimestamp(ctime).strftime("%Y-%m-%d %H:%M:%S")
        if ctime else ""
    )
    rpid = rp.get("rpid") or rp.get("rpid_str") or ""
    member = rp.get("member") or {}
    rc = rp.get("reply_control") or {}

    return {
        "platform": "bilibili",
        **video_meta,
        "comment_id": str(rpid),
        "user_id": str(member.get("mid") or ""),
        "username": member.get("uname") or "",
        "gender": member.get("sex") or "保密",
        "ip_location": rc.get("location") or "",
        "like_count": like_count,
        "comment_time": comment_time,
        "content": content,
    }


# ─────────────────────────────────────────
# 主爬虫（generator）
# ─────────────────────────────────────────

CrawlYield = tuple[float, str, list[dict]]
"""(progress 0~1, message, accumulated_comments)"""


def crawl_bilibili(
    bvid: str,
    cookie: str,
    max_pages: int = 30,
) -> Generator[CrawlYield, None, None]:
    """
    Generator，每次 yield (progress, message, all_comments_so_far)
    调用方在 Streamlit 里迭代即可驱动进度条。
    最后一次 yield progress==1.0 表示完成。
    """
    video_url = f"https://www.bilibili.com/video/{bvid}"
    headers = _build_headers(cookie, bvid)
    session = requests.Session()

    # ── Step 1: 获取视频信息 ──
    yield 0.02, "正在获取视频信息...", []
    try:
        r = session.get(
            "https://api.bilibili.com/x/web-interface/view",
            params={"bvid": bvid},
            headers=headers,
            timeout=10,
        )
        j = r.json()
        if j.get("code") != 0:
            msg = j.get("message") or "未知错误"
            yield 1.0, f"❌ 获取视频信息失败：{msg}（code={j.get('code')}）\n\n可能原因：Cookie 已失效或视频不存在", []
            return
        video_meta = _parse_video_meta(j["data"], bvid, video_url)
        aid = video_meta["aid"]
        title = video_meta["video_title"]
    except Exception as e:
        yield 1.0, f"❌ 网络请求失败：{e}", []
        return

    yield 0.05, f"视频《{title}》，开始爬取评论...", []

    # ── Step 2: 逐页爬取 ──
    all_comments: list[dict] = []
    page = 1

    while page <= max_pages:
        data = None
        last_error = None
        for attempt in range(1, PAGE_MAX_RETRIES + 1):
            try:
                resp = session.get(
                    "https://api.bilibili.com/x/v2/reply/main",
                    params={"type": 1, "mode": 3, "oid": aid, "next": page},
                    headers=headers,
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as e:
                last_error = e
                if attempt < PAGE_MAX_RETRIES:
                    yield (
                        min(0.05 + page / max_pages * 0.9, 0.99),
                        f"⚠️ 第 {page} 页请求失败，正在重试 {attempt}/{PAGE_MAX_RETRIES - 1}...",
                        all_comments,
                    )
                    time.sleep(PAGE_RETRY_SLEEP * attempt)

        if data is None:
            yield min(0.05 + page / max_pages * 0.9, 0.99), f"⚠️ 第 {page} 页多次请求失败：{last_error}，已停止爬取", all_comments
            break

        d = data.get("data") or {}
        cursor = d.get("cursor") or {}
        replies = d.get("replies") or []

        for rp in replies:
            parsed = _parse_reply(rp, video_meta)
            if parsed:
                all_comments.append(parsed)

        progress = min(0.05 + page / max_pages * 0.90, 0.99)
        yield (
            progress,
            f"正在爬取第 {page} 页，已获取 {len(all_comments)} 条评论...",
            all_comments,
        )

        if cursor.get("is_end") is True or not replies:
            break

        page += 1
        time.sleep(random.uniform(*PAGE_INTERVAL_RANGE))

    yield 1.0, f"✅ 爬取完成！共获取 {len(all_comments)} 条评论", all_comments
