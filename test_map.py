"""
地图热图快速测试工具
用法: python test_map.py [csv文件路径]
默认使用 data/map_test_data.csv
"""
import os
import sys
import webbrowser

import pandas as pd

# 直接复用主项目逻辑
from backend.sentiment_insights import build_china_map_rows, render_china_sentiment_map
from backend.utils import enrich_province_column

if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "map_test_data.csv")
    print(f"读取: {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"共 {len(df)} 条 | 列: {list(df.columns)}")

    # 兼容两种列名（ip_location / ip_province）
    if "ip_province" not in df.columns and "ip_location" in df.columns:
        df = enrich_province_column(df)
    elif "ip_province" not in df.columns:
        # 测试数据 ip_location 直接是短名（如"广东"），复制过去
        df["ip_province"] = df.get("ip_location", "未知")

    # 如果没有 sentiment 列就用 content 随机填充（仅供结构测试）
    if "sentiment" not in df.columns:
        print("WARNING: 无 sentiment 列，随机填充用于测试")
        import random
        df["sentiment"] = [random.choice(["positive", "negative", "neutral"]) for _ in range(len(df))]

    rows = build_china_map_rows(df)
    print(f"\n地图数据 ({len(rows)} 省):")
    for r in rows:
        print(f"  {r['province']:6s}  ratio={r['ratio']:.2f}  "
              f"pos={r['positive']} neg={r['negative']} neu={r['neutral']}")

    if not rows:
        print("ERROR: 无可用数据（需同时有正向和负向评论）")
        sys.exit(1)

    embed = render_china_sentiment_map(rows)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "map_test_output.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"<!DOCTYPE html><html><head><meta charset='utf-8'></head>"
                f"<body style='margin:0;padding:20px;background:#f5f7fa'>{embed}</body></html>")

    print(f"\n已生成: {out}")
    webbrowser.open(f"file:///{out.replace(os.sep, '/')}")
