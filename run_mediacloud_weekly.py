"""按7天窗口重新抓取 MediaCloud 历史新闻（2022-2026），获得更好的日期分布。"""
import os, sys

SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "OilSense 原油风险智能预警系统",
    "技术文档",
    "OilSense_源代码",
)
sys.path.insert(0, SRC)
os.chdir(SRC)

from fetch_mediacloud_news import (
    _window_starts, fetch_query, _is_relevant_story,
    QUERY_GROUPS, _collection_ids_from_env
)
import mediacloud.api
import pandas as pd
import time
from datetime import date
from dotenv import load_dotenv

load_dotenv(os.path.join(SRC, ".env"))
api_key = os.getenv("MEDIACLOUD_API_KEY", "")
if not api_key:
    raise RuntimeError("MEDIACLOUD_API_KEY 未配置")

collection_ids = _collection_ids_from_env()
mc = mediacloud.api.SearchApi(api_key)
mc.TIMEOUT_SECS = 25

start = date(2022, 1, 1)
end   = date(2026, 3, 1)
window_days = 7
max_per_window = 14   # 60/30*7 ≈ 14 篇/周
sleep_s = 0.3

queries = [(name, q, "") for name, q in QUERY_GROUPS.items()
           if not name.endswith("_body")]   # skip body query

all_rows = []
windows = list(_window_starts(start, end, window_days))
print(f"共 {len(windows)} 个周窗口，collections: {collection_ids}")

for i, (ws, we) in enumerate(windows):
    print(f"[{i+1}/{len(windows)}] {ws} ~ {we}", end="  ", flush=True)
    week_rows = 0
    for keyword, query, country in queries:
        try:
            rows = fetch_query(mc, query, keyword, ws, we, collection_ids,
                               max_per_window, 50, sleep_s, 2)
            all_rows.extend(rows)
            week_rows += len(rows)
        except Exception as exc:
            print(f"ERR({keyword}:{exc!s:.60})", end=" ")
        time.sleep(sleep_s)
    print(f"→ {week_rows}条")

out = os.path.join(
    os.path.dirname(os.path.dirname(SRC)),
    "github_oilsense", "data", "raw", "mediacloud_history_weekly.csv"
)
# 实际路径：同级的 github_oilsense/data/raw/
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "data", "raw", "mediacloud_history_weekly.csv")

df = pd.DataFrame(all_rows)
if len(df):
    df = df.drop_duplicates(subset=["url", "title", "date"], keep="first")
    df = df.sort_values(["date", "source", "title"])

df.to_csv(out, index=False, encoding="utf-8-sig")
print(f"\n完成！共 {len(df)} 条  →  {out}")

# 按年日期分布
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"])
print("按年文章数:", df.groupby(df.date.dt.year).size().to_string())
unique_days = df.groupby(df.date.dt.date).ngroups
print(f"独立日期数: {unique_days}")
