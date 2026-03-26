import os
import time
import pandas as pd
from datetime import datetime, timedelta
from gdeltdoc import GdeltDoc, Filters
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

HISTORICAL_WINDOWS = [
    {"name": "新冠暴跌",   "start": "2020-02-01", "end": "2020-05-31"},
    {"name": "俄乌冲突",   "start": "2022-01-01", "end": "2022-04-30"},
    {"name": "以哈冲突",   "start": "2023-09-01", "end": "2023-11-30"},
    {"name": "特朗普就职", "start": "2024-12-01", "end": "2025-02-28"},
]

KEYWORDS    = ["crude oil", "OPEC", "oil price", "energy market"]
MAX_PER_CHUNK = 30   # 每段最多保留30条，控制总量和费用

def fetch_gdelt_chunk(keyword, start_str, end_str, max_retries=3):
    """拉取单个时间段，带重试机制"""
    for attempt in range(max_retries):
        try:
            gd = GdeltDoc()  # 每次新建实例，避免连接复用导致断线
            f  = Filters(
                keyword    = keyword,
                start_date = start_str,
                end_date   = end_str,
            )
            articles = gd.article_search(f)
            if articles is not None and len(articles) > 0:
                # 每段只保留前 MAX_PER_CHUNK 条
                return articles.head(MAX_PER_CHUNK)
            return pd.DataFrame()
        except Exception as e:
            wait = (attempt + 1) * 5
            print("      第" + str(attempt+1) + "次失败: " + str(e)[:60])
            print("      等待 " + str(wait) + " 秒后重试...")
            time.sleep(wait)
    print("      已达最大重试次数，跳过此段")
    return pd.DataFrame()

def fetch_gdelt_window(start_str, end_str, keyword):
    """按周分段拉取，每段之间等待足够长"""
    results = []
    start   = datetime.strptime(start_str, "%Y-%m-%d")
    end     = datetime.strptime(end_str,   "%Y-%m-%d")
    current = start

    while current < end:
        chunk_end = min(current + timedelta(days=6), end)
        cs = current.strftime("%Y-%m-%d")
        ce = chunk_end.strftime("%Y-%m-%d")

        print("    " + cs + " ~ " + ce, end=" ... ")
        df = fetch_gdelt_chunk(keyword, cs, ce)

        if len(df) > 0:
            df["keyword"] = keyword
            results.append(df)
            print(str(len(df)) + " 条 ✓")
        else:
            print("0 条")

        current = chunk_end + timedelta(days=1)
        time.sleep(5)  # 每段之间固定等5秒，避免触发限流

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()

def fetch_all_historical():
    all_articles = []

    for window in HISTORICAL_WINDOWS:
        print("正在获取【" + window["name"] + "】(" +
              window["start"] + " ~ " + window["end"] + ")")

        for keyword in KEYWORDS:
            print("  关键词: [" + keyword + "]")
            df = fetch_gdelt_window(window["start"], window["end"], keyword)
            if len(df) > 0:
                df["window"] = window["name"]
                all_articles.append(df)
            print("  [" + keyword + "] 完成，等待10秒...")
            time.sleep(10)  # 每个关键词之间等10秒

        print("【" + window["name"] + "】完成，等待15秒...")
        time.sleep(15)  # 每个窗口之间等15秒

    if not all_articles:
        print("未获取到任何数据")
        return

    combined = pd.concat(all_articles, ignore_index=True)

    # 统一列名
    title_col = next((c for c in combined.columns if "title" in c.lower()), None)
    url_col   = next((c for c in combined.columns if "url"   in c.lower()), None)
    date_col  = next((c for c in combined.columns
                      if "date" in c.lower() or "time" in c.lower()), None)

    out = pd.DataFrame()
    out["date"]        = pd.to_datetime(combined[date_col]).dt.strftime("%Y-%m-%d") if date_col else ""
    out["title"]       = combined[title_col] if title_col else ""
    out["description"] = combined[url_col]   if url_col   else ""
    out["source"]      = "GDELT"
    out["keyword"]     = combined["keyword"] if "keyword" in combined.columns else ""
    out["window"]      = combined["window"]  if "window"  in combined.columns else ""

    out.drop_duplicates(subset=["title"], inplace=True)
    out.dropna(subset=["title"], inplace=True)

    out_path = os.path.join(ROOT_DIR, "data", "raw", "historical_news.csv")
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("历史新闻保存完成：" + out_path)
    print("共 " + str(len(out)) + " 条（去重后）")
    print("各窗口数据量：")
    print(out.groupby("window")["title"].count().to_string())

if __name__ == "__main__":
    fetch_all_historical()
