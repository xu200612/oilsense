import os
import time
import feedparser
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

# ── RSS 数据源配置 ─────────────────────────────────────────────────────────
RSS_FEEDS = [
    # 能源官方
    ("EIA Official",        "https://www.eia.gov/rss/press_rss.xml"),
    ("EIA Today In Energy", "https://www.eia.gov/rss/todayinenergy.xml"),

    # 白宫替换为美联社政治新闻
    ("AP Politics",         "https://feeds.apnews.com/rss/apf-politics"),
    ("AP World",  "https://feeds.apnews.com/rss/apf-worldnews"),
    ("AP Business",         "https://feeds.apnews.com/rss/apf-business"),
    # 主流财经媒体
    ("BBC Business",        "https://feeds.bbci.co.uk/news/business/rss.xml"),
    ("BBC World",           "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("CNBC Energy",         "https://www.cnbc.com/id/19836768/device/rss/rss.html"),
    ("MarketWatch",         "https://feeds.marketwatch.com/marketwatch/topstories/"),
    ("Yahoo Finance",       "https://finance.yahoo.com/rss/topfinstories"),
    ("Yahoo Energy","https://finance.yahoo.com/rss/industry?ind=energy"),
    # 地缘政治 / 中东
    ("Al Jazeera",          "https://www.aljazeera.com/xml/rss/all.xml"),
    ("Middle East Eye",     "https://www.middleeasteye.net/rss"),
    ("Gulf News Energy",    "https://gulfnews.com/rss/energy"),
    ("Oil Price.com",       "https://oilprice.com/rss/main"),
    # 产油国媒体
    ("TASS Russia",         "https://tass.com/rss/v2.xml"),
    ("RT Business",         "https://www.rt.com/rss/business/"),

    # OPEC / 国际能源
    ("Platts Oil",          "https://www.spglobal.com/commodityinsights/en/rss-feed/oil"),
]

# ── 油价相关关键词 ─────────────────────────────────────────────────────────
OIL_KEYWORDS = [
    "oil", "crude", "petroleum", "OPEC", "energy",
    "barrel", "WTI", "Brent", "refinery", "pipeline",
    "Iran", "Saudi", "Russia", "Iraq", "Venezuela", "Libya",
    "Trump", "sanction", "tariff", "geopolit",
    "supply", "demand", "inventory", "EIA", "IEA",
    "drilling", "shale", "LNG", "gasoline", "fuel",
    "Middle East", "Gulf", "OPEC+", "production cut",
]

def is_oil_related(title, summary=""):
    text = (title + " " + summary).lower()
    return any(kw.lower() in text for kw in OIL_KEYWORDS)

def parse_date(entry):
    """统一解析各种日期格式"""
    for attr in ["published_parsed", "updated_parsed", "created_parsed"]:
        if hasattr(entry, attr) and getattr(entry, attr):
            try:
                t = getattr(entry, attr)
                return datetime(*t[:6]).strftime("%Y-%m-%d")
            except:
                pass
    return datetime.today().strftime("%Y-%m-%d")

def fetch_all_rss(days_back=3):
    """抓取所有 RSS 源，过滤油价相关新闻"""
    cutoff  = datetime.today() - timedelta(days=days_back)
    results = []

    print("正在抓取 " + str(len(RSS_FEEDS)) + " 个数据源...")
    print("-" * 50)

    for source_name, url in RSS_FEEDS:
        try:
            feed  = feedparser.parse(url)
            count = 0

            for entry in feed.entries:
                title    = entry.get("title", "").strip()
                summary  = entry.get("summary", entry.get("description", "")).strip()
                date_str = parse_date(entry)
                link     = entry.get("link", "")

                # 过滤日期
                try:
                    if datetime.strptime(date_str, "%Y-%m-%d") < cutoff:
                        continue
                except:
                    pass

                # 过滤关键词
                if not title or not is_oil_related(title, summary):
                    continue

                results.append({
                    "date"       : date_str,
                    "title"      : title,
                    "description": summary[:300] if summary else "",
                    "source"     : source_name,
                    "keyword"    : "rss_realtime",
                    "url"        : link
                })
                count += 1

            status = str(count) + " 条 ✓" if count > 0 else "无相关新闻"
            print("  " + source_name.ljust(22) + status)

        except Exception as e:
            print("  " + source_name.ljust(22) + "失败: " + str(e)[:40])

        time.sleep(0.8)

    return results

def update_news_data(articles):
    """合并新数据到 news_data.csv，保留最近90天"""
    news_path = os.path.join(ROOT_DIR, "data", "raw", "news_data.csv")

    if os.path.exists(news_path):
        existing = pd.read_csv(news_path)
    else:
        existing = pd.DataFrame()

    new_df = pd.DataFrame(articles)
    if len(new_df) == 0:
        print("无新数据写入")
        return existing

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined.drop_duplicates(subset=["title"], inplace=True)

    # 只保留最近90天，防止文件无限增大
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    cutoff = pd.Timestamp.today() - pd.Timedelta(days=90)
    combined = combined[combined["date"] >= cutoff]
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
    combined.sort_values("date", ascending=False, inplace=True)
    combined.to_csv(news_path, index=False, encoding="utf-8-sig")

    print("新闻数据已更新：共 " + str(len(combined)) +
          " 条（本次新增 " + str(len(new_df)) + " 条）")
    return combined

def print_summary(articles):
    """按来源打印统计"""
    if not articles:
        return
    df = pd.DataFrame(articles)
    print("-" * 50)
    print("各来源统计：")
    for src, cnt in df.groupby("source")["title"].count().items():
        print("  " + str(src).ljust(22) + str(cnt) + " 条")
    print("总计：" + str(len(df)) + " 条油价相关新闻")

if __name__ == "__main__":
    print("=" * 50)
    print("OilSense 实时新闻抓取")
    print("时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 50)

    articles = fetch_all_rss(days_back=3)
    print_summary(articles)

    if articles:
        update_news_data(articles)
    else:
        print("未抓取到任何相关新闻，请检查网络连接")

    print("=" * 50)
    print("完成！下一步运行 python sentiment_analysis.py 更新情感因子")
