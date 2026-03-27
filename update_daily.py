# update_daily.py
import os
import time
import requests
import feedparser
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

PORTWATCH_URL = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services"
    "/Daily_Chokepoints_Data/FeatureServer/0/query"
)
CHOKEPOINTS = {
    "chokepoint6" : "霍尔木兹海峡",
    "chokepoint4" : "曼德海峡",
    "chokepoint1" : "苏伊士运河",
    "chokepoint5" : "马六甲海峡",
    "chokepoint3" : "博斯普鲁斯海峡",
    "chokepoint11": "台湾海峡",
    "chokepoint7" : "好望角",
}

TRUSTED_SOURCES = {
    "Reuters", "Bloomberg", "Financial Times", "The Wall Street Journal",
    "Associated Press", "BBC News", "CNBC", "S&P Global", "Platts",
    "MarketWatch", "OilPrice.com", "Oil Price", "Rigzone", "Hart Energy",
    "Financial Post", "The Guardian", "CNN", "NBC News", "ABC News",
    "Yahoo Finance", "AP News", "EIA", "IEA", "World Oil",
    "Gulf News", "Al Jazeera", "Middle East Eye", "TASS",
    "Upstream Online", "Energy Intelligence", "Argus Media",
}

RSS_FEEDS = [
    ("EIA Official",        "https://www.eia.gov/rss/press_rss.xml"),
    ("EIA Today In Energy", "https://www.eia.gov/rss/todayinenergy.xml"),
    ("AP World",            "https://feeds.apnews.com/rss/apf-worldnews"),
    ("AP Business",         "https://feeds.apnews.com/rss/apf-business"),
    ("BBC Business",        "https://feeds.bbci.co.uk/news/business/rss.xml"),
    ("BBC World",           "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("CNBC Energy",         "https://www.cnbc.com/id/19836768/device/rss/rss.html"),
    ("MarketWatch",         "https://feeds.marketwatch.com/marketwatch/topstories/"),
    ("OilPrice.com",        "https://oilprice.com/rss/main"),
    ("Al Jazeera",          "https://www.aljazeera.com/xml/rss/all.xml"),
    ("Gulf News",           "https://gulfnews.com/rss/energy"),
    ("Rigzone",             "https://www.rigzone.com/news/rss/rigzone_latest.aspx"),
]

OIL_KEYWORDS = [
    "oil", "crude", "petroleum", "OPEC", "energy", "barrel",
    "WTI", "Brent", "refinery", "pipeline", "Iran", "Saudi",
    "Russia", "Iraq", "Venezuela", "Libya", "Trump", "sanction",
    "supply", "demand", "inventory", "EIA", "IEA", "drilling",
    "shale", "LNG", "gasoline", "fuel", "Middle East", "Gulf",
    "OPEC+", "production cut", "Hormuz", "chokepoint",
]

TIER1_KEYWORDS = [
    ("crude oil price",         20),
    ("OPEC production",         20),
    ("oil supply disruption",   20),
    ("Strait of Hormuz",        20),
    ("Iran sanctions oil",      20),
    ("Russia oil sanctions",    20),
    ("Black Sea oil tanker",    20),
    ("Ukraine Russia shipping", 15),
    ("Baltic Sea energy",       15),
]

TIER2_KEYWORDS = [
    ("Saudi Arabia oil",   10),
    ("Iraq oil OPEC",      10),
    ("oil demand outlook", 10),
    ("WTI Brent spread",   10),
    ("Red Sea shipping",   10),
    ("energy market",      10),
    ("oil refinery",       10),
    ("Venezuela oil",      10),
]

AIS_CHOKEPOINTS = {
    "霍尔木兹海峡" : {"min_lat": 25.5, "max_lat": 27.5, "min_lon": 55.5, "max_lon": 57.5,  "normal_count": 32, "importance": "全球20%石油贸易经过此处",    "coverage": True},
    "曼德海峡"     : {"min_lat": 11.0, "max_lat": 13.5, "min_lon": 42.5, "max_lon": 44.5,  "normal_count": 18, "importance": "红海通往印度洋的唯一通道",    "coverage": False},
    "苏伊士运河"   : {"min_lat": 29.5, "max_lat": 31.5, "min_lon": 32.0, "max_lon": 33.0,  "normal_count": 15, "importance": "欧洲与亚洲最短海上航线",      "coverage": False},
    "马六甲海峡"   : {"min_lat":  1.0, "max_lat":  4.0, "min_lon": 99.0, "max_lon": 104.0, "normal_count": 85, "importance": "中东原油运往亚洲的主要通道",   "coverage": True},
    "博斯普鲁斯海峡": {"min_lat": 40.5, "max_lat": 41.5, "min_lon": 28.5, "max_lon": 29.5, "normal_count": 12, "importance": "俄罗斯黑海原油出口唯一通道",  "coverage": True},
}


# ══════════════════════════════════════════════════════════════════════════
# 工具函数
# ══════════════════════════════════════════════════════════════════════════
def _safe_concat(a, b):
    a = a.loc[:, a.notna().any()]
    b = b.loc[:, b.notna().any()]
    return pd.concat([a, b])


def _merge_indexed(existing, new_df):
    if len(existing) == 0:
        return new_df
    combined = _safe_concat(existing, new_df)
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)
    return combined


def _merge_news(existing, new_df):
    combined = pd.concat([existing, new_df], ignore_index=True) if len(existing) > 0 else new_df
    combined = combined.drop_duplicates(subset=["title"])
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    cutoff   = datetime.today() - timedelta(days=90)
    combined = combined[combined["date"] >= cutoff]
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
    combined.sort_values("date", ascending=False, inplace=True)
    return combined


def _is_oil_related(title, summary=""):
    text = (title + " " + summary).lower()
    return any(kw.lower() in text for kw in OIL_KEYWORDS)


def _parse_rss_date(entry):
    for attr in ["published_parsed", "updated_parsed", "created_parsed"]:
        if hasattr(entry, attr) and getattr(entry, attr):
            try:
                t = getattr(entry, attr)
                return datetime(*t[:6]).strftime("%Y-%m-%d")
            except:
                pass
    return datetime.today().strftime("%Y-%m-%d")


# ══════════════════════════════════════════════════════════════════════════
# 油价
# ══════════════════════════════════════════════════════════════════════════
def update_oil_prices():
    print("\n[油价] 增量更新...")
    try:
        from fredapi import Fred
        fred     = Fred(api_key=FRED_API_KEY)
        out_path = os.path.join(ROOT_DIR, "data", "raw", "oil_prices.csv")

        if os.path.exists(out_path):
            existing = pd.read_csv(out_path, index_col=0, parse_dates=True)
            start    = (existing.index.max() + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            existing = pd.DataFrame()
            start    = "2020-01-01"

        print(f"  起始日期: {start}")
        frames = {}
        for name, code in [("WTI", "DCOILWTICO"), ("Brent", "DCOILBRENTEU")]:
            try:
                s = fred.get_series(code, observation_start=start)
                frames[name] = s
                print(f"  {name} ✓  新增 {len(s)} 条")
            except Exception as e:
                print(f"  {name} 失败: {e}")

        if not frames:
            print("  无新数据")
            return

        new_df       = pd.DataFrame(frames)
        new_df.index = pd.to_datetime(new_df.index)
        new_df       = new_df.dropna(how="all")
        combined     = _merge_indexed(existing, new_df)
        combined.to_csv(out_path)
        print(f"  完成，共 {len(combined)} 条")

    except Exception as e:
        print(f"  失败: {e}")


# ══════════════════════════════════════════════════════════════════════════
# 宏观数据
# ══════════════════════════════════════════════════════════════════════════
def update_macro_data():
    print("\n[宏观] 增量更新...")
    try:
        from fredapi import Fred
        fred     = Fred(api_key=FRED_API_KEY)
        out_path = os.path.join(ROOT_DIR, "data", "raw", "macro_data.csv")

        if os.path.exists(out_path):
            existing = pd.read_csv(out_path, index_col=0, parse_dates=True)
            start    = (existing.index.max() + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            existing = pd.DataFrame()
            start    = "2020-01-01"

        print(f"  起始日期: {start}")
        series = {
            "DXY"        : "DTWEXBGS",
            "US_CPI"     : "CPIAUCSL",
            "FED_RATE"   : "FEDFUNDS",
            "US10Y"      : "DGS10",
            "VIX"        : "VIXCLS",
            "US_EPU"     : "USEPUINDXD",
            "GLOBAL_EPU" : "GEPUCURRENT",
            "US_PPI"     : "PPIACO",
        }

        frames = {}
        for name, code in series.items():
            try:
                s = fred.get_series(code, observation_start=start)
                frames[name] = s
                print(f"  {name} ✓  新增 {len(s)} 条")
            except Exception as e:
                print(f"  {name} 失败: {e}")

        if not frames:
            print("  无新数据")
            return

        new_df       = pd.DataFrame(frames)
        new_df.index = pd.to_datetime(new_df.index)
        combined     = _merge_indexed(existing, new_df)
        combined.to_csv(out_path)
        print(f"  完成，共 {len(combined)} 条")

    except Exception as e:
        print(f"  失败: {e}")


# ══════════════════════════════════════════════════════════════════════════
# GDELT
# ══════════════════════════════════════════════════════════════════════════
def update_gdelt():
    print("\n[GDELT] 增量更新（最近7天）...")
    try:
        from fetch_gdelt import update_gdelt_recent
        update_gdelt_recent(days_back=7)
    except Exception as e:
        print(f"  失败: {e}")


# ══════════════════════════════════════════════════════════════════════════
# 新闻：NewsAPI
# ══════════════════════════════════════════════════════════════════════════
def update_news_api():
    print("\n[新闻-API] NewsAPI增量更新...")
    try:
        from newsapi import NewsApiClient
        newsapi    = NewsApiClient(api_key=NEWS_API_KEY)
        out_path   = os.path.join(ROOT_DIR, "data", "raw", "news_data.csv")
        end_date   = datetime.today()
        start_date = end_date - timedelta(days=3)
        existing   = pd.read_csv(out_path) if os.path.exists(out_path) else pd.DataFrame()

        all_articles = []
        all_keywords = list(TIER1_KEYWORDS) + list(TIER2_KEYWORDS)

        for keyword, limit in all_keywords:
            try:
                resp = newsapi.get_everything(
                    q          = keyword,
                    from_param = start_date.strftime("%Y-%m-%d"),
                    to         = end_date.strftime("%Y-%m-%d"),
                    language   = "en",
                    sort_by    = "relevancy",
                    page_size  = limit,
                )
                added = 0
                for a in resp.get("articles", []):
                    src = a["source"]["name"]
                    if src not in TRUSTED_SOURCES:
                        continue
                    all_articles.append({
                        "date"        : a["publishedAt"][:10],
                        "title"       : a["title"],
                        "description" : a.get("description", ""),
                        "source"      : src,
                        "keyword"     : keyword,
                        "url"         : a.get("url", ""),
                    })
                    added += 1
                print(f"  '{keyword}' ✓  {added} 条")
            except Exception as e:
                print(f"  '{keyword}' 失败: {e}")

        if not all_articles:
            print("  无新数据")
            return

        new_df   = pd.DataFrame(all_articles).drop_duplicates(subset=["title"])
        combined = _merge_news(existing, new_df)
        combined.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  NewsAPI完成，共 {len(combined)} 条")

    except Exception as e:
        print(f"  失败: {e}")


# ══════════════════════════════════════════════════════════════════════════
# 新闻：RSS
# ══════════════════════════════════════════════════════════════════════════
def update_news_rss():
    print("\n[新闻-RSS] RSS增量更新...")
    try:
        out_path = os.path.join(ROOT_DIR, "data", "raw", "news_data.csv")
        cutoff   = datetime.today() - timedelta(days=3)
        existing = pd.read_csv(out_path) if os.path.exists(out_path) else pd.DataFrame()
        articles = []

        for source_name, url in RSS_FEEDS:
            try:
                feed  = feedparser.parse(url)
                count = 0
                for entry in feed.entries:
                    title    = entry.get("title", "").strip()
                    summary  = entry.get("summary", entry.get("description", "")).strip()
                    date_str = _parse_rss_date(entry)
                    link     = entry.get("link", "")
                    try:
                        if datetime.strptime(date_str, "%Y-%m-%d") < cutoff:
                            continue
                    except:
                        pass
                    if not title or not _is_oil_related(title, summary):
                        continue
                    articles.append({
                        "date"        : date_str,
                        "title"       : title,
                        "description" : summary[:300] if summary else "",
                        "source"      : source_name,
                        "keyword"     : "rss",
                        "url"         : link,
                    })
                    count += 1
                print(f"  {source_name.ljust(22)} {count} 条 ✓")
            except Exception as e:
                print(f"  {source_name.ljust(22)} 失败: {str(e)[:40]}")
            time.sleep(0.5)

        if not articles:
            print("  无新数据")
            return

        new_df   = pd.DataFrame(articles).drop_duplicates(subset=["title"])
        combined = _merge_news(existing, new_df)
        combined.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  RSS完成，共 {len(combined)} 条")

    except Exception as e:
        print(f"  失败: {e}")


# ══════════════════════════════════════════════════════════════════════════
# 情感分析
# ══════════════════════════════════════════════════════════════════════════
def update_sentiment():
    print("\n[情感] 增量情感分析...")
    try:
        from sentiment_analysis import incremental_sentiment_analysis
        incremental_sentiment_analysis(max_articles=50)
    except Exception as e:
        print(f"  失败: {e}")


# ══════════════════════════════════════════════════════════════════════════
# PortWatch
# ══════════════════════════════════════════════════════════════════════════
def update_portwatch():
    print("\n[PortWatch] 增量更新...")
    try:
        out_path = os.path.join(ROOT_DIR, "data", "raw", "portwatch_chokepoints.csv")

        if os.path.exists(out_path):
            existing  = pd.read_csv(out_path, index_col=0, parse_dates=True)
            last_date = existing.index.max()
            start_dt  = last_date + timedelta(days=1)
            print(f"  现有数据截至 {last_date.date()}，从 {start_dt.date()} 开始增量")
        else:
            existing = pd.DataFrame()
            start_dt = datetime(2019, 1, 1)
            print("  无现有数据，全量下载")

        if start_dt.date() >= datetime.today().date():
            print("  已是最新，跳过")
            return

        start_ts = int(start_dt.timestamp() * 1000)
        end_ts   = int(datetime.today().timestamp() * 1000)
        headers  = {"User-Agent": "Mozilla/5.0"}
        all_dfs  = []

        for portid, name in CHOKEPOINTS.items():
            print(f"  [{portid}] {name}...", end=" ")
            records = []
            offset  = 0

            while True:
                params = {
                    "where"            : f"portid='{portid}' AND date >= {start_ts} AND date <= {end_ts}",
                    "outFields"        : "date,portid,portname,n_tanker,n_total,capacity_tanker,capacity",
                    "resultRecordCount": 2000,
                    "resultOffset"     : offset,
                    "orderByFields"    : "date ASC",
                    "f"                : "json",
                }
                try:
                    r = requests.get(PORTWATCH_URL, params=params, headers=headers, timeout=15)
                    d = r.json()
                except Exception as e:
                    print(f"请求失败: {e}")
                    break

                features = d.get("features", [])
                if not features:
                    break

                for feat in features:
                    a = feat["attributes"]
                    records.append({
                        "date"           : datetime.utcfromtimestamp(a.get("date", 0) / 1000).strftime("%Y-%m-%d"),
                        "n_tanker"       : a.get("n_tanker", 0),
                        "n_total"        : a.get("n_total", 0),
                        "capacity_tanker": a.get("capacity_tanker", 0),
                        "capacity_total" : a.get("capacity", 0),
                    })

                offset += 2000
                if not d.get("exceededTransferLimit", False):
                    break
                time.sleep(0.3)

            if not records:
                print("无新数据")
                continue

            df         = pd.DataFrame(records)
            df["date"] = pd.to_datetime(df["date"])
            df         = df.set_index("date").sort_index()
            short      = portid.replace("chokepoint", "cp")
            df         = df.rename(columns={
                "n_tanker"       : f"{short}_tanker",
                "n_total"        : f"{short}_total",
                "capacity_tanker": f"{short}_cap_tanker",
                "capacity_total" : f"{short}_cap_total",
            })
            df = df[[f"{short}_tanker", f"{short}_total",
                     f"{short}_cap_tanker", f"{short}_cap_total"]]
            all_dfs.append(df)
            print(f"✓ 新增 {len(df)} 条")

        if not all_dfs:
            print("  无新数据")
            return

        new_wide = pd.concat(all_dfs, axis=1)
        combined = _merge_indexed(existing, new_wide) if len(existing) > 0 else new_wide

        if "cp6_tanker" in combined.columns:
            combined["hormuz_tanker_ma7"]    = combined["cp6_tanker"].rolling(7).mean()
            combined["hormuz_tanker_zscore"] = (
                (combined["cp6_tanker"] - combined["cp6_tanker"].rolling(90).mean()) /
                (combined["cp6_tanker"].rolling(90).std() + 1e-6)
            )
            combined["hormuz_blocked"] = (
                combined["cp6_tanker"] < combined["cp6_tanker"].rolling(90).mean() * 0.5
            ).astype(int)

        if "cp4_tanker" in combined.columns:
            combined["mandeb_tanker_ma7"] = combined["cp4_tanker"].rolling(7).mean()
            combined["mandeb_blocked"]    = (
                combined["cp4_tanker"] < combined["cp4_tanker"].rolling(90).mean() * 0.5
            ).astype(int)

        if "cp7_tanker" in combined.columns and "cp6_tanker" in combined.columns:
            combined["cape_reroute_signal"] = (
                combined["cp7_tanker"].rolling(7).mean() /
                (combined["cp7_tanker"].rolling(90).mean() + 1e-6)
            )

        combined.to_csv(out_path)
        print(f"  完成，共 {len(combined)} 条，截至 {combined.index.max().date()}")

    except Exception as e:
        print(f"  失败: {e}")


# ══════════════════════════════════════════════════════════════════════════
# AIS快照
# ══════════════════════════════════════════════════════════════════════════
def update_ais():
    print("\n[AIS] 快照更新（120秒）...")
    try:
        import asyncio
        import websockets
        import json

        AIS_API_KEY = os.getenv("AIS_API_KEY", "")
        if not AIS_API_KEY:
            print("  AIS_API_KEY未配置，跳过")
            return

        out_path = os.path.join(ROOT_DIR, "data", "raw", "ais_snapshot.json")

        def assess_risk(count, normal_count):
            if normal_count == 0:
                return "未知", "#95a5a6"
            ratio = count / normal_count
            if ratio < 0.4:    return "极高风险", "#e74c3c"
            elif ratio < 0.7:  return "高风险",   "#e67e22"
            elif ratio < 0.85: return "中等风险", "#f1c40f"
            else:              return "正常",     "#2ecc71"

        async def fetch():
            url = "wss://stream.aisstream.io/v0/stream"
            bounding_boxes = [
                [[b["min_lat"], b["min_lon"]], [b["max_lat"], b["max_lon"]]]
                for b in AIS_CHOKEPOINTS.values()
            ]
            subscribe_msg = {
                "APIKey"            : AIS_API_KEY,
                "BoundingBoxes"     : bounding_boxes,
                "FilterMessageTypes": ["PositionReport"],
            }
            counts = {name: 0 for name in AIS_CHOKEPOINTS}
            try:
                async with websockets.connect(
                    url, ping_interval=20, ping_timeout=30, close_timeout=10
                ) as ws:
                    await ws.send(json.dumps(subscribe_msg))
                    start = time.time()
                    while time.time() - start < 120:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=5)
                            if isinstance(raw, bytes):
                                raw = raw.decode("utf-8")
                            data = json.loads(raw)
                            pos  = data.get("Message", {}).get("PositionReport", {})
                            lat  = pos.get("Latitude",  0)
                            lon  = pos.get("Longitude", 0)
                            for name, box in AIS_CHOKEPOINTS.items():
                                if (box["min_lat"] <= lat <= box["max_lat"] and
                                        box["min_lon"] <= lon <= box["max_lon"]):
                                    counts[name] += 1
                                    break
                        except asyncio.TimeoutError:
                            continue
            except Exception as e:
                print(f"  AIS连接错误: {e}")
            return counts

        try:
            import nest_asyncio
            nest_asyncio.apply()
            loop   = asyncio.get_event_loop()
            counts = loop.run_until_complete(fetch())
        except RuntimeError:
            counts = asyncio.run(fetch())

        results = {}
        for name, info in AIS_CHOKEPOINTS.items():
            count       = counts[name]
            risk, color = assess_risk(count, info["normal_count"])
            ratio       = round(count / info["normal_count"] * 100, 1) if info["normal_count"] else 0
            results[name] = {
                "count"        : count,
                "normal_count" : info["normal_count"],
                "risk"         : risk if info["coverage"] else "数据不足",
                "color"        : color if info["coverage"] else "#95a5a6",
                "importance"   : info["importance"],
                "ratio"        : ratio,
                "ais_coverage" : info["coverage"],
                "timestamp"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            print(f"  {name.ljust(10)}: {count:3d} 艘  ({ratio}%)  {risk}")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("  AIS快照已保存")

    except Exception as e:
        print(f"  失败: {e}")

def update_feature_matrix():
    print("[特征矩阵] 增量更新...")
    try:
        from train_model import load_and_merge, build_features, OPEC_DATES

        raw_df = load_and_merge()

        # 尝试用雅虎财经补充最新价格
        try:
            import yfinance as yf
            wti   = yf.Ticker("CL=F").history(period="5d")[["Close"]].rename(columns={"Close": "WTI"})
            brent = yf.Ticker("BZ=F").history(period="5d")[["Close"]].rename(columns={"Close": "Brent"})
            wti.index   = pd.to_datetime(wti.index).tz_localize(None)
            brent.index = pd.to_datetime(brent.index).tz_localize(None)
            yf_new = wti.join(brent, how="outer")
            yf_new = yf_new[yf_new.index > raw_df.index.max()]
            if len(yf_new) > 0:
                for col in raw_df.columns:
                    if col not in ["WTI", "Brent"]:
                        yf_new[col] = raw_df[col].iloc[-1]
                raw_df = pd.concat([raw_df, yf_new])
                raw_df = raw_df[~raw_df.index.duplicated(keep="last")]
                raw_df.sort_index(inplace=True)
                print(f"  雅虎财经补充 {len(yf_new)} 条，最新至 {raw_df.index.max().date()}")
            else:
                print(f"  雅虎财经无新数据，使用FRED数据截至 {raw_df.index.max().date()}")
        except Exception as e:
            print(f"  雅虎财经跳过: {e}")

        # 带target的完整矩阵（用于训练）
        feat, all_feature_cols, _ = build_features(raw_df, target_col="WTI", horizon=10)
        out_path = os.path.join(ROOT_DIR, "data", "processed", "feature_matrix.csv")
        feat.to_csv(out_path)
        print(f"  特征矩阵已更新，共 {len(feat)} 条，截至 {feat.index.max().date()}")

        # 最新特征行（无target，用于实时预测）
        feat_full = raw_df.copy()
        feat_full["return_1d"]   = feat_full["WTI"].pct_change(1)
        feat_full["return_5d"]   = feat_full["WTI"].pct_change(5)
        feat_full["return_10d"]  = feat_full["WTI"].pct_change(10)
        feat_full["return_20d"]  = feat_full["WTI"].pct_change(20)
        feat_full["ma_5"]        = feat_full["WTI"].rolling(5).mean()
        feat_full["ma_20"]       = feat_full["WTI"].rolling(20).mean()
        feat_full["ma_60"]       = feat_full["WTI"].rolling(60).mean()
        feat_full["ma_ratio"]    = feat_full["ma_5"] / feat_full["ma_20"]
        feat_full["ma_ratio_60"] = feat_full["ma_20"] / feat_full["ma_60"]
        feat_full["volatility"]  = feat_full["WTI"].rolling(10).std()
        feat_full["vol_ratio"]   = feat_full["volatility"] / feat_full["WTI"].rolling(60).std()
        feat_full["high_vol"]    = (
            feat_full["volatility"] > feat_full["volatility"].rolling(60).mean()
        ).astype(int)
        if "Brent" in feat_full.columns:
            feat_full["wti_brent_spread"] = feat_full["WTI"] - feat_full["Brent"]

        opec_dates = pd.to_datetime(OPEC_DATES)
        feat_full["opec_flag"] = 0
        for od in opec_dates:
            mask = (feat_full.index >= od - pd.Timedelta(days=5)) & \
                   (feat_full.index <= od + pd.Timedelta(days=5))
            feat_full.loc[mask, "opec_flag"] = 1

        if "gdelt_goldstein" in feat_full.columns:
            feat_full["gdelt_goldstein_chg"] = feat_full["gdelt_goldstein"].diff(5)
            feat_full["gdelt_conflict_ma5"]  = feat_full["gdelt_conflict_cnt"].rolling(5).mean()
            feat_full["gdelt_tone_chg"]      = feat_full["gdelt_tone"].diff(3)

        latest_row  = feat_full[all_feature_cols].ffill().dropna().tail(1)
        latest_path = os.path.join(ROOT_DIR, "data", "processed", "latest_features.csv")
        latest_row.to_csv(latest_path)
        print(f"  最新预测特征已保存，日期：{latest_row.index[-1].date()}")

    except Exception as e:
        print(f"  失败: {e}")


# ══════════════════════════════════════════════════════════════════════════
# 入口
# ══════════════════════════════════════════════════════════════════════════
def run_update():
    print("\n" + "="*50)
    print(f"OilSense 数据更新  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*50)
    update_oil_prices()
    update_macro_data()
    update_gdelt()
    update_news_api()
    update_news_rss()
    update_sentiment()
    update_portwatch()
    update_feature_matrix()
    # AIS单独手动跑，不在自动流程里（耗时120秒）
    # update_ais()
    print("\n" + "="*50)
    print("全部更新完成！")
    print("="*50 + "\n")


if __name__ == "__main__":
    run_update()
