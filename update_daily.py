# update_daily.py
import os
import re
import time
import requests
import feedparser
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))
LEGACY_ENV_PATH = os.path.join(
    os.path.dirname(ROOT_DIR),
    "OilSense 原油风险智能预警系统",
    "技术文档",
    "OilSense_源代码",
    ".env",
)
if os.path.exists(LEGACY_ENV_PATH):
    load_dotenv(dotenv_path=LEGACY_ENV_PATH)

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")
EIA_API_KEY  = os.getenv("EIA_API_KEY")

PORTWATCH_URL = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services"
    "/Daily_Chokepoints_Data/FeatureServer/0/query"
)


def _frozen_block_signal(series, trigger_ratio=0.5, release_ratio=0.6,
                         window=90, min_periods=20):
    """Detect blockade using the last normal pre-blockade baseline."""
    rolling_mean = series.rolling(window, min_periods=min_periods).mean()
    blocked_raw = series < rolling_mean * trigger_ratio
    blocked = []
    frozen_baseline = None

    for i, (dt, is_blocked) in enumerate(blocked_raw.items()):
        value = series.loc[dt]
        if pd.isna(value):
            blocked.append(0)
            continue

        if frozen_baseline is None:
            blocked.append(int(bool(is_blocked)))
            if bool(is_blocked):
                pre_idx = max(0, i - 1)
                baseline = series.iloc[max(0, pre_idx - window + 1): pre_idx + 1].dropna()
                if len(baseline) >= min_periods:
                    frozen_baseline = float(baseline.mean())
        else:
            blocked.append(int(value < frozen_baseline * trigger_ratio))
            if i >= 2:
                recent = series.iloc[i - 2: i + 1].dropna()
                if len(recent) == 3 and all(recent > frozen_baseline * release_ratio):
                    frozen_baseline = None

    return pd.Series(blocked, index=series.index, dtype="int64")


def _parse_portwatch_date(value):
    if isinstance(value, str):
        return pd.to_datetime(value, errors="coerce").strftime("%Y-%m-%d")
    return datetime.utcfromtimestamp((value or 0) / 1000).strftime("%Y-%m-%d")


def _fetch_yahoo_chart_daily(symbol, name, start_date=None, days_back=30):
    end_dt = datetime.now() + timedelta(days=1)
    if start_date is None:
        start_dt = datetime.now() - timedelta(days=days_back)
    else:
        start_dt = pd.to_datetime(start_date).to_pydatetime() - timedelta(days=3)

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": int(start_dt.timestamp()),
        "period2": int(end_dt.timestamp()),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    r = requests.get(url, params=params, headers=headers, timeout=15)
    r.raise_for_status()
    result = r.json()["chart"]["result"][0]
    timestamps = result.get("timestamp", [])
    quote = result.get("indicators", {}).get("quote", [{}])[0]
    closes = quote.get("close", [])
    if not timestamps or not closes:
        return pd.Series(dtype="float64", name=name)
    idx = pd.to_datetime(timestamps, unit="s").normalize()
    s = pd.Series(closes, index=idx, name=name).dropna()
    s = s[~s.index.duplicated(keep="last")]
    return s


def _fetch_yahoo_oil_prices(start_date=None):
    frames = {}
    for symbol, name in [("CL=F", "WTI"), ("BZ=F", "Brent")]:
        try:
            s = _fetch_yahoo_chart_daily(symbol, name, start_date=start_date)
            frames[name] = s
            print(f"  Yahoo {name} ✓  {len(s)} 条")
        except Exception as e:
            print(f"  Yahoo {name} 失败: {e}")
    if not frames:
        return pd.DataFrame()
    df = pd.DataFrame(frames).dropna(how="all")
    df.index = pd.to_datetime(df.index)
    return df

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

COUNTRY_KEYWORDS = {
    "美国"    : ["United States", "U.S.", "US", "America", "American", "Washington", "Trump", "Biden", "shale", "Permian"],
    "俄罗斯"  : ["Russia", "Russian", "Moscow", "Kremlin"],
    "沙特阿拉伯": ["Saudi", "Saudi Arabia", "Riyadh", "Aramco"],
    "伊拉克"  : ["Iraq", "Iraqi", "Baghdad", "Kirkuk", "Basra"],
    "伊朗"    : ["Iran", "Iranian", "Tehran", "Hormuz"],
    "阿联酋"  : ["UAE", "United Arab Emirates", "Dubai", "Abu Dhabi"],
    "科威特"  : ["Kuwait", "Kuwaiti"],
    "挪威"    : ["Norway", "Norwegian", "Equinor"],
    "哈萨克斯坦": ["Kazakhstan", "Kazakh", "CPC pipeline"],
    "尼日利亚": ["Nigeria", "Nigerian"],
    "利比亚"  : ["Libya", "Libyan", "Tripoli"],
    "委内瑞拉": ["Venezuela", "Venezuelan", "Maduro"],
    "阿尔及利亚": ["Algeria", "Algerian", "Sonatrach"],
}

COUNTRY_NEWS_QUERIES = [
    ("美国", "United States oil OR shale oil OR crude oil"),
    ("俄罗斯", "Russia oil OR Russian crude OR energy sanctions"),
    ("沙特阿拉伯", "Saudi Arabia oil OR Aramco OR OPEC"),
    ("伊拉克", "Iraq oil OR Basra crude OR OPEC"),
    ("伊朗", "Iran oil OR Hormuz OR sanctions"),
    ("阿联酋", "UAE oil OR Abu Dhabi oil OR ADNOC"),
    ("科威特", "Kuwait oil OR Kuwaiti crude"),
    ("挪威", "Norway oil OR Equinor OR North Sea oil"),
    ("哈萨克斯坦", "Kazakhstan oil OR CPC pipeline OR Kazakh crude"),
    ("尼日利亚", "Nigeria oil OR Nigerian crude"),
    ("利比亚", "Libya oil OR Libyan crude"),
    ("委内瑞拉", "Venezuela oil OR Venezuelan crude OR Maduro"),
    ("阿尔及利亚", "Algeria oil OR Sonatrach OR Algerian gas"),
]

COUNTRY_PRODUCTION_BASELINE = {
    "美国": ("USA", 13.2),
    "俄罗斯": ("RUS", 10.3),
    "沙特阿拉伯": ("SAU", 9.7),
    "伊拉克": ("IRQ", 4.3),
    "伊朗": ("IRN", 3.4),
    "阿联酋": ("ARE", 3.3),
    "科威特": ("KWT", 2.7),
    "挪威": ("NOR", 1.8),
    "哈萨克斯坦": ("KAZ", 1.8),
    "尼日利亚": ("NGA", 1.5),
    "利比亚": ("LBY", 0.9),
    "委内瑞拉": ("VEN", 0.9),
    "阿尔及利亚": ("DZA", 0.9),
}
WORLD_OIL_SUPPLY_MBD = 102.5

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
    combined = new_df.combine_first(existing)
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
    text = (str(title or "") + " " + str(summary or "")).lower()
    return any(kw.lower() in text for kw in OIL_KEYWORDS)


def _detect_country_focus(title, summary=""):
    text = (str(title or "") + " " + str(summary or "")).lower()
    hits = []
    for country, keywords in COUNTRY_KEYWORDS.items():
        for kw in keywords:
            pattern = r"\b" + re.escape(kw.lower()) + r"\b"
            if re.search(pattern, text):
                hits.append(country)
                break
    return ";".join(hits)


def _tag_country_focus(df):
    if len(df) == 0:
        return df
    if "country_focus" not in df.columns:
        df["country_focus"] = ""
    for idx, row in df.iterrows():
        current = row.get("country_focus", "")
        if pd.notna(current) and str(current).strip():
            continue
        df.at[idx, "country_focus"] = _detect_country_focus(
            str(row.get("title", "")),
            str(row.get("description", "")),
        )
    return df


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
        out_path = os.path.join(ROOT_DIR, "data", "raw", "oil_prices.csv")

        if os.path.exists(out_path):
            existing = pd.read_csv(out_path, index_col=0, parse_dates=True)
            start    = (existing.index.max() + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            existing = pd.DataFrame()
            start    = "2020-01-01"

        print(f"  起始日期: {start}")
        frames = {}
        if FRED_API_KEY:
            try:
                from fredapi import Fred
                fred = Fred(api_key=FRED_API_KEY)
                for name, code in [("WTI", "DCOILWTICO"), ("Brent", "DCOILBRENTEU")]:
                    try:
                        s = fred.get_series(code, observation_start=start)
                        frames[name] = s
                        print(f"  FRED {name} ✓  新增 {len(s)} 条")
                    except Exception as e:
                        print(f"  FRED {name} 失败: {e}")
            except Exception as e:
                print(f"  FRED跳过: {e}")
        else:
            print("  FRED_API_KEY未配置，跳过FRED")

        new_df = pd.DataFrame(frames).dropna(how="all") if frames else pd.DataFrame()
        if len(new_df) > 0:
            new_df.index = pd.to_datetime(new_df.index)

        yahoo_df = _fetch_yahoo_oil_prices(start_date=start)
        if len(yahoo_df) > 0:
            new_df = yahoo_df.combine_first(new_df) if len(new_df) > 0 else yahoo_df

        if len(new_df) == 0:
            print("  无新数据")
            return

        combined     = _merge_indexed(existing, new_df)
        combined.to_csv(out_path)
        print(f"  完成，共 {len(combined)} 条，截至 {combined.index.max().date()}")

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
                series_start = start
                if len(existing) > 0:
                    first_existing = existing.index.min().strftime("%Y-%m-%d")
                    needs_backfill = (
                        name not in existing.columns or
                        existing[name].dropna().empty or
                        existing[name].first_valid_index() > existing.index.min() + pd.Timedelta(days=45)
                    )
                    if needs_backfill:
                        series_start = first_existing
                s = fred.get_series(code, observation_start=series_start)
                frames[name] = s
                label = "回填/更新" if series_start != start else "新增"
                print(f"  {name} ✓  {label} {len(s)} 条")
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
                        "country_focus": _detect_country_focus(a["title"], a.get("description", "")),
                    })
                    added += 1
                print(f"  '{keyword}' ✓  {added} 条")
            except Exception as e:
                print(f"  '{keyword}' 失败: {e}")

        for country, query in COUNTRY_NEWS_QUERIES:
            try:
                resp = newsapi.get_everything(
                    q          = query,
                    from_param = start_date.strftime("%Y-%m-%d"),
                    to         = end_date.strftime("%Y-%m-%d"),
                    language   = "en",
                    sort_by    = "relevancy",
                    page_size  = 20,
                )
                added = 0
                for a in resp.get("articles", []):
                    title = a.get("title", "")
                    desc  = a.get("description", "")
                    if not title or not _is_oil_related(title, desc):
                        continue
                    src = a["source"]["name"]
                    focus = _detect_country_focus(title, desc)
                    if country not in focus:
                        focus = (focus + ";" if focus else "") + country
                    all_articles.append({
                        "date"        : a["publishedAt"][:10],
                        "title"       : title,
                        "description" : desc,
                        "source"      : src,
                        "keyword"     : "country:" + country,
                        "url"         : a.get("url", ""),
                        "country_focus": focus,
                    })
                    added += 1
                print(f"  '{country}' 产油国新闻 ✓  {added} 条")
            except Exception as e:
                print(f"  '{country}' 产油国新闻失败: {e}")

        if not all_articles:
            print("  无新数据")
            return

        new_df   = pd.DataFrame(all_articles).drop_duplicates(subset=["title"])
        combined = _merge_news(existing, new_df)
        combined = _tag_country_focus(combined)
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
                        "country_focus": _detect_country_focus(title, summary),
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
        combined = _tag_country_focus(combined)
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
        incremental_sentiment_analysis(max_articles=150)
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

        start_str = start_dt.strftime("%Y-%m-%d")
        end_str   = datetime.today().strftime("%Y-%m-%d")
        headers  = {"User-Agent": "Mozilla/5.0"}
        all_dfs  = []

        for portid, name in CHOKEPOINTS.items():
            print(f"  [{portid}] {name}...", end=" ")
            records = []
            offset  = 0

            while True:
                params = {
                    "where"            : f"portid='{portid}' AND date >= DATE '{start_str}' AND date <= DATE '{end_str}'",
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
                        "date"           : _parse_portwatch_date(a.get("date")),
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
            combined["hormuz_blocked"] = _frozen_block_signal(combined["cp6_tanker"])

        if "cp4_tanker" in combined.columns:
            combined["mandeb_tanker_ma7"] = combined["cp4_tanker"].rolling(7).mean()
            combined["mandeb_blocked"]    = _frozen_block_signal(combined["cp4_tanker"])

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
                    url, open_timeout=35, ping_interval=20, ping_timeout=30,
                    close_timeout=10, proxy=None
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
                return None
            return counts

        try:
            import nest_asyncio
            nest_asyncio.apply()
            loop   = asyncio.get_event_loop()
            counts = loop.run_until_complete(fetch())
        except RuntimeError:
            counts = asyncio.run(fetch())

        if counts is None:
            print("  AIS快照未更新：连接失败，保留上一次有效快照")
            return

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


def update_country_production_data():
    print("\n[产油国实时/准实时产量] 更新...")
    out_path = os.path.join(ROOT_DIR, "data", "raw", "country_production.csv")
    rows = []
    api_key = os.getenv("EIA_API_KEY", "")
    session = requests.Session()
    session.trust_env = False

    if api_key:
        for country_cn, (eia_code, baseline) in COUNTRY_PRODUCTION_BASELINE.items():
            got = False
            try:
                r = session.get(
                    "https://api.eia.gov/v2/international/data/",
                    params={
                        "api_key": api_key,
                        "frequency": "monthly",
                        "data[0]": "value",
                        "facets[countryRegionId][]": eia_code,
                        "facets[productId][]": "55",
                        "sort[0][column]": "period",
                        "sort[0][direction]": "desc",
                        "offset": 0,
                        "length": 20,
                    },
                    headers={"User-Agent": "OilSenseAcademicPrototype/1.0"},
                    timeout=25,
                    verify=False,
                )
                if r.status_code != 200:
                    print(f"  {country_cn}/EIA HTTP {r.status_code}: {r.text[:120]}")
                else:
                    data = r.json().get("response", {}).get("data", [])
                    production_rows = [
                        x for x in data
                        if "production" in str(x.get("activityName", x.get("activityId", ""))).lower()
                    ]
                    item = production_rows[0] if production_rows else (data[0] if data else None)
                    if item:
                        value = float(item.get("value", baseline))
                        mbd = value / 1000 if value > 100 else value
                        rows.append({
                            "country": country_cn,
                            "eia_code": eia_code,
                            "production_mbd": round(mbd, 3),
                            "period": item.get("period", ""),
                            "source": "EIA International monthly API",
                            "status": "准实时（月度官方）",
                            "activity": item.get("activityName", item.get("activityId", "")),
                            "product": item.get("productName", item.get("productId", "")),
                            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        })
                        got = True
            except Exception as e:
                print(f"  {country_cn}/EIA失败: {str(e)[:100]}")

            if not got:
                rows.append({
                    "country": country_cn,
                    "eia_code": eia_code,
                    "production_mbd": baseline,
                    "period": "baseline",
                    "source": "OilSense baseline; EIA query failed or returned no production row",
                    "status": "EIA回退基线",
                    "activity": "",
                    "product": "",
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                })
            time.sleep(0.2)
    else:
        print("  未配置 EIA_API_KEY，写入可解释基线；页面会标注非实时")
        for country_cn, (eia_code, baseline) in COUNTRY_PRODUCTION_BASELINE.items():
            rows.append({
                "country": country_cn,
                "eia_code": eia_code,
                "production_mbd": baseline,
                "period": "baseline",
                "source": "OilSense baseline; configure EIA_API_KEY for EIA monthly official update",
                "status": "待配置EIA_API_KEY",
                "activity": "",
                "product": "",
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })

    df = pd.DataFrame(rows)
    df["share_pct"] = (df["production_mbd"] / WORLD_OIL_SUPPLY_MBD * 100).round(2)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    official = int((df["status"] == "准实时（月度官方）").sum())
    print(f"  产油国产量表已保存：{len(df)} 个国家，EIA官方 {official} 个")


def update_feature_matrix():
    print("[特征矩阵] 增量更新...")
    try:
        from train_model import load_and_merge, build_features, OPEC_DATES

        raw_df = load_and_merge()

        # 尝试用 Yahoo Chart API 补充最新价格，避免 yfinance 本地缓存报错
        try:
            yf_new = _fetch_yahoo_oil_prices(start_date=raw_df.index.max())
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

        # 近期补充特征（无target）：feat_full已含完整特征，但最后10行无法算target
        # 保存这批行供回测展示延伸到今日，不参与训练
        recent_mask = feat_full.index > feat.index.max()
        if recent_mask.any():
            _save_cols = [c for c in all_feature_cols if c in feat_full.columns]
            for _col in ['WTI', 'Brent']:
                if _col in feat_full.columns and _col not in _save_cols:
                    _save_cols = [_col] + _save_cols
            recent_rows = feat_full.loc[recent_mask, _save_cols].ffill().copy()
            recent_rows['target'] = float('nan')
            recent_path = os.path.join(ROOT_DIR, "data", "processed", "recent_features.csv")
            recent_rows.to_csv(recent_path)
            print(f"  近期补充特征：{len(recent_rows)} 条，{recent_rows.index[0].date()} ~ {recent_rows.index[-1].date()}")

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


def update_shipping():
    """ShipFinder/VesselAPI 双源 AIS 实时快照（API Key 未配置时自动跳过）"""
    print("\n[AIS实时] ShipFinder/VesselAPI 航运快照...")
    try:
        from shipping_sources import fetch_realtime_shipping_snapshot
        fetch_realtime_shipping_snapshot()
    except Exception as e:
        print(f"  ShipFinder/VesselAPI 跳过: {e}")


def update_shap():
    """TreeSHAP 因子贡献计算（生成展示用 CSV，失败不影响主流程）"""
    print("\n[SHAP] 更新因子归因解释...")
    try:
        from shap_explain import compute_shap_outputs
        compute_shap_outputs()
    except Exception as e:
        print(f"  SHAP 更新失败: {e}")


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
    update_country_production_data()
    update_feature_matrix()
    update_shipping()   # ShipFinder/VesselAPI（API Key 未配置时跳过）
    update_shap()       # TreeSHAP 因子归因（依赖 feature_matrix 更新后运行）
    # 原 AIS WebSocket（耗时120秒，手动跑）
    # update_ais()
    print("\n" + "="*50)
    print("全部更新完成！")
    print("="*50 + "\n")


if __name__ == "__main__":
    run_update()
