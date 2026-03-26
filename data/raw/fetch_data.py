import os
import pandas as pd
from fredapi import Fred
from newsapi import NewsApiClient
from datetime import datetime, timedelta
from dotenv import load_dotenv

# ── 定位根目录，加载 .env ──────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

for key in ["SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"]:
    os.environ.pop(key, None)

NEWS_API_KEY = os.getenv("NEWS_API_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

print(f"FRED Key: {'✓' if FRED_API_KEY else '✗'}  NEWS Key: {'✓' if NEWS_API_KEY else '✗'}")

# ── 1. 油价数据（全部从 FRED 获取，更稳定）────────────────────────────────
def fetch_oil_prices(start="2020-01-01"):
    print("\n正在获取油价数据...")
    fred = Fred(api_key=FRED_API_KEY)

    oil_series = {
        "WTI"   : "DCOILWTICO",   # WTI 原油现货价格
        "Brent" : "DCOILBRENTEU", # Brent 原油现货价格
    }

    frames = {}
    for name, code in oil_series.items():
        try:
            s = fred.get_series(code, observation_start=start)
            frames[name] = s
            print(f"  {name} ✓ ({len(s)} 条)")
        except Exception as e:
            print(f"  {name} 获取失败: {e}")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df.dropna(how="all", inplace=True)
    out_path = os.path.join(ROOT_DIR, "data", "raw", "oil_prices.csv")
    df.to_csv(out_path)
    print(f"  油价数据保存完成，共 {len(df)} 条记录")
    return df

# ── 2. 宏观经济数据（FRED）────────────────────────────────────────────────
def fetch_macro_data(start="2020-01-01"):
    print("\n正在获取宏观数据...")
    fred = Fred(api_key=FRED_API_KEY)

    series = {
        "DXY": "DTWEXBGS",  # 美元指数
        "US_CPI": "CPIAUCSL",  # 美国CPI
        "FED_RATE": "FEDFUNDS",  # 联邦基金利率
        "US10Y": "DGS10",  # 10年期美债收益率
        "VIX": "VIXCLS",  # 恐慌指数
        "GPR": "GPR_USA",   # 地缘政治风险指数（正确ID）
        "US_EPU": "USEPUINDXD",  # 美国经济政策不确定性
        "GLOBAL_EPU" : "GEPUCURRENT", # 全球经济不确定性（正确ID）
    }

    frames = {}
    for name, code in series.items():
        try:
            s = fred.get_series(code, observation_start=start)
            frames[name] = s
            print(f"  {name} ✓")
        except Exception as e:
            print(f"  {name} 获取失败: {e}")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    out_path = os.path.join(ROOT_DIR, "data", "raw", "macro_data.csv")
    df.to_csv(out_path)
    print(f"  宏观数据保存完成，共 {len(df)} 条记录")
    return df
def fetch_news(days_back=25):
    print("\n正在获取新闻数据...")
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)

    end_date   = datetime.today()
    start_date = end_date - timedelta(days=days_back)

    all_articles = []

    # 全局能源关键词
    global_keywords = [
        "crude oil", "OPEC", "oil price", "energy market",
        "Trump energy", "oil supply", "petroleum",
    ]

    # 国家维度关键词
    country_keywords = [
        "Iran oil", "Iran sanctions", "Iran war",
        "Russia oil", "Russia energy", "Russia sanctions",
        "Saudi Arabia oil", "Saudi Aramco",
        "Iraq oil", "Iraq OPEC",
        "Libya oil", "Libya conflict",
        "Venezuela oil", "Venezuela sanctions",
        "Nigeria oil", "Nigeria pipeline",
        "Strait of Hormuz", "Red Sea shipping",
        "OPEC production cut",
    ]

    all_keywords = global_keywords + country_keywords

    for keyword in all_keywords:
        try:
            response = newsapi.get_everything(
                q          = keyword,
                from_param = start_date.strftime("%Y-%m-%d"),
                to         = end_date.strftime("%Y-%m-%d"),
                language   = "en",
                sort_by    = "publishedAt",
                page_size  = 100,   # 从20提升到100
            )
            articles = response.get("articles", [])
            for a in articles:
                all_articles.append({
                    "date"       : a["publishedAt"][:10],
                    "title"      : a["title"],
                    "description": a.get("description", ""),
                    "source"     : a["source"]["name"],
                    "keyword"    : keyword,
                    "url"        : a.get("url", ""),
                })
            print(f"  '{keyword}' → {len(articles)} 条")
        except Exception as e:
            print(f"  '{keyword}' 获取失败: {e}")

    df = pd.DataFrame(all_articles).drop_duplicates(subset=["title"])
    out_path = os.path.join(ROOT_DIR, "data", "raw", "news_data.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  新闻数据保存完成，共 {len(df)} 条（去重后）")
    return df
# ── 主程序 ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fetch_oil_prices(start="2020-01-01")
    fetch_macro_data(start="2020-01-01")
    fetch_news(days_back=25)
    print("\n所有数据获取完成！")
