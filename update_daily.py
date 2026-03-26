# update_daily.py
import os
import time
import requests
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


# ── 工具函数 ──────────────────────────────────────────────────────────────
def _safe_concat(a, b):
    a = a.loc[:, a.notna().any()]
    b = b.loc[:, b.notna().any()]
    return pd.concat([a, b])


def _merge(existing, new_df, by_index=True):
    if len(existing) == 0:
        return new_df
    combined = _safe_concat(existing, new_df) if by_index else pd.concat([existing, new_df], ignore_index=True)
    if by_index:
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
    return combined


# ── 油价 ──────────────────────────────────────────────────────────────────
def update_oil_prices():
    print("\n[油价] 增量更新...")
    try:
        from fredapi import Fred
        fred     = Fred(api_key=FRED_API_KEY)
        out_path = os.path.join(ROOT_DIR, "data", "raw", "oil_prices.csv")

        if os.path.exists(out_path):
            existing  = pd.read_csv(out_path, index_col=0, parse_dates=True)
            start     = (existing.index.max() + timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            existing  = pd.DataFrame()
            start     = "2020-01-01"

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
        combined     = _merge(existing, new_df)
        combined.to_csv(out_path)
        print(f"  完成，共 {len(combined)} 条")

    except Exception as e:
        print(f"  失败: {e}")


# ── 宏观数据 ──────────────────────────────────────────────────────────────
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
        combined     = _merge(existing, new_df)
        combined.to_csv(out_path)
        print(f"  完成，共 {len(combined)} 条")

    except Exception as e:
        print(f"  失败: {e}")


# ── 新闻 ──────────────────────────────────────────────────────────────────
def update_news():
    print("\n[新闻] 增量更新（最近3天）...")
    try:
        from newsapi import NewsApiClient
        newsapi    = NewsApiClient(api_key=NEWS_API_KEY)
        out_path   = os.path.join(ROOT_DIR, "data", "raw", "news_data.csv")
        end_date   = datetime.today()
        start_date = end_date - timedelta(days=3)

        existing = pd.read_csv(out_path) if os.path.exists(out_path) else pd.DataFrame()

        keywords = [
            "crude oil", "OPEC", "oil price", "energy market",
            "Trump energy", "oil supply", "petroleum",
            "Iran oil", "Iran sanctions",
            "Russia oil", "Russia energy",
            "Saudi Arabia oil", "Saudi Aramco",
            "Strait of Hormuz", "Red Sea shipping",
        ]

        all_articles = []
        for keyword in keywords:
            try:
                resp = newsapi.get_everything(
                    q          = keyword,
                    from_param = start_date.strftime("%Y-%m-%d"),
                    to         = end_date.strftime("%Y-%m-%d"),
                    language   = "en",
                    sort_by    = "publishedAt",
                    page_size  = 100,
                )
                for a in resp.get("articles", []):
                    all_articles.append({
                        "date"        : a["publishedAt"][:10],
                        "title"       : a["title"],
                        "description" : a.get("description", ""),
                        "source"      : a["source"]["name"],
                        "keyword"     : keyword,
                        "url"         : a.get("url", ""),
                    })
                print(f"  '{keyword}' ✓")
            except Exception as e:
                print(f"  '{keyword}' 失败: {e}")

        if not all_articles:
            print("  无新数据")
            return

        new_df   = pd.DataFrame(all_articles).drop_duplicates(subset=["title"])
        combined = pd.concat([existing, new_df], ignore_index=True) if len(existing) > 0 else new_df
        combined = combined.drop_duplicates(subset=["title"])
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
        combined = combined[combined["date"] >= datetime.today() - timedelta(days=90)]
        combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
        combined.sort_values("date", ascending=False, inplace=True)
        combined.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"  完成，共 {len(combined)} 条")

    except Exception as e:
        print(f"  失败: {e}")


# ── PortWatch ─────────────────────────────────────────────────────────────
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

        all_dfs = []
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
        combined = _merge(existing, new_wide) if len(existing) > 0 else new_wide

        # 重新计算衍生特征
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

# ── AIS快照 ───────────────────────────────────────────────────────────────
def update_ais():
    AIS_COVERAGE = {
        "霍尔木兹海峡": True,
        "曼德海峡": False,  # 无AIS基站
        "苏伊士运河": False,  # 无AIS基站
        "马六甲海峡": True,
        "博斯普鲁斯海峡": True,
    }

    print("[AIS] 快照更新（120秒）...")
    try:
        import asyncio
        import websockets
        import json

        AIS_API_KEY = os.getenv("AIS_API_KEY", "")
        if not AIS_API_KEY:
            print("  AIS_API_KEY未配置，跳过")
            return

        out_path = os.path.join(ROOT_DIR, "data", "raw", "ais_snapshot.json")

        CHOKEPOINTS = {
            "霍尔木兹海峡": {
                "min_lat": 25.5, "max_lat": 27.5,
                "min_lon": 55.5, "max_lon": 57.5,
                "normal_count": 32,
                "importance": "全球20%石油贸易经过此处",
            },
            "曼德海峡": {
                "min_lat": 11.0, "max_lat": 13.5,
                "min_lon": 42.5, "max_lon": 44.5,
                "normal_count": 18,
                "importance": "红海通往印度洋的唯一通道",
            },
            "苏伊士运河": {
                "min_lat": 29.5, "max_lat": 31.5,
                "min_lon": 32.0, "max_lon": 33.0,
                "normal_count": 15,
                "importance": "欧洲与亚洲最短海上航线",
            },
            "马六甲海峡": {
                "min_lat":  1.0, "max_lat":  4.0,
                "min_lon": 99.0, "max_lon": 104.0,
                "normal_count": 85,
                "importance": "中东原油运往亚洲的主要通道",
            },
            "博斯普鲁斯海峡": {
                "min_lat": 40.5, "max_lat": 41.5,
                "min_lon": 28.5, "max_lon": 29.5,
                "normal_count": 12,
                "importance": "俄罗斯黑海原油出口唯一通道",
            },
        }

        def assess_risk(count, normal_count):
            if normal_count == 0:
                return "未知", "#95a5a6"
            ratio = count / normal_count
            if ratio < 0.4:
                return "极高风险", "#e74c3c"
            elif ratio < 0.7:
                return "高风险",   "#e67e22"
            elif ratio < 0.85:
                return "中等风险", "#f1c40f"
            else:
                return "正常",     "#2ecc71"

        async def fetch():
            url = "wss://stream.aisstream.io/v0/stream"
            bounding_boxes = [
                [[b["min_lat"], b["min_lon"]], [b["max_lat"], b["max_lon"]]]
                for b in CHOKEPOINTS.values()
            ]
            subscribe_msg = {
                "APIKey"            : AIS_API_KEY,
                "BoundingBoxes"     : bounding_boxes,
                "FilterMessageTypes": ["PositionReport"],
            }

            counts = {name: 0 for name in CHOKEPOINTS}

            try:
                import time
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=30,
                    close_timeout=10,
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
                            meta = data.get("MetaData", {})
                            lat  = pos.get("Latitude",  0)
                            lon  = pos.get("Longitude", 0)
                            for name, box in CHOKEPOINTS.items():
                                if (box["min_lat"] <= lat <= box["max_lat"] and
                                        box["min_lon"] <= lon <= box["max_lon"]):
                                    counts[name] += 1
                                    break
                        except asyncio.TimeoutError:
                            continue
            except Exception as e:
                print(f"  AIS连接错误: {e}")

            return counts

        counts = asyncio.run(fetch())

        results = {}
        for name, info in CHOKEPOINTS.items():
            count       = counts[name]
            risk, color = assess_risk(count, info["normal_count"])
            ratio       = round(count / info["normal_count"] * 100, 1) if info["normal_count"] else 0
            results[name] = {
                "count": count,
                "normal_count": info["normal_count"],
                "risk": risk if AIS_COVERAGE[name] else "数据不足",
                "color": color if AIS_COVERAGE[name] else "#95a5a6",
                "importance": info["importance"],
                "ratio": ratio,
                "ais_coverage": AIS_COVERAGE[name],
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            print(f"  {name.ljust(10)}: {count:3d} 艘  ({ratio}%)  {risk}")

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"  AIS快照已保存")

    except Exception as e:
        print(f"  失败: {e}")


# ── 入口 ──────────────────────────────────────────────────────────────────
def run_update():
    print("\n" + "="*50)
    print(f"OilSense 数据更新  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*50)
    update_oil_prices()
    update_macro_data()
    update_news()
    update_portwatch()
    update_ais()
    print("\n" + "="*50)
    print("全部更新完成！")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_update()
