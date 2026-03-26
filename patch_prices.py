import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

AV_KEY = os.getenv("ALPHA_VANTAGE_KEY")

def fetch_realtime_price(symbol, retries=3):
    """抓取 Yahoo Finance 实时价格，带重试"""
    url     = "https://query1.finance.yahoo.com/v8/finance/chart/" + symbol
    headers = {"User-Agent": "Mozilla/5.0"}
    for attempt in range(retries):
        try:
            r     = requests.get(url, headers=headers, timeout=15)
            data  = r.json()
            meta  = data["chart"]["result"][0]["meta"]
            price = meta["regularMarketPrice"]
            ts    = meta["regularMarketTime"]
            date  = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
            return date, price
        except Exception as e:
            print("  第" + str(attempt+1) + "次失败: " + str(e)[:60])
            if attempt < retries - 1:
                import time; time.sleep(3)
    return None, None

def patch_oil_prices():
    today = datetime.today().strftime("%Y-%m-%d")
    print("正在更新油价数据（" + today + "）...")

    # 读取现有完整历史数据
    oil_path = os.path.join(ROOT_DIR, "data", "raw", "oil_prices.csv")
    existing = pd.read_csv(oil_path, index_col=0, parse_dates=True)
    print("  现有数据：" + str(len(existing)) + " 条，截至 " +
          str(existing.dropna().index[-1].date()))

    # 获取实时价格
    print("  获取 Yahoo Finance 实时价格...")
    wti_date,   wti_price   = fetch_realtime_price("CL=F")
    brent_date, brent_price = fetch_realtime_price("BZ=F")

    today_data = {}
    if wti_price:
        today_data["WTI"] = wti_price
        print("  WTI   实时价格: $" + str(round(wti_price, 2)) +
              "  (" + str(wti_date) + ")")
    else:
        print("  WTI   获取失败，跳过")

    if brent_price:
        today_data["Brent"] = brent_price
        print("  Brent 实时价格: $" + str(round(brent_price, 2)) +
              "  (" + str(brent_date) + ")")
    else:
        print("  Brent 获取失败，跳过")

    if not today_data:
        print("  两个价格均获取失败，数据保持不变")
        print("  当前最新数据截至: " + str(existing.dropna().index[-1].date()))
        return existing

    # 追加今日数据
    ref_date = wti_date if wti_date else brent_date
    today_df = pd.DataFrame(today_data, index=[pd.Timestamp(ref_date)])

    combined = pd.concat([existing, today_df])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)
    combined.dropna(how="all", inplace=True)
    combined.to_csv(oil_path)

    print("  油价数据已更新至: " + str(combined.dropna().index[-1].date()))
    print("  共 " + str(len(combined)) + " 条记录")
    return combined

if __name__ == "__main__":
    patch_oil_prices()
    print("完成！如需重新训练模型，运行 python train_model.py")
