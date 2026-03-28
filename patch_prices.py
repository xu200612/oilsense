import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

AV_KEY = os.getenv("ALPHA_VANTAGE_KEY")

def fetch_realtime_price(symbol, retries=3):
    """
    抓取 Yahoo Finance 实时价格，带重试。
    非交易时段返回最新可用价格，日期统一用今天，避免被 drop_duplicates 丢弃。
    """
    url     = "https://query1.finance.yahoo.com/v8/finance/chart/" + symbol
    headers = {"User-Agent": "Mozilla/5.0"}

    for attempt in range(retries):
        try:
            r    = requests.get(url, headers=headers, timeout=15)
            data = r.json()
            meta = data["chart"]["result"][0]["meta"]

            # 优先取实时价，非交易时段回退到上次收盘价
            price = meta.get("regularMarketPrice") or meta.get("chartPreviousClose")
            if not price:
                raise ValueError("价格字段为空")

            # 日期统一用今天，不用市场时间戳
            # 原因：非交易时段时间戳是上次收盘日，写入会被 drop_duplicates 当重复数据丢弃
            today = datetime.today().strftime("%Y-%m-%d")
            return today, float(price)

        except Exception as e:
            print("  第" + str(attempt + 1) + "次失败: " + str(e)[:60])
            if attempt < retries - 1:
                import time; time.sleep(3)

    return None, None


def patch_oil_prices():
    today = datetime.today().strftime("%Y-%m-%d")
    print("正在更新油价数据（" + today + "）...")

    oil_path = os.path.join(ROOT_DIR, "data", "raw", "oil_prices.csv")
    existing = pd.read_csv(oil_path, index_col=0, parse_dates=True)
    print("  现有数据：" + str(len(existing)) + " 条，截至 " +
          str(existing.dropna().index[-1].date()))

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

    # 写入今天的价格（即使是非交易日，也用今天日期写入，保证数据不断档）
    today_df = pd.DataFrame(today_data, index=[pd.Timestamp(today)])

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