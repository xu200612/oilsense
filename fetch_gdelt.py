import os
import io
import zipfile
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

# GDELT 事件类型：与能源/地缘政治相关的 CAMEO 代码
# 14x = 抗议, 15x = 展示武力, 17x = 胁迫, 18x = 袭击, 19x = 战斗
# 11x = 制裁/施压, 12x = 上诉/呼吁
ENERGY_COUNTRIES = [
    "US", "RS", "IR", "SA", "IZ", "AE", "KU", "VE", "LY",
    "NG", "NO", "KZ", "AL", "QA", "OA"  # OA = OPEC
]

CAMEO_RELEVANT = ["11", "12", "13", "14", "15", "17", "18", "19", "20"]

GDELT_COLS = [
    "GlobalEventID", "Day", "MonthYear", "Year", "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
    "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
    "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions", "NumSources",
    "NumArticles", "AvgTone", "Actor1Geo_Type", "Actor1Geo_FullName",
    "Actor1Geo_CountryCode", "Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code",
    "Actor1Geo_Lat", "Actor1Geo_Long", "Actor1Geo_FeatureID",
    "Actor2Geo_Type", "Actor2Geo_FullName", "Actor2Geo_CountryCode",
    "Actor2Geo_ADM1Code", "Actor2Geo_ADM2Code", "Actor2Geo_Lat",
    "Actor2Geo_Long", "Actor2Geo_FeatureID", "ActionGeo_Type",
    "ActionGeo_FullName", "ActionGeo_CountryCode", "ActionGeo_ADM1Code",
    "ActionGeo_ADM2Code", "ActionGeo_Lat", "ActionGeo_Long",
    "ActionGeo_FeatureID", "DATEADDED", "SOURCEURL"
]

def download_gdelt_day(date_str):
    """
    下载指定日期的 GDELT 数据（格式：YYYYMMDD）
    GDELT v2 按15分钟分块，我们取当天第一个文件做日频代理
    """
    url = ("http://data.gdeltproject.org/gdeltv2/" +
           date_str + "000000.export.CSV.zip")
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return None
        z = zipfile.ZipFile(io.BytesIO(r.content))
        fname = z.namelist()[0]
        df = pd.read_csv(
            z.open(fname), sep="\t", header=None,
            names=GDELT_COLS, low_memory=False,
            on_bad_lines="skip"
        )
        return df
    except Exception as e:
        print("    下载失败: " + str(e)[:50])
        return None

def extract_energy_sentiment(df, date_str):
    """
    从单日 GDELT 数据提取能源相关地缘政治情感指标
    """
    if df is None or len(df) == 0:
        return None

    # 过滤：涉及产油国的事件
    mask = (
        df["Actor1CountryCode"].isin(ENERGY_COUNTRIES) |
        df["Actor2CountryCode"].isin(ENERGY_COUNTRIES) |
        df["ActionGeo_CountryCode"].isin(ENERGY_COUNTRIES)
    )
    energy_df = df[mask].copy()

    if len(energy_df) == 0:
        return None

    # 过滤：相关事件类型
    energy_df["EventRootCode"] = energy_df["EventRootCode"].astype(str)
    relevant = energy_df[
        energy_df["EventRootCode"].isin(CAMEO_RELEVANT)
    ]

    date = datetime.strptime(date_str, "%Y%m%d").strftime("%Y-%m-%d")

    return {
        "date"              : date,
        # GoldsteinScale 均值：负值=冲突/制裁，正值=合作
        "gdelt_goldstein"   : energy_df["GoldsteinScale"].mean(),
        # 平均情感极性
        "gdelt_tone"        : energy_df["AvgTone"].mean(),
        # 冲突事件数量（QuadClass=3,4 代表冲突对抗）
        "gdelt_conflict_cnt": int((energy_df["QuadClass"] >= 3).sum()),
        # 合作事件数量
        "gdelt_coop_cnt"    : int((energy_df["QuadClass"] <= 2).sum()),
        # 相关事件提及次数（衡量事件重要性）
        "gdelt_mentions"    : int(energy_df["NumMentions"].sum()),
        # 冲突强度（仅取冲突事件的 Goldstein）
        "gdelt_conflict_intensity": (
            relevant["GoldsteinScale"].mean()
            if len(relevant) > 0 else 0
        ),
    }

def build_gdelt_history(start_date="2022-01-01", end_date=None):
    """
    批量下载历史 GDELT 数据，构建日频地缘政治情感序列
    建议先从2022年开始（覆盖俄乌冲突至今）
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.strptime(end_date,   "%Y-%m-%d")

    # 只取工作日
    dates = pd.bdate_range(start, end)
    print("共需下载 " + str(len(dates)) + " 个交易日的数据")
    print("预计耗时约 " + str(round(len(dates) * 5 / 60, 1)) + " 分钟")
    print("-" * 50)

    records = []
    for i, date in enumerate(dates):
        date_str = date.strftime("%Y%m%d")
        print("  [" + str(i+1) + "/" + str(len(dates)) + "] " +
              date.strftime("%Y-%m-%d") + "...", end=" ")

        df     = download_gdelt_day(date_str)
        record = extract_energy_sentiment(df, date_str)

        if record:
            records.append(record)
            print("✓ Goldstein=" + str(round(record["gdelt_goldstein"], 2)) +
                  " Tone=" + str(round(record["gdelt_tone"], 2)) +
                  " 冲突=" + str(record["gdelt_conflict_cnt"]))
        else:
            print("跳过")

    if not records:
        print("未获取到任何数据")
        return pd.DataFrame()

    result = pd.DataFrame(records).set_index("date")
    result.index = pd.to_datetime(result.index)
    result.sort_index(inplace=True)

    # 保存
    out_path = os.path.join(ROOT_DIR, "data", "raw", "gdelt_sentiment.csv")
    result.to_csv(out_path)
    print("-" * 50)
    print("GDELT 数据已保存：" + str(len(result)) + " 条")
    print("路径：" + out_path)
    return result

def update_gdelt_recent(days_back=7):
    """
    只更新最近 N 天的 GDELT 数据（日常更新用）
    """
    print("更新最近 " + str(days_back) + " 天的 GDELT 数据...")
    end   = datetime.today()
    start = end - timedelta(days=days_back)
    dates = pd.bdate_range(start, end)

    out_path = os.path.join(ROOT_DIR, "data", "raw", "gdelt_sentiment.csv")
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path, index_col=0, parse_dates=True)
    else:
        existing = pd.DataFrame()

    records = []
    for date in dates:
        date_str = date.strftime("%Y%m%d")
        print("  " + date.strftime("%Y-%m-%d") + "...", end=" ")
        df     = download_gdelt_day(date_str)
        record = extract_energy_sentiment(df, date_str)
        if record:
            records.append(record)
            print("✓")
        else:
            print("跳过")

    if records:
        new_df = pd.DataFrame(records).set_index("date")
        new_df.index = pd.to_datetime(new_df.index)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        combined.to_csv(out_path)
        print("更新完成：共 " + str(len(combined)) + " 条")

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "recent"

    if mode == "history":
        # 首次运行：构建历史数据（从2022年至今，约1000个交易日）
        print("构建历史 GDELT 数据（2022-01-01 至今）...")
        build_gdelt_history(start_date="2022-01-01")
    elif mode == "history_full":
        # 完整历史：从2020年至今
        print("构建完整历史 GDELT 数据（2020-01-01 至今）...")
        build_gdelt_history(start_date="2020-01-01")
    else:
        # 日常更新：只拉最近7天
        update_gdelt_recent(days_back=7)

    print("完成！")
