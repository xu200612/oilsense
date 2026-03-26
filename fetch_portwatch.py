import os
import time
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

BASE_URL = (
    "https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services"
    "/Daily_Chokepoints_Data/FeatureServer/0/query"
)

# 我们重点关注的咽喉点
CHOKEPOINTS = {
    "chokepoint6" : "霍尔木兹海峡",
    "chokepoint4" : "曼德海峡",
    "chokepoint1" : "苏伊士运河",
    "chokepoint5" : "马六甲海峡",
    "chokepoint3" : "博斯普鲁斯海峡",
    "chokepoint11": "台湾海峡",
    "chokepoint7" : "好望角",      # 霍尔木兹封锁后绕行路线
}

HEADERS = {"User-Agent": "Mozilla/5.0"}

def fetch_chokepoint_range(portid, start_year=2019, end_year=2026):
    """拉取单个咽喉点的全部历史数据"""
    all_records = []
    offset = 0
    batch  = 2000

    while True:
        params = {
            "where"           : f"portid='{portid}'",
            "outFields"       : "date,year,month,day,portid,portname,"
                                "n_tanker,n_total,capacity_tanker,capacity",
            "resultRecordCount": batch,
            "resultOffset"    : offset,
            "orderByFields"   : "date ASC",
            "f"               : "json",
        }
        try:
            r = requests.get(BASE_URL, params=params,
                             headers=HEADERS, timeout=15)
            d = r.json()
        except Exception as e:
            print("    请求失败: " + str(e)[:50])
            break

        features = d.get("features", [])
        if not features:
            break

        for feat in features:
            a = feat["attributes"]
            # ArcGIS 时间戳是毫秒
            ts   = a.get("date", 0)
            date = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d")
            all_records.append({
                "date"            : date,
                "portid"          : a.get("portid", ""),
                "portname"        : a.get("portname", ""),
                "n_tanker"        : a.get("n_tanker", 0),
                "n_total"         : a.get("n_total", 0),
                "capacity_tanker" : a.get("capacity_tanker", 0),
                "capacity_total"  : a.get("capacity", 0),
            })

        offset += batch
        # 用 exceededTransferLimit 判断是否还有更多数据
        # 而不是用返回数量，避免刚好整除时提前终止
        if not d.get("exceededTransferLimit", False):
            break
        time.sleep(0.3)

    return all_records

def build_portwatch_history():
    """下载所有关键咽喉点历史数据，合并成宽表"""
    print("开始下载 IMF PortWatch 咽喉点数据...")
    print("数据源：IMF PortWatch / Oxford University")
    print("-" * 50)

    all_dfs = []
    for portid, name in CHOKEPOINTS.items():
        print(f"  [{portid}] {name}...", end=" ")
        records = fetch_chokepoint_range(portid)
        if not records:
            print("无数据")
            continue

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        # 重命名列，加上咽喉点前缀
        short = portid.replace("chokepoint", "cp")
        df = df.rename(columns={
            "n_tanker"        : f"{short}_tanker",
            "n_total"         : f"{short}_total",
            "capacity_tanker" : f"{short}_cap_tanker",
            "capacity_total"  : f"{short}_cap_total",
        })
        df = df[[f"{short}_tanker", f"{short}_total",
                 f"{short}_cap_tanker", f"{short}_cap_total"]]

        all_dfs.append(df)
        print(f"✓ {len(df)} 条  ({df.index[0].date()} ~ {df.index[-1].date()})")

    if not all_dfs:
        print("未获取到任何数据")
        return pd.DataFrame()

    # 合并所有咽喉点
    result = pd.concat(all_dfs, axis=1)
    result.sort_index(inplace=True)

    # 衍生特征：霍尔木兹封锁信号
    if "cp6_tanker" in result.columns:
        # 7日滚动均值
        result["hormuz_tanker_ma7"]  = result["cp6_tanker"].rolling(7).mean()
        # 与历史同期对比的异常度（z-score）
        result["hormuz_tanker_zscore"] = (
            (result["cp6_tanker"] - result["cp6_tanker"].rolling(90).mean()) /
            (result["cp6_tanker"].rolling(90).std() + 1e-6)
        )
        # 封锁信号：油轮数量低于历史均值50%
        result["hormuz_blocked"] = (
            result["cp6_tanker"] <
            result["cp6_tanker"].rolling(90).mean() * 0.5
        ).astype(int)

    if "cp4_tanker" in result.columns:
        result["mandeb_tanker_ma7"] = result["cp4_tanker"].rolling(7).mean()
        result["mandeb_blocked"]    = (
            result["cp4_tanker"] <
            result["cp4_tanker"].rolling(90).mean() * 0.5
        ).astype(int)

    # 好望角绕行信号：霍尔木兹/曼德封锁时好望角流量上升
    if "cp7_tanker" in result.columns and "cp6_tanker" in result.columns:
        result["cape_reroute_signal"] = (
            result["cp7_tanker"].rolling(7).mean() /
            (result["cp7_tanker"].rolling(90).mean() + 1e-6)
        )

    # 保存
    out_path = os.path.join(ROOT_DIR, "data", "raw", "portwatch_chokepoints.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result.to_csv(out_path)

    print("-" * 50)
    print(f"PortWatch 数据已保存：{len(result)} 条，{result.shape[1]} 个字段")
    print(f"路径：{out_path}")

    # 打印霍尔木兹近30天数据
    if "cp6_tanker" in result.columns:
        print()
        print("霍尔木兹海峡近30天油轮通过量：")
        recent = result[["cp6_tanker", "hormuz_tanker_ma7",
                         "hormuz_blocked"]].tail(30)
        print(recent.to_string())

    return result

def get_chokepoint_status():
    """
    获取当前各咽喉点实时状态快照
    用于前端地球页面显示
    """
    out_path = os.path.join(ROOT_DIR, "data", "raw", "portwatch_chokepoints.csv")
    if not os.path.exists(out_path):
        print("数据文件不存在，请先运行 build_portwatch_history()")
        return {}

    df = pd.read_csv(out_path, index_col=0, parse_dates=True)
    status = {}

    cp_map = {
        "cp6" : "霍尔木兹海峡",
        "cp4" : "曼德海峡",
        "cp1" : "苏伊士运河",
        "cp5" : "马六甲海峡",
        "cp3" : "博斯普鲁斯海峡",
        "cp11": "台湾海峡",
        "cp7" : "好望角",
    }

    for cp, name in cp_map.items():
        col = f"{cp}_tanker"
        if col not in df.columns:
            continue

        recent   = df[col].dropna().tail(90)
        latest   = df[col].dropna().iloc[-1]
        avg_90d  = recent.mean()
        ratio    = latest / avg_90d if avg_90d > 0 else 1.0

        if ratio < 0.3:
            risk, color = "极高风险", "#e74c3c"
        elif ratio < 0.6:
            risk, color = "高风险",   "#e67e22"
        elif ratio < 0.85:
            risk, color = "偏低",     "#f1c40f"
        else:
            risk, color = "正常",     "#2ecc71"

        status[name] = {
            "latest_tankers" : int(latest),
            "avg_90d_tankers": round(avg_90d, 1),
            "ratio"          : round(ratio, 3),
            "risk"           : risk,
            "color"          : color,
            "last_date"      : str(df[col].dropna().index[-1].date()),
        }

    return status

if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "history"

    if mode == "status":
        status = get_chokepoint_status()
        print("当前咽喉点状态：")
        for name, s in status.items():
            print(f"  {name.ljust(12)}: {s['risk'].ljust(6)} "
                  f"油轮={s['latest_tankers']} "
                  f"均值={s['avg_90d_tankers']} "
                  f"比率={s['ratio']}")
    else:
        build_portwatch_history()
