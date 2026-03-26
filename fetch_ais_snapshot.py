import asyncio
import websockets
import json
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

API_KEY = os.getenv("AIS_API_KEY", "你的API Key")

CHOKEPOINTS = {
    "霍尔木兹海峡": {
        "min_lat": 25.5, "max_lat": 27.5,
        "min_lon": 55.5, "max_lon": 57.5,
        "normal_count": 32,   # 历史正常水平（艘/120秒窗口）
        "importance": "全球20%石油贸易经过此处",
        "risk_countries": ["伊朗", "阿联酋", "科威特", "伊拉克", "沙特阿拉伯"],
    },
    "曼德海峡": {
        "min_lat": 11.0, "max_lat": 13.5,
        "min_lon": 42.5, "max_lon": 44.5,
        "normal_count": 18,
        "importance": "红海通往印度洋的唯一通道",
        "risk_countries": ["也门", "厄立特里亚"],
    },
    "苏伊士运河": {
        "min_lat": 29.5, "max_lat": 31.5,
        "min_lon": 32.0, "max_lon": 33.0,
        "normal_count": 15,
        "importance": "欧洲与亚洲最短海上航线",
        "risk_countries": ["埃及"],
    },
    "马六甲海峡": {
        "min_lat":  1.0, "max_lat":  4.0,
        "min_lon": 99.0, "max_lon": 104.0,
        "normal_count": 85,
        "importance": "中东原油运往亚洲的主要通道",
        "risk_countries": ["马来西亚", "印度尼西亚", "新加坡"],
    },
    "博斯普鲁斯海峡": {
        "min_lat": 40.5, "max_lat": 41.5,
        "min_lon": 28.5, "max_lon": 29.5,
        "normal_count": 12,
        "importance": "俄罗斯黑海原油出口唯一通道",
        "risk_countries": ["俄罗斯", "土耳其"],
    },
}

def assess_risk(count, normal_count):
    """根据当前船舶数量与历史均值对比评估风险"""
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

async def fetch_snapshot(duration=120):
    """抓取指定秒数的 AIS 快照数据"""
    url = "wss://stream.aisstream.io/v0/stream"

    bounding_boxes = [
        [[box["min_lat"], box["min_lon"]], [box["max_lat"], box["max_lon"]]]
        for box in CHOKEPOINTS.values()
    ]

    subscribe_msg = {
        "APIKey"             : API_KEY,
        "BoundingBoxes"      : bounding_boxes,
        "FilterMessageTypes" : ["PositionReport"]
    }

    counts = {name: 0 for name in CHOKEPOINTS}
    ships  = {name: [] for name in CHOKEPOINTS}

    print("正在抓取 AIS 快照（" + str(duration) + "秒）...")

    try:
        async with websockets.connect(
            url,
            ping_interval=20,
            ping_timeout=10,
        ) as ws:
            await ws.send(json.dumps(subscribe_msg))

            import time
            start = time.time()
            while time.time() - start < duration:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5)
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")

                    data = json.loads(raw)
                    pos  = data.get("Message", {}).get("PositionReport", {})
                    meta = data.get("MetaData", {})

                    lat       = pos.get("Latitude",  0)
                    lon       = pos.get("Longitude", 0)
                    ship_name = meta.get("ShipName", "").strip()
                    mmsi      = meta.get("MMSI", "")
                    speed     = pos.get("SpeedOverGround", 0)

                    for name, box in CHOKEPOINTS.items():
                        if (box["min_lat"] <= lat <= box["max_lat"] and
                                box["min_lon"] <= lon <= box["max_lon"]):
                            counts[name] += 1
                            ships[name].append({
                                "name" : ship_name,
                                "mmsi" : mmsi,
                                "lat"  : lat,
                                "lon"  : lon,
                                "speed": speed,
                            })
                            break

                except asyncio.TimeoutError:
                    continue

    except Exception as e:
        print("AIS 连接错误: " + str(e)[:80])

    # 生成快照结果
    results = {}
    for name, info in CHOKEPOINTS.items():
        count        = counts[name]
        risk, color  = assess_risk(count, info["normal_count"])
        results[name] = {
            "count"        : count,
            "normal_count" : info["normal_count"],
            "risk"         : risk,
            "color"        : color,
            "importance"   : info["importance"],
            "risk_countries": info["risk_countries"],
            "ships"        : ships[name][:20],  # 最多保留20艘
            "timestamp"    : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        ratio = round(count / info["normal_count"] * 100, 1) if info["normal_count"] else 0
        print("  " + name.ljust(12) +
              ": " + str(count).rjust(3) + " 艘  " +
              "（正常水平" + str(ratio) + "%）  " + risk)

    # 保存快照
    out_path = os.path.join(ROOT_DIR, "data", "raw", "ais_snapshot.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        # ships 列表不存入文件，只保留统计数据
        save_data = {
            k: {kk: vv for kk, vv in v.items() if kk != "ships"}
            for k, v in results.items()
        }
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print("快照已保存：" + out_path)

    return results

def load_snapshot():
    """读取最新快照，若不存在返回默认值"""
    out_path = os.path.join(ROOT_DIR, "data", "raw", "ais_snapshot.json")
    if not os.path.exists(out_path):
        return None
    with open(out_path, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    results = asyncio.run(fetch_snapshot(duration=120))
    print()
    print("── 咽喉点风险评估 ──")
    for name, data in results.items():
        print("  " + name + ": " + data["risk"] +
              "  (" + str(data["count"]) + "/" +
              str(data["normal_count"]) + " 艘)")
