import json
import os
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

VESSELAPI_BASE_URL = "https://api.vesselapi.com/v1"
VESSELAPI_API_KEY = os.getenv("VESSELAPI_API_KEY", "")

SHIPFINDER_BASE_URL = "https://api.elaneglobal.com/v1/AIS"
SHIPFINDER_API_KEY = os.getenv("SHIPFINDER_API_KEY", "")


CHOKEPOINTS = {
    "霍尔木兹海峡": {
        "normal_count": 32,
        "importance": "全球约20%石油海运贸易通道",
        "boxes": [
            {"lat_bottom": 25.5, "lat_top": 27.5, "lon_left": 55.5, "lon_right": 57.5},
        ],
        "shipfinder_regions": [
            "55.50,25.50-57.50,25.50-57.50,27.50-55.50,27.50",
        ],
    },
    "曼德海峡": {
        "normal_count": 18,
        "importance": "红海通往印度洋关键通道",
        "boxes": [
            {"lat_bottom": 11.0, "lat_top": 13.0, "lon_left": 42.5, "lon_right": 44.5},
        ],
        "shipfinder_regions": [
            "42.50,11.00-44.50,11.00-44.50,13.00-42.50,13.00",
        ],
    },
    "苏伊士运河": {
        "normal_count": 15,
        "importance": "欧亚最短海上航线",
        "boxes": [
            {"lat_bottom": 29.5, "lat_top": 31.5, "lon_left": 32.0, "lon_right": 33.0},
        ],
        "shipfinder_regions": [
            "32.00,29.50-33.00,29.50-33.00,31.50-32.00,31.50",
        ],
    },
    "马六甲海峡": {
        "normal_count": 85,
        "importance": "中东原油运往亚洲主要通道",
        "boxes": [
            {"lat_bottom": 1.0, "lat_top": 3.0, "lon_left": 99.0, "lon_right": 101.0},
            {"lat_bottom": 1.0, "lat_top": 3.0, "lon_left": 101.0, "lon_right": 103.0},
            {"lat_bottom": 1.0, "lat_top": 3.0, "lon_left": 103.0, "lon_right": 104.0},
        ],
        "shipfinder_regions": [
            "99.00,1.00-101.00,1.00-101.00,3.00-99.00,3.00",
            "101.00,1.00-103.00,1.00-103.00,3.00-101.00,3.00",
            "103.00,1.00-104.00,1.00-104.00,3.00-103.00,3.00",
        ],
    },
    "博斯普鲁斯海峡": {
        "normal_count": 12,
        "importance": "黑海原油出口关键通道",
        "boxes": [
            {"lat_bottom": 40.5, "lat_top": 41.5, "lon_left": 28.5, "lon_right": 29.5},
        ],
        "shipfinder_regions": [
            "28.50,40.50-29.50,40.50-29.50,41.50-28.50,41.50",
        ],
    },
}


def _risk_level(count: int, normal_count: int):
    if normal_count <= 0:
        return "未知", "#95a5a6", 0.0
    ratio = count / normal_count
    if ratio < 0.4:
        return "极高风险", "#e74c3c", ratio
    if ratio < 0.7:
        return "高风险", "#e67e22", ratio
    if ratio < 0.85:
        return "中等风险", "#f1c40f", ratio
    return "正常", "#2ecc71", ratio


def _extract_vessels(payload):
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return []
    for key in ["vessels", "data", "results", "items", "ship_list"]:
        value = payload.get(key)
        if isinstance(value, list):
            return value
    return []


def _vessel_id(vessel):
    return str(
        vessel.get("mmsi")
        or vessel.get("MMSI")
        or vessel.get("imo")
        or vessel.get("id")
        or vessel.get("vessel_id")
        or ""
    )


def _is_tanker(vessel):
    ship_type = vessel.get("ship_type")
    if isinstance(ship_type, int) and 80 <= ship_type <= 89:
        return True

    text = " ".join(
        str(vessel.get(k, ""))
        for k in [
            "type", "ship_type", "vessel_type", "classification",
            "category", "subtype", "cargo_type", "ship_name", "dest",
        ]
    ).lower()
    return any(
        term in text
        for term in ["tanker", "crude", "oil", "product", "chemical", "lng", "lpg"]
    )


def _fetch_vesselapi_box(session, box):
    params = {
        "filter.latBottom": box["lat_bottom"],
        "filter.latTop": box["lat_top"],
        "filter.lonLeft": box["lon_left"],
        "filter.lonRight": box["lon_right"],
        "pagination.limit": 50,
    }
    url = f"{VESSELAPI_BASE_URL}/location/vessels/bounding-box"
    response = session.get(url, params=params, timeout=20)
    if response.status_code >= 400:
        raise RuntimeError(f"{response.status_code}: {response.text[:300]}")
    return _extract_vessels(response.json())


def _fetch_shipfinder_region(session, region):
    url = f"{SHIPFINDER_BASE_URL}/VesselsInZone"
    response = session.get(
        url,
        params={"key": SHIPFINDER_API_KEY, "region": region, "output": 1},
        timeout=20,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"{response.status_code}: {response.text[:300]}")

    payload = response.json()
    status = payload.get("status")
    if status == 21:
        raise PermissionError("ShipFinder VesselsInZone 未授权")
    if status not in [0, "0", None]:
        raise RuntimeError(f"ShipFinder status={status}: {payload.get('msg', '')}")
    return _extract_vessels(payload)


def _summarize_vessels(vessels):
    seen = set()
    total = 0
    tankers = 0
    for vessel in vessels:
        vid = _vessel_id(vessel)
        if vid and vid in seen:
            continue
        if vid:
            seen.add(vid)
        total += 1
        if _is_tanker(vessel):
            tankers += 1
    return total, tankers


def _write_shipping_outputs(snapshot, rows):
    raw_dir = os.path.join(ROOT_DIR, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    ais_compatible = {}
    for name, item in snapshot.items():
        ais_compatible[name] = {
            "count": int(item["risk_count"]),
            "normal_count": int(item["normal_count"]),
            "risk": item["risk"],
            "color": item["color"],
            "importance": item["importance"],
            "ratio": item["ratio"],
            "ais_coverage": item["status"] not in ["failed", "no_data", "unauthorized"],
            "source": item["source"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    json_path = os.path.join(raw_dir, "shipping_realtime_snapshot.json")
    csv_path = os.path.join(raw_dir, "shipping_realtime_signals.csv")
    ais_path = os.path.join(raw_dir, "ais_snapshot.json")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    with open(ais_path, "w", encoding="utf-8") as f:
        json.dump(ais_compatible, f, ensure_ascii=False, indent=2)
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"  已保存：{json_path}")
    print(f"  已同步：{ais_path}")
    print(f"  已保存：{csv_path}")


def _make_record(name, cfg, total, tankers, source, status, errors, timestamp):
    risk_count = tankers if tankers > 0 else total
    risk, color, ratio = _risk_level(risk_count, cfg["normal_count"])
    if status in ["no_data", "unauthorized", "failed"]:
        risk = "数据不足" if status != "unauthorized" else "未授权"
        color = "#95a5a6"
        ratio = 0.0

    return {
        "source": source,
        "timestamp": timestamp,
        "count": int(total),
        "tanker_count": int(tankers),
        "risk_count": int(risk_count),
        "normal_count": int(cfg["normal_count"]),
        "ratio": round(float(ratio) * 100, 1),
        "risk": risk,
        "color": color,
        "importance": cfg["importance"],
        "status": status,
        "errors": errors[:3],
    }


def _row_from_record(name, record):
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "chokepoint": name,
        "count": int(record["count"]),
        "tanker_count": int(record["tanker_count"]),
        "risk_count": int(record["risk_count"]),
        "normal_count": int(record["normal_count"]),
        "ratio": round(float(record["ratio"]) / 100, 4),
        "risk": record["risk"],
        "status": record["status"],
        "source": record["source"],
        "timestamp": record["timestamp"],
    }


def fetch_shipfinder_snapshot(write_outputs=True):
    if not SHIPFINDER_API_KEY:
        print("  SHIPFINDER_API_KEY 未配置，跳过 ShipFinder")
        return {}

    session = requests.Session()
    session.headers.update({
        "User-Agent": "OilSense/1.0",
        "Accept": "application/json",
    })

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    snapshot = {}
    rows = []
    unauthorized_count = 0

    print("[ShipFinder] 区域 AIS 快照...")
    for name, cfg in CHOKEPOINTS.items():
        vessels = []
        errors = []
        status = "ok"
        for region in cfg["shipfinder_regions"]:
            try:
                vessels.extend(_fetch_shipfinder_region(session, region))
            except PermissionError as e:
                unauthorized_count += 1
                errors.append(str(e))
                status = "unauthorized"
                break
            except Exception as e:
                errors.append(str(e)[:120])
                status = "partial" if vessels else "failed"

        total, tankers = _summarize_vessels(vessels)
        if total == 0 and status == "ok":
            status = "no_data"

        record = _make_record(name, cfg, total, tankers, "ShipFinder", status, errors, now)
        snapshot[name] = record
        rows.append(_row_from_record(name, record))
        print(
            f"  {name}: vessels={total}, tankers={tankers}, "
            f"risk={record['risk']}, status={status}"
        )

    if unauthorized_count == len(CHOKEPOINTS):
        print("  ShipFinder 区域查询未开通权限：VesselsInZone 返回 Unauthorized")

    if write_outputs:
        _write_shipping_outputs(snapshot, rows)
    return snapshot


def fetch_vesselapi_snapshot(write_outputs=True):
    if not VESSELAPI_API_KEY:
        print("  VESSELAPI_API_KEY 未配置，跳过 VesselAPI 实时航运")
        return {}

    session = requests.Session()
    session.trust_env = False
    session.headers.update({
        "Authorization": f"Bearer {VESSELAPI_API_KEY}",
        "User-Agent": "OilSense/1.0",
        "Accept": "application/json",
    })

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    snapshot = {}
    rows = []

    print("[VesselAPI] 实时航运快照...")
    for name, cfg in CHOKEPOINTS.items():
        vessels = []
        errors = []
        for box in cfg["boxes"]:
            try:
                vessels.extend(_fetch_vesselapi_box(session, box))
            except Exception as e:
                errors.append(str(e)[:120])

        total, tankers = _summarize_vessels(vessels)
        if total == 0:
            status = "no_data"
        else:
            status = "ok" if not errors else "partial"

        record = _make_record(name, cfg, total, tankers, "VesselAPI", status, errors, now)
        snapshot[name] = record
        rows.append(_row_from_record(name, record))
        print(
            f"  {name}: vessels={total}, tankers={tankers}, "
            f"risk={record['risk']}, status={status}"
        )

    if write_outputs:
        _write_shipping_outputs(snapshot, rows)
    return snapshot


def fetch_realtime_shipping_snapshot():
    shipfinder = fetch_shipfinder_snapshot(write_outputs=False) if SHIPFINDER_API_KEY else {}
    vesselapi = fetch_vesselapi_snapshot(write_outputs=False) if VESSELAPI_API_KEY else {}

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    snapshot = {}
    rows = []
    for name, cfg in CHOKEPOINTS.items():
        sf = shipfinder.get(name)
        va = vesselapi.get(name)
        if sf and sf["status"] in ["ok", "partial"]:
            selected = sf
        elif va:
            selected = va.copy()
            if sf and sf["status"] == "unauthorized":
                selected["errors"] = (selected.get("errors") or []) + ["ShipFinder VesselsInZone 未授权，已回退 VesselAPI"]
                selected["source"] = "VesselAPI(+ShipFinder未授权)"
        elif sf:
            selected = sf
        else:
            selected = _make_record(name, cfg, 0, 0, "None", "failed", ["未配置可用实时 AIS API"], now)

        snapshot[name] = selected
        rows.append(_row_from_record(name, selected))

    print("[实时航运] 最终采用数据源：")
    for name, item in snapshot.items():
        print(f"  {name}: source={item['source']}, vessels={item['count']}, status={item['status']}")

    _write_shipping_outputs(snapshot, rows)
    return snapshot


if __name__ == "__main__":
    fetch_realtime_shipping_snapshot()
