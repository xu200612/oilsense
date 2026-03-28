# extreme_scenario.py
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 历史极端事件库（2008年至今，手动标注）────────────────────────────────
# 每个事件记录：触发窗口、触发类型、后续30日典型价格路径（相对涨跌幅）
# path_30d: 大致描述后续走势方向和幅度，用于加权推算当前预测
EXTREME_EVENTS = [
    # ── 宏观冲击类 ────────────────────────────────────────────────────────
    {
        "name"       : "金融危机油价崩盘",
        "start"      : "2008-09-15",
        "peak"       : "2008-10-16",
        "end"        : "2008-12-31",
        "trigger"    : "macro",
        "description": "雷曼兄弟破产，全球需求预期崩塌，WTI从147美元跌至35美元",
        "typical_30d": -0.45,   # 30日典型涨跌幅
        "severity"   : "extreme",
    },
    {
        "name"       : "新冠疫情需求崩溃",
        "start"      : "2020-02-24",
        "peak"       : "2020-03-09",
        "end"        : "2020-04-21",
        "trigger"    : "macro",
        "description": "新冠全球封锁+沙俄价格战，WTI从60美元跌至负值",
        "typical_30d": -0.55,
        "severity"   : "extreme",
    },
    {
        "name"       : "WTI期货负油价",
        "start"      : "2020-04-15",
        "peak"       : "2020-04-20",
        "end"        : "2020-04-28",
        "trigger"    : "macro",
        "description": "储存容量耗尽，WTI 5月合约跌至-37美元/桶，史无前例",
        "typical_30d": -0.30,
        "severity"   : "extreme",
    },
    {
        "name"       : "特朗普关税冲击",
        "start"      : "2025-03-24",
        "peak"       : "2025-04-02",
        "end"        : "2025-04-15",
        "trigger"    : "macro",
        "description": "特朗普宣布全面对等关税，全球需求担忧，油价单周暴跌12%",
        "typical_30d": -0.15,
        "severity"   : "severe",
    },

    # ── 地缘政治类 ────────────────────────────────────────────────────────
    {
        "name"       : "利比亚内战供应中断",
        "start"      : "2011-02-15",
        "peak"       : "2011-02-23",
        "end"        : "2011-03-15",
        "trigger"    : "geopolitics",
        "description": "利比亚内战爆发，150万桶/日供应中断，布伦特突破120美元",
        "typical_30d": 0.18,
        "severity"   : "severe",
    },
    {
        "name"       : "沙特阿美无人机袭击",
        "start"      : "2019-09-14",
        "peak"       : "2019-09-16",
        "end"        : "2019-09-30",
        "trigger"    : "geopolitics",
        "description": "胡塞武装袭击沙特最大炼油厂，单日暴涨15%，两周内回落",
        "typical_30d": -0.05,
        "severity"   : "moderate",
    },
    {
        "name"       : "俄乌战争爆发",
        "start"      : "2022-02-18",
        "peak"       : "2022-03-07",
        "end"        : "2022-03-31",
        "trigger"    : "geopolitics",
        "description": "俄罗斯入侵乌克兰，制裁俄油，布伦特飙至139美元",
        "typical_30d": 0.22,
        "severity"   : "extreme",
    },
    {
        "name"       : "以哈冲突爆发",
        "start"      : "2023-10-07",
        "peak"       : "2023-10-09",
        "end"        : "2023-10-31",
        "trigger"    : "geopolitics",
        "description": "哈马斯突袭以色列，中东局势紧张，但供应未直接受影响",
        "typical_30d": 0.05,
        "severity"   : "moderate",
    },
    {
        "name"       : "霍尔木兹封锁",
        "start"      : "2026-02-16",
        "peak"       : "2026-02-27",
        "end"        : "2026-03-15",
        "trigger"    : "geopolitics",
        "description": "伊朗封锁霍尔木兹海峡，全球20%石油贸易中断，油价持续飙升",
        "typical_30d": 0.35,
        "severity"   : "extreme",
    },

    # ── 供应政策类 ────────────────────────────────────────────────────────
    {
        "name"       : "OPEC价格战2014",
        "start"      : "2014-11-27",
        "peak"       : "2015-01-15",
        "end"        : "2015-03-31",
        "trigger"    : "supply_policy",
        "description": "OPEC拒绝减产，沙特打压页岩油，WTI从80美元跌至45美元",
        "typical_30d": -0.30,
        "severity"   : "severe",
    },
    {
        "name"       : "油价跌破26美元",
        "start"      : "2016-01-04",
        "peak"       : "2016-01-20",
        "end"        : "2016-02-11",
        "trigger"    : "supply_policy",
        "description": "供应过剩持续，WTI跌至13年低点26美元，沙特拒绝减产",
        "typical_30d": -0.20,
        "severity"   : "severe",
    },
    {
        "name"       : "沙俄价格战2020",
        "start"      : "2020-03-06",
        "peak"       : "2020-03-09",
        "end"        : "2020-03-18",
        "trigger"    : "supply_policy",
        "description": "OPEC+谈判破裂，沙特宣布增产，单日暴跌25%",
        "typical_30d": -0.40,
        "severity"   : "extreme",
    },
    {
        "name"       : "OPEC意外减产2023",
        "start"      : "2023-04-02",
        "peak"       : "2023-04-03",
        "end"        : "2023-04-14",
        "trigger"    : "supply_policy",
        "description": "OPEC+宣布意外自愿减产166万桶/日，油价单日暴涨6%",
        "typical_30d": 0.08,
        "severity"   : "moderate",
    },

    # ── 需求恢复类 ────────────────────────────────────────────────────────
    {
        "name"       : "疫情复苏+OPEC减产",
        "start"      : "2020-10-28",
        "peak"       : "2020-11-13",
        "end"        : "2020-12-31",
        "trigger"    : "demand_recovery",
        "description": "疫苗消息+拜登当选+OPEC延续减产，油价快速反弹至50美元",
        "typical_30d": 0.25,
        "severity"   : "moderate",
    },
    {
        "name"       : "后疫情需求爆发",
        "start"      : "2021-03-01",
        "peak"       : "2021-06-01",
        "end"        : "2021-09-30",
        "trigger"    : "demand_recovery",
        "description": "全球重新开放，需求超预期复苏，WTI从45美元涨至75美元",
        "typical_30d": 0.20,
        "severity"   : "moderate",
    },
]

# ── 触发类型分组（用于分类匹配）─────────────────────────────────────────
TRIGGER_GROUPS = {
    "geopolitics"    : ["geopolitics"],
    "macro"          : ["macro"],
    "supply_policy"  : ["supply_policy"],
    "demand_recovery": ["demand_recovery"],
    "mixed"          : ["geopolitics", "macro", "supply_policy", "demand_recovery"],
}

# ── 用于相似度匹配的特征（按重要性排序）─────────────────────────────────
MATCH_FEATURES = [
    "volatility", "vol_ratio", "VIX",
    "return_1d", "return_5d", "return_10d",
    "hormuz_tanker_zscore", "mandeb_blocked",
    "gdelt_conflict_intensity", "gdelt_goldstein",
    "sentiment_score", "geopolitics_flag",
    "ma_ratio", "high_vol",
]

# 各特征权重（地缘政治和航运类权重更高）
FEATURE_WEIGHTS = {
    "volatility"              : 1.0,
    "vol_ratio"               : 1.2,
    "VIX"                     : 1.2,
    "return_1d"               : 0.8,
    "return_5d"               : 1.0,
    "return_10d"              : 0.8,
    "hormuz_tanker_zscore"    : 2.0,   # 霍尔木兹异常权重最高
    "mandeb_blocked"          : 1.5,
    "gdelt_conflict_intensity": 1.5,
    "gdelt_goldstein"         : 1.2,
    "sentiment_score"         : 1.0,
    "geopolitics_flag"        : 1.3,
    "ma_ratio"                : 0.7,
    "high_vol"                : 0.9,
}


def _load_feature_matrix():
    path = os.path.join(ROOT_DIR, "data", "processed", "feature_matrix.csv")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _infer_trigger_type(current_features: pd.Series) -> str:
    """
    根据当前特征推断最可能的触发类型，用于分类匹配
    """
    hormuz_z = abs(float(current_features.get("hormuz_tanker_zscore", 0)))
    geo_flag = float(current_features.get("geopolitics_flag", 0))
    gdelt_ci = float(current_features.get("gdelt_conflict_intensity", 0))
    vix      = float(current_features.get("VIX", 20))
    vol_ratio= float(current_features.get("vol_ratio", 1))
    ret_5d   = float(current_features.get("return_5d", 0))

    # 霍尔木兹/航运异常 → 地缘政治
    if hormuz_z > 2.0 or (geo_flag > 0 and gdelt_ci < -3):
        return "geopolitics"
    # VIX极高+油价下跌 → 宏观冲击
    if vix > 35 and ret_5d < -0.05:
        return "macro"
    # 高波动+油价下跌（无明显地缘信号）→ 供应政策
    if vol_ratio > 2.0 and ret_5d < -0.03:
        return "supply_policy"
    # 高波动+油价上涨 → 可能是需求恢复或地缘
    if vol_ratio > 1.5 and ret_5d > 0.03:
        return "geopolitics" if geo_flag > 0 else "demand_recovery"

    return "mixed"


def find_similar_events(current_features: pd.Series, top_k: int = 3) -> list:
    """
    用加权余弦相似度找历史上最相似的极端事件
    改进：使用触发窗口5日均值特征，而非单个时间点，抗噪性更强
    """
    feat = _load_feature_matrix()

    # 推断触发类型，优先在同类事件里匹配
    trigger_type = _infer_trigger_type(current_features)

    # 只取有效特征列
    match_cols = [c for c in MATCH_FEATURES if c in feat.columns
                  and c in current_features.index]
    weights    = np.array([FEATURE_WEIGHTS.get(c, 1.0) for c in match_cols])

    feat_sub = feat[match_cols].ffill().fillna(0)

    # 当前特征向量（加权）
    cur_vec = (current_features[match_cols].fillna(0).values * weights).reshape(1, -1)

    # 对每个极端事件取触发窗口前后5日均值特征（抗噪）
    event_vecs    = []
    event_meta    = []
    for ev in EXTREME_EVENTS:
        try:
            start_ts = pd.Timestamp(ev["start"])
            # 取触发日前后5日窗口均值
            window_start = start_ts - pd.Timedelta(days=2)
            window_end   = start_ts + pd.Timedelta(days=2)
            window_data  = feat_sub.loc[window_start:window_end]

            if len(window_data) == 0:
                # 找最近有效行
                idx = feat_sub.index.get_indexer([start_ts], method="nearest")[0]
                vec = feat_sub.iloc[idx].values
                matched_date = feat_sub.index[idx]
            else:
                vec          = window_data.mean().values
                matched_date = feat_sub.index[
                    feat_sub.index.get_indexer([start_ts], method="nearest")[0]
                ]

            event_vecs.append(vec * weights)
            event_meta.append((ev, matched_date))
        except Exception:
            continue

    if not event_vecs:
        return []

    event_matrix = np.array(event_vecs)
    dists        = cdist(cur_vec, event_matrix, metric="cosine")[0]

    # 同类型事件给予距离折扣（优先匹配）
    for i, (ev, _) in enumerate(event_meta):
        if trigger_type != "mixed" and ev["trigger"] == trigger_type:
            dists[i] *= 0.75   # 同类事件距离缩短25%，相似度提升

    top_idx = np.argsort(dists)[:top_k]

    results = []
    for i in top_idx:
        ev, trigger_date = event_meta[i]
        similarity = float(np.clip(1 - dists[i], 0, 1))

        # 从特征矩阵提取该事件后续实际涨跌幅
        try:
            loc          = feat.index.get_loc(trigger_date)
            # 后10日涨跌幅（模型horizon）
            ret_10d      = feat["target"].iloc[loc] if loc < len(feat) else None
            # 后30日累计涨跌幅（从油价直接算，异常时回退到手动标注值）
            if "WTI" in feat.columns and loc + 30 < len(feat):
                p0       = feat["WTI"].iloc[loc]
                p30      = feat["WTI"].iloc[loc + 30]
                computed = float((p30 - p0) / p0) if p0 > 0 else None
                typical  = ev.get("typical_30d", 0)
                # 计算值方向与手动标注相反，或幅度差异超过5倍，优先用手动标注
                if computed is not None and typical != 0:
                    same_dir = (computed * typical) > 0
                    ratio    = abs(computed / typical) if typical != 0 else 0
                    ret_30d  = computed if same_dir and 0.2 < ratio < 5.0 else typical
                else:
                    ret_30d  = computed if computed is not None else typical
            else:
                ret_30d = ev.get("typical_30d")
        except Exception:
            ret_10d = None
            ret_30d = ev.get("typical_30d")

        results.append({
            "event"        : ev["name"],
            "trigger"      : ev["trigger"],
            "severity"     : ev.get("severity", "moderate"),
            "description"  : ev["description"],
            "trigger_date" : str(trigger_date.date()),
            "similarity"   : round(similarity, 3),
            "actual_return": ret_10d,           # 10日涨跌幅（模型对齐）
            "return_30d"   : ret_30d,           # 30日涨跌幅（报告用）
            "typical_30d"  : ev.get("typical_30d"),
            "distance"     : round(float(dists[i]), 3),
        })

    return results


def get_extreme_prediction(current_features: pd.Series, base_low: float,
                           base_mid: float, base_high: float) -> dict:
    """
    第二层极端事件预测
    激活条件：高波动 或 VIX恐慌 或 航运异常
    输出：调整后的10日置信区间 + 30日情景路径 + 相似历史事件
    """
    vol_ratio = float(current_features.get("vol_ratio", 1))
    vix       = float(current_features.get("VIX", 20))
    hormuz_z  = float(current_features.get("hormuz_tanker_zscore", 0))
    geo_flag  = float(current_features.get("geopolitics_flag", 0))
    gdelt_ci  = float(current_features.get("gdelt_conflict_intensity", 0))

    # 激活条件（任意一个触发）
    is_extreme = (
        vol_ratio      > 2.0  or
        vix            > 30   or
        abs(hormuz_z)  > 2.0  or
        (geo_flag > 0 and gdelt_ci < -4.0)   # 极端地缘冲突
    )

    if not is_extreme:
        return {
            "activated"     : False,
            "pred_low"      : base_low,
            "pred_mid"      : base_mid,
            "pred_high"     : base_high,
            "similar_events": [],
            "scale_factor"  : 1.0,
            "trigger_type"  : "none",
        }

    trigger_type = _infer_trigger_type(current_features)
    similar      = find_similar_events(current_features, top_k=3)

    if not similar:
        # 无匹配，保守扩大置信区间
        return {
            "activated"     : True,
            "pred_low"      : base_low  * 2.5,
            "pred_mid"      : base_mid,
            "pred_high"     : base_high * 2.5,
            "similar_events": [],
            "scale_factor"  : 2.5,
            "trigger_type"  : trigger_type,
        }

    # ── 加权计算修正区间 ─────────────────────────────────────────────────
    weights  = np.array([ev["similarity"] for ev in similar])
    weights  = weights / (weights.sum() + 1e-9)

    # 10日涨跌幅加权
    returns_10d = np.array([
        ev["actual_return"] if ev["actual_return"] is not None else base_mid
        for ev in similar
    ])
    weighted_return_10d = float(np.dot(weights, returns_10d))

    # 30日涨跌幅加权（用于报告情景描述）
    returns_30d = np.array([
        ev["return_30d"] if ev["return_30d"] is not None else ev["typical_30d"] or 0
        for ev in similar
    ])
    weighted_return_30d = float(np.dot(weights, returns_30d))

    # 离散度作为区间宽度
    spread = float(np.std(returns_10d)) if len(returns_10d) > 1 else abs(weighted_return_10d) * 0.6

    adj_low  = weighted_return_10d - 2.0 * spread
    adj_mid  = weighted_return_10d
    adj_high = weighted_return_10d + 2.0 * spread

    # 取两层包络（更保守的低端，更激进的高端）
    final_low  = min(base_low,  adj_low)
    final_mid  = adj_mid
    final_high = max(base_high, adj_high)

    scale_factor = abs(final_high - final_low) / max(abs(base_high - base_low), 1e-6)

    # ── 生成30日分情景路径 ───────────────────────────────────────────────
    # 取相似度最高的事件作为主要参考情景
    top_ev   = similar[0]
    severity = top_ev.get("severity", "moderate")

    scenarios_30d = {
        "快速缓解": round(weighted_return_30d * 0.3, 4),
        "僵持延续": round(weighted_return_30d * 0.8, 4),
        "全面升级": round(weighted_return_30d * 1.5, 4),
    }

    return {
        "activated"          : True,
        "pred_low"           : round(final_low,  4),
        "pred_mid"           : round(final_mid,  4),
        "pred_high"          : round(final_high, 4),
        "similar_events"     : similar,
        "scale_factor"       : round(scale_factor, 2),
        "weighted_return_10d": round(weighted_return_10d, 4),
        "weighted_return_30d": round(weighted_return_30d, 4),
        "scenarios_30d"      : scenarios_30d,
        "trigger_type"       : trigger_type,
        "severity"           : severity,
        "vol_ratio"          : round(vol_ratio, 2),
        "vix"                : round(vix, 1),
        "hormuz_zscore"      : round(hormuz_z, 2),
    }


if __name__ == "__main__":
    feat   = _load_feature_matrix()
    latest = feat.iloc[-1]
    print("测试日期: " + str(feat.index[-1].date()))
    print("vol_ratio: " + str(round(float(latest.get("vol_ratio", 0)), 2)))
    print("VIX: "       + str(round(float(latest.get("VIX", 0)), 1)))
    print("hormuz_z: "  + str(round(float(latest.get("hormuz_tanker_zscore", 0)), 2)))

    result = get_extreme_prediction(latest, -0.05, 0.02, 0.08)
    print("\n第二层激活: " + str(result["activated"]))
    print("触发类型: "   + result.get("trigger_type", "none"))
    print("修正后区间: " + str(result["pred_low"]) + " / " +
          str(result["pred_mid"]) + " / " + str(result["pred_high"]))
    print("规模因子: "   + str(result["scale_factor"]))

    if result.get("similar_events"):
        print("\n最相似历史事件:")
        for ev in result["similar_events"]:
            ret_str = "{:.3f}".format(ev["actual_return"]) if ev["actual_return"] is not None else "N/A"
            print("  " + ev["event"] +
                  "（相似度:" + str(ev["similarity"]) +
                  " 严重程度:" + ev["severity"] +
                  " 10日涨跌:" + ret_str + "）")

    if result.get("scenarios_30d"):
        print("\n30日情景路径:")
        for name, val in result["scenarios_30d"].items():
            print("  " + name + ": " + "{:+.1%}".format(val))