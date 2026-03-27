# extreme_scenario.py
import os
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 历史极端事件库（手动标注）
EXTREME_EVENTS = [
    {
        "name"       : "俄乌战争爆发",
        "start"      : "2022-02-18",
        "peak"       : "2022-02-23",
        "end"        : "2022-03-08",
        "trigger"    : "geopolitics",
        "description": "俄罗斯入侵乌克兰，制裁俄油，布伦特飙至139美元",
    },
    {
        "name"       : "疫情复苏+OPEC减产",
        "start"      : "2020-10-28",
        "peak"       : "2020-11-13",
        "end"        : "2020-12-30",
        "trigger"    : "demand_recovery",
        "description": "疫苗消息+拜登当选+OPEC延续减产，油价快速反弹",
    },
    {
        "name"       : "SVB危机+OPEC意外减产",
        "start"      : "2023-03-17",
        "peak"       : "2023-03-23",
        "end"        : "2023-03-30",
        "trigger"    : "policy",
        "description": "银行危机后OPEC+意外宣布减产100万桶/日",
    },
    {
        "name"       : "特朗普关税+需求担忧",
        "start"      : "2025-03-24",
        "peak"       : "2025-04-02",
        "end"        : "2025-04-02",
        "trigger"    : "macro",
        "description": "特朗普全面关税落地，全球需求担忧，油价暴跌",
    },
    {
        "name"       : "霍尔木兹封锁",
        "start"      : "2026-02-16",
        "peak"       : "2026-02-27",
        "end"        : "2026-03-05",
        "trigger"    : "geopolitics",
        "description": "伊朗封锁霍尔木兹海峡，全球原油供应中断",
    },
]

# 用于相似度匹配的特征
MATCH_FEATURES = [
    "volatility", "vol_ratio", "VIX", "return_1d", "return_5d",
    "hormuz_tanker_zscore", "gdelt_conflict_intensity", "sentiment_score",
]


def _load_feature_matrix():
    path = os.path.join(ROOT_DIR, "data", "processed", "feature_matrix.csv")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def find_similar_events(current_features: pd.Series, top_k=3) -> list:
    """
    用余弦相似度找历史上最相似的极端事件时段
    返回top_k个最相似事件及其后续价格路径
    """
    feat = _load_feature_matrix()

    # 只取有效特征列
    match_cols = [c for c in MATCH_FEATURES if c in feat.columns and c in current_features.index]

    # 归一化
    feat_sub = feat[match_cols].ffill().fillna(0)
    cur_vec  = current_features[match_cols].fillna(0).values.reshape(1, -1)

    # 对每个极端事件取触发日的特征向量
    event_vecs  = []
    event_dates = []
    for ev in EXTREME_EVENTS:
        try:
            start = pd.Timestamp(ev["start"])
            # 找最近的有效行
            idx = feat_sub.index.get_indexer([start], method="nearest")[0]
            event_vecs.append(feat_sub.iloc[idx].values)
            event_dates.append((ev, feat_sub.index[idx]))
        except:
            continue

    if not event_vecs:
        return []

    event_matrix = np.array(event_vecs)

    # 余弦距离
    dists = cdist(cur_vec, event_matrix, metric="cosine")[0]
    top_idx = np.argsort(dists)[:top_k]

    results = []
    for i in top_idx:
        ev, trigger_date = event_dates[i]
        similarity = 1 - dists[i]

        # 提取该事件后10日的实际价格路径
        try:
            loc = feat.index.get_loc(trigger_date)
            path_slice = feat["target"].iloc[loc: loc + 1]
            actual_return = float(path_slice.iloc[0]) if len(path_slice) > 0 else None
        except:
            actual_return = None

        results.append({
            "event"        : ev["name"],
            "trigger"      : ev["trigger"],
            "description"  : ev["description"],
            "trigger_date" : str(trigger_date.date()),
            "similarity"   : round(float(similarity), 3),
            "actual_return": actual_return,
            "distance"     : round(float(dists[i]), 3),
        })

    return results


def get_extreme_prediction(current_features: pd.Series, base_low: float,
                           base_mid: float, base_high: float) -> dict:
    """
    第二层极端事件预测
    当检测到高波动环境时，用历史情景匹配修正置信区间
    """
    feat = _load_feature_matrix()

    # 判断是否需要激活第二层
    volatility  = current_features.get("volatility", 0)
    vol_ratio   = current_features.get("vol_ratio", 1)
    vix         = current_features.get("VIX", 20)
    hormuz_z    = current_features.get("hormuz_tanker_zscore", 0)

    # 激活条件：任意一个触发
    is_extreme = (
        vol_ratio   > 2.0  or   # 当前波动率是历史均值2倍以上
        vix         > 30   or   # VIX恐慌指数超过30
        abs(hormuz_z) > 2.0     # 霍尔木兹航运异常
    )

    if not is_extreme:
        return {
            "activated"  : False,
            "pred_low"   : base_low,
            "pred_mid"   : base_mid,
            "pred_high"  : base_high,
            "similar_events": [],
            "scale_factor"  : 1.0,
        }

    # 找相似历史事件
    similar = find_similar_events(current_features, top_k=3)

    if not similar:
        return {
            "activated"     : True,
            "pred_low"      : base_low  * 2.0,
            "pred_mid"      : base_mid,
            "pred_high"     : base_high * 2.0,
            "similar_events": [],
            "scale_factor"  : 2.0,
        }

    # 用相似事件的实际涨跌幅加权计算修正区间
    weights  = np.array([ev["similarity"] for ev in similar])
    weights  = weights / weights.sum()
    returns  = np.array([ev["actual_return"] or base_mid for ev in similar])

    weighted_return = float(np.dot(weights, returns))

    # 计算历史极端事件的离散度作为区间宽度
    spread = float(np.std(returns)) if len(returns) > 1 else abs(weighted_return) * 0.5

    # 修正后的置信区间
    adj_mid  = weighted_return
    adj_low  = weighted_return - 2.0 * spread
    adj_high = weighted_return + 2.0 * spread

    # 取第一层和第二层的包络（更保守的低端，更激进的高端）
    final_low  = min(base_low,  adj_low)
    final_mid  = adj_mid
    final_high = max(base_high, adj_high)

    scale_factor = abs(final_high - final_low) / max(abs(base_high - base_low), 1e-6)

    return {
        "activated"     : True,
        "pred_low"      : final_low,
        "pred_mid"      : final_mid,
        "pred_high"     : final_high,
        "similar_events": similar,
        "scale_factor"  : round(scale_factor, 2),
        "weighted_return": round(weighted_return, 4),
        "vol_ratio"     : round(float(vol_ratio), 2),
        "vix"           : round(float(vix), 1),
    }


if __name__ == "__main__":
    # 测试：用feature_matrix最后一行模拟当前状态
    feat = _load_feature_matrix()
    latest = feat.iloc[-1]
    print(f"测试日期: {feat.index[-1].date()}")
    print(f"vol_ratio: {latest.get('vol_ratio', 'N/A')}")
    print(f"VIX: {latest.get('VIX', 'N/A')}")

    result = get_extreme_prediction(latest, -0.05, 0.02, 0.08)
    print(f"\n第二层激活: {result['activated']}")
    print(f"修正后区间: {result['pred_low']:.3f} / {result['pred_mid']:.3f} / {result['pred_high']:.3f}")
    if result["similar_events"]:
        print("\n最相似历史事件:")
        for ev in result["similar_events"]:
            print(f"  {ev['event']} (相似度:{ev['similarity']}) 实际涨跌:{ev['actual_return']:.3f}")
