import os

import joblib
import numpy as np
import pandas as pd
from xgboost import DMatrix, XGBRegressor


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODEL_DIR = os.path.join(ROOT_DIR, "models")

# 特征分组——与 github_oilsense 的 enhanced_mid 实际特征集对齐
FACTOR_GROUPS = {
    "价格动量": [
        "return_1d", "return_5d", "return_10d", "return_20d",
        "ma_ratio", "ma_ratio_60", "volatility", "vol_ratio", "high_vol",
        "wti_brent_spread",
    ],
    "宏观金融": [
        "DXY", "US_CPI", "FED_RATE", "US10Y", "VIX", "US_PPI",
        "US_EPU", "GLOBAL_EPU",
    ],
    "GDELT地缘政治": [
        "gdelt_goldstein", "gdelt_tone", "gdelt_conflict_cnt",
        "gdelt_coop_cnt", "gdelt_conflict_intensity", "gdelt_mentions",
        "gdelt_goldstein_chg", "gdelt_conflict_ma5", "gdelt_tone_chg",
    ],
    "航运咽喉点": [
        "cp6_tanker", "cp4_tanker", "cp1_tanker", "cp5_tanker",
        "cp11_tanker", "cp7_tanker",
        "hormuz_tanker_ma7", "hormuz_tanker_zscore", "hormuz_blocked",
        "mandeb_tanker_ma7", "mandeb_blocked", "cape_reroute_signal",
    ],
    "新闻情感": [
        "sentiment_score", "news_count", "geopolitics_flag", "policy_flag",
        "opec_flag",
    ],
}


def _feature_group(feature: str) -> str:
    for group, cols in FACTOR_GROUPS.items():
        if feature in cols:
            return group
    return "其他"


def _load_model_and_features():
    fmap = joblib.load(os.path.join(MODEL_DIR, "model_feature_map.pkl"))
    cols = fmap["enhanced_mid"]
    model = XGBRegressor()
    model.load_model(os.path.join(MODEL_DIR, "enhanced_mid.json"))
    return model, cols


def compute_shap_outputs(window: int = 90, top_n: int = 16):
    feat = pd.read_csv(
        os.path.join(PROCESSED_DIR, "feature_matrix.csv"),
        index_col=0,
        parse_dates=True,
    )
    model, cols = _load_model_and_features()
    data = feat[cols + ["target"]].replace([np.inf, -np.inf], np.nan).dropna()
    data = data.tail(window)
    x = data[cols]

    booster = model.get_booster()
    contrib = booster.predict(DMatrix(x, feature_names=cols), pred_contribs=True)
    shap_values = pd.DataFrame(contrib[:, :-1], index=x.index, columns=cols)
    bias = pd.Series(contrib[:, -1], index=x.index, name="bias")

    global_shap = (
        shap_values.abs()
        .mean()
        .rename("mean_abs_shap")
        .reset_index()
        .rename(columns={"index": "feature"})
    )
    global_shap["mean_shap"] = global_shap["feature"].map(shap_values.mean())
    global_shap["group"] = global_shap["feature"].map(_feature_group)
    global_shap = global_shap.sort_values("mean_abs_shap", ascending=False)

    latest = shap_values.tail(1).T.reset_index()
    latest.columns = ["feature", "shap_value"]
    latest["abs_shap"] = latest["shap_value"].abs()
    latest["direction"] = np.where(latest["shap_value"] >= 0, "push_up", "push_down")
    latest["group"] = latest["feature"].map(_feature_group)
    latest = latest.sort_values("abs_shap", ascending=False)
    latest["prediction_bias"] = float(bias.iloc[-1])
    latest["prediction_mid"] = float(shap_values.iloc[-1].sum() + bias.iloc[-1])
    latest["asof_date"] = shap_values.index[-1].strftime("%Y-%m-%d")

    top_features = list(global_shap.head(top_n)["feature"])
    matrix = shap_values[top_features].copy()
    matrix.index.name = "date"
    matrix = matrix.reset_index()

    group_matrix = pd.DataFrame(index=shap_values.index)
    for group in sorted(set(_feature_group(c) for c in cols)):
        group_cols = [c for c in cols if _feature_group(c) == group and c in shap_values.columns]
        if group_cols:
            group_matrix[group] = shap_values[group_cols].sum(axis=1)
    group_matrix.index.name = "date"
    group_matrix = group_matrix.reset_index()

    # 当日预测的 SHAP 分解（用 latest_features.csv）
    latest_feature_path = os.path.join(PROCESSED_DIR, "latest_features.csv")
    current = pd.DataFrame()
    if os.path.exists(latest_feature_path):
        latest_features = pd.read_csv(latest_feature_path, index_col=0, parse_dates=True)
        avail_cols = [c for c in cols if c in latest_features.columns]
        if avail_cols and len(avail_cols) == len(cols):
            current_x = latest_features[cols].replace([np.inf, -np.inf], np.nan).ffill().dropna()
            if not current_x.empty:
                current_x = current_x.tail(1)
                current_contrib = booster.predict(
                    DMatrix(current_x, feature_names=cols),
                    pred_contribs=True,
                )
                current = pd.DataFrame({
                    "feature": cols,
                    "shap_value": current_contrib[0, :-1],
                })
                current["abs_shap"] = current["shap_value"].abs()
                current["direction"] = np.where(current["shap_value"] >= 0, "push_up", "push_down")
                current["group"] = current["feature"].map(_feature_group)
                current["prediction_bias"] = float(current_contrib[0, -1])
                current["prediction_mid"] = float(current_contrib[0, :-1].sum() + current_contrib[0, -1])
                current["asof_date"] = current_x.index[-1].strftime("%Y-%m-%d")
                current = current.sort_values("abs_shap", ascending=False)

    global_shap.to_csv(os.path.join(PROCESSED_DIR, "shap_global.csv"), index=False, encoding="utf-8-sig")
    latest.to_csv(os.path.join(PROCESSED_DIR, "shap_latest.csv"), index=False, encoding="utf-8-sig")
    matrix.to_csv(os.path.join(PROCESSED_DIR, "shap_matrix.csv"), index=False, encoding="utf-8-sig")
    group_matrix.to_csv(os.path.join(PROCESSED_DIR, "shap_group_matrix.csv"), index=False, encoding="utf-8-sig")
    if not current.empty:
        current.to_csv(os.path.join(PROCESSED_DIR, "shap_current.csv"), index=False, encoding="utf-8-sig")

    print("SHAP 输出已保存:")
    print("  解释日期:", latest["asof_date"].iloc[0])
    print("  Top 5 贡献因子:")
    for _, row in latest.head(5).iterrows():
        print(f"    {row['feature']}: {row['shap_value']*100:+.2f}pp ({row['direction']})")


if __name__ == "__main__":
    compute_shap_outputs()
