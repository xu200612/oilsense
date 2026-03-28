import os
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

OPEC_DATES = [
    "2020-03-06","2020-06-06","2020-07-15","2020-11-30",
    "2021-03-04","2021-04-01","2021-05-27","2021-07-01",
    "2021-09-01","2021-10-04","2021-11-04","2022-02-02",
    "2022-03-02","2022-04-05","2022-05-05","2022-06-02",
    "2022-09-05","2022-10-05","2022-12-04","2023-02-01",
    "2023-04-03","2023-06-04","2023-08-04","2023-11-26",
    "2024-02-01","2024-03-03","2024-06-02","2024-11-03",
    "2025-02-03","2025-03-03","2025-05-05","2025-06-01",
]

def load_and_merge():
    print("正在加载数据...")
    oil = pd.read_csv(os.path.join(ROOT_DIR, "data", "raw", "oil_prices.csv"),
                      index_col=0, parse_dates=True)
    macro = pd.read_csv(os.path.join(ROOT_DIR, "data", "raw", "macro_data.csv"),
                        index_col=0, parse_dates=True)
    sentiment = pd.read_csv(
        os.path.join(ROOT_DIR, "data", "processed", "daily_sentiment.csv"),
        parse_dates=["date"]).set_index("date")

    df = oil.copy()
    df = df.join(macro, how="left")
    df = df.join(sentiment[["sentiment_score", "news_count",
                             "geopolitics_flag", "policy_flag"]], how="left")
    for col in ["sentiment_score", "news_count", "geopolitics_flag", "policy_flag"]:
        df[col] = df[col].fillna(0)

    # 接入 GDELT 历史地缘政治情感数据
    gdelt_path = os.path.join(ROOT_DIR, "data", "raw", "gdelt_sentiment.csv")
    if os.path.exists(gdelt_path):
        gdelt = pd.read_csv(gdelt_path, index_col=0, parse_dates=True)
        df = df.join(gdelt, how="left")
        gdelt_cols = ["gdelt_goldstein", "gdelt_tone", "gdelt_conflict_cnt",
                      "gdelt_coop_cnt", "gdelt_conflict_intensity", "gdelt_mentions"]
        for col in gdelt_cols:
            if col in df.columns:
                df[col] = df[col].ffill()
                df[col] = df[col].fillna(df[col].rolling(30, min_periods=1).mean())
                df[col] = df[col].fillna(0)
        print("  GDELT 数据已接入，覆盖 " + str(gdelt[gdelt.notna().any(axis=1)].shape[0]) + " 天")
    else:
        print("  GDELT 数据未找到，跳过（运行 python fetch_gdelt.py history 生成）")
    # 接入 IMF PortWatch 咽喉点航运数据
    portwatch_path = os.path.join(ROOT_DIR, "data", "raw", "portwatch_chokepoints.csv")
    if os.path.exists(portwatch_path):
        portwatch = pd.read_csv(portwatch_path, index_col=0, parse_dates=True)
        # 只取关键特征列
        pw_cols = [c for c in portwatch.columns if any(x in c for x in [
            "cp6_tanker", "cp4_tanker", "cp1_tanker", "cp5_tanker",
            "cp11_tanker", "cp7_tanker",
            "hormuz_tanker_ma7", "hormuz_tanker_zscore", "hormuz_blocked",
            "mandeb_tanker_ma7", "mandeb_blocked",
            "cape_reroute_signal"
        ])]
        portwatch = portwatch[pw_cols]
        # PortWatch 是周频，前向填充到日频
        portwatch = portwatch.resample("D").last().ffill()
        df = df.join(portwatch, how="left")
        # 缺失值用滚动均值填充
        for col in pw_cols:
            if col in df.columns:
                df[col] = df[col].ffill()
                df[col] = df[col].fillna(df[col].rolling(30, min_periods=1).mean())
                df[col] = df[col].fillna(0)
        print("  PortWatch 数据已接入：" + str(len(pw_cols)) + " 个字段")
    else:
        print("  PortWatch 数据未找到，跳过（运行 python fetch_portwatch.py 生成）")

    df.ffill(inplace=True)
    df.dropna(inplace=True)
    print("  合并后数据：" + str(len(df)) + " 条，" + str(df.shape[1]) + " 个字段")
    return df

def build_features(df, target_col="WTI", horizon=10):
    print("正在构建特征（预测未来 " + str(horizon) + " 日）...")
    feat = df.copy()

    # 技术指标
    feat["return_1d"]   = feat[target_col].pct_change(1)
    feat["return_5d"]   = feat[target_col].pct_change(5)
    feat["return_10d"]  = feat[target_col].pct_change(10)
    feat["return_20d"]  = feat[target_col].pct_change(20)
    feat["ma_5"]        = feat[target_col].rolling(5).mean()
    feat["ma_20"]       = feat[target_col].rolling(20).mean()
    feat["ma_60"]       = feat[target_col].rolling(60).mean()
    feat["ma_ratio"]    = feat["ma_5"] / feat["ma_20"]
    feat["ma_ratio_60"] = feat["ma_20"] / feat["ma_60"]
    feat["volatility"]  = feat[target_col].rolling(10).std()
    feat["vol_ratio"]   = feat["volatility"] / feat[target_col].rolling(60).std()
    feat["high_vol"]    = (feat["volatility"] > feat["volatility"].rolling(60).mean()).astype(int)

    if "Brent" in feat.columns:
        feat["wti_brent_spread"] = feat[target_col] - feat["Brent"]

    # OPEC 事件窗口
    opec_dates = pd.to_datetime(OPEC_DATES)
    feat["opec_flag"] = 0
    for od in opec_dates:
        mask = (feat.index >= od - pd.Timedelta(days=5)) & \
               (feat.index <= od + pd.Timedelta(days=5))
        feat.loc[mask, "opec_flag"] = 1

    # GDELT 衍生特征
    if "gdelt_goldstein" in feat.columns:
        # 冲突升级信号：Goldstein 5日变化
        feat["gdelt_goldstein_chg"] = feat["gdelt_goldstein"].diff(5)
        # 冲突强度滚动均值
        feat["gdelt_conflict_ma5"]  = feat["gdelt_conflict_cnt"].rolling(5).mean()
        # 情感极性变化速度
        feat["gdelt_tone_chg"]      = feat["gdelt_tone"].diff(3)
        print("  GDELT 衍生特征已构建")

    # IMF PortWatch 航运衍生特征
    portwatch_cols = [
        "cp6_tanker", "cp4_tanker", "cp1_tanker", "cp5_tanker",
        "cp11_tanker", "cp7_tanker",
        "hormuz_tanker_ma7", "hormuz_tanker_zscore", "hormuz_blocked",
        "mandeb_tanker_ma7", "mandeb_blocked",
        "cape_reroute_signal",
    ]
    portwatch_present = [c for c in portwatch_cols if c in feat.columns]
    if portwatch_present:
        print("  PortWatch 航运特征已纳入：" + str(len(portwatch_present)) + " 个字段")

    # 目标变量：严格截断防止数据泄露
    feat["target"] = feat[target_col].pct_change(horizon).shift(-horizon)
    feat = feat.iloc[:-horizon]
    feat.dropna(inplace=True)

    print("  最后5行 target 验证（应无未来泄露）:")
    print(feat["target"].tail())
    print("  target 均值: " + str(round(feat["target"].mean(), 6)))

    # Enhanced 特征集（含 GDELT 和情感因子）
    all_feature_cols = [
        "return_1d", "return_5d", "return_10d", "return_20d",
        "ma_ratio", "ma_ratio_60", "volatility", "vol_ratio", "high_vol",
        "wti_brent_spread",
        "DXY", "US_CPI", "FED_RATE", "US10Y", "VIX", "US_PPI",
        "US_EPU", "GLOBAL_EPU",
        "opec_flag",
        # NewsAPI 情感因子
        "sentiment_score", "news_count", "geopolitics_flag", "policy_flag",
        # GDELT 地缘政治因子
        "gdelt_goldstein", "gdelt_tone",
        "gdelt_conflict_cnt", "gdelt_coop_cnt",
        "gdelt_conflict_intensity", "gdelt_mentions",
        "gdelt_goldstein_chg", "gdelt_conflict_ma5", "gdelt_tone_chg",
        # IMF PortWatch 航运因子
        "cp6_tanker", "cp4_tanker", "cp1_tanker", "cp5_tanker",
        "cp11_tanker", "cp7_tanker",
        "hormuz_tanker_ma7", "hormuz_tanker_zscore", "hormuz_blocked",
        "mandeb_tanker_ma7", "mandeb_blocked",
        "cape_reroute_signal",
        # 宏观补充
        "US_PPI",
    ]

    # Baseline 特征集（仅传统量化因子）
    baseline_feature_cols = [
        "return_1d", "return_5d", "return_10d", "return_20d",
        "ma_ratio", "ma_ratio_60", "volatility", "vol_ratio", "high_vol",
        "wti_brent_spread",
        "DXY", "US_CPI", "FED_RATE", "US10Y", "VIX", "US_PPI",
        "US_EPU", "GLOBAL_EPU",
        "opec_flag",
    ]

    all_feature_cols      = [c for c in all_feature_cols      if c in feat.columns]
    baseline_feature_cols = [c for c in baseline_feature_cols if c in feat.columns]

    print("  Enhanced特征数：" + str(len(all_feature_cols)))
    print("  Baseline特征数：" + str(len(baseline_feature_cols)))
    print("  样本数量：" + str(len(feat)))
    return feat, all_feature_cols, baseline_feature_cols

def train_models(feat, all_feature_cols, baseline_feature_cols):
    print("正在训练模型...")

    # 训练和测试只用黑天鹅之前的数据，避免极端事件污染验证集导致early stopping失效
    # 黑天鹅期间由 extreme_scenario.py 的第二层模型负责
    TRAIN_CUTOFF = pd.Timestamp("2026-01-01")
    feat_train = feat[feat.index < TRAIN_CUTOFF]
    print("  训练数据截止：" + str(TRAIN_CUTOFF.date()) +
          "（共 " + str(len(feat_train)) + " 条，排除黑天鹅期间）")

    X = feat_train[all_feature_cols]
    y = feat_train["target"]

    split     = int(len(X) * 0.8)
    val_split = int(split * 0.85)

    X_train, X_test = X.iloc[:split],           X.iloc[split:]
    y_train, y_test = y.iloc[:split],           y.iloc[split:]
    X_tr,    X_val  = X_train.iloc[:val_split], X_train.iloc[val_split:]
    y_tr,    y_val  = y_train.iloc[:val_split], y_train.iloc[val_split:]

    print("  训练集：" + str(len(X_train)) + " 条  测试集：" + str(len(X_test)) + " 条")

    models = {}

    # ── Baseline ───────────────────────────────────────────────────────────
    for quantile, label in [(0.1, "low"), (0.9, "high")]:
        model = XGBRegressor(
            n_estimators=500, max_depth=3, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.7, min_child_weight=2,
            objective="reg:quantileerror", quantile_alpha=quantile,
            random_state=42, early_stopping_rounds=80, eval_metric="mae"
        )
        model.fit(X_tr[baseline_feature_cols], y_tr,
                  eval_set=[(X_val[baseline_feature_cols], y_val)], verbose=False)
        models["baseline_" + label] = (model, baseline_feature_cols)
        print("  Baseline " + label + " ✓  (迭代: " + str(model.best_iteration) + ")")

    baseline_mid = XGBRegressor(
        n_estimators=500, max_depth=3, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=2,
        objective="reg:squarederror",
        random_state=42, early_stopping_rounds=80, eval_metric="mae"
    )
    baseline_mid.fit(X_tr[baseline_feature_cols], y_tr,
                     eval_set=[(X_val[baseline_feature_cols], y_val)], verbose=False)
    models["baseline_mid"] = (baseline_mid, baseline_feature_cols)
    print("  Baseline mid ✓  (迭代: " + str(baseline_mid.best_iteration) + ")")

    # ── Enhanced ───────────────────────────────────────────────────────────
    for quantile, label in [(0.1, "low"), (0.9, "high")]:
        model = XGBRegressor(
            n_estimators=500, max_depth=3, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.7, min_child_weight=2,
            objective="reg:quantileerror", quantile_alpha=quantile,
            random_state=42, early_stopping_rounds=80, eval_metric="mae"
        )
        model.fit(X_tr[all_feature_cols], y_tr,
                  eval_set=[(X_val[all_feature_cols], y_val)], verbose=False)
        models["enhanced_" + label] = (model, all_feature_cols)
        print("  Enhanced " + label + " ✓  (迭代: " + str(model.best_iteration) + ")")

    enhanced_mid = XGBRegressor(
        n_estimators=500, max_depth=3, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=2,
        objective="reg:squarederror",
        random_state=42, early_stopping_rounds=80, eval_metric="mae"
    )
    enhanced_mid.fit(X_tr[all_feature_cols], y_tr,
                     eval_set=[(X_val[all_feature_cols], y_val)], verbose=False)
    models["enhanced_mid"] = (enhanced_mid, all_feature_cols)
    print("  Enhanced mid ✓  (迭代: " + str(enhanced_mid.best_iteration) + ")")

    # ── 评估 ───────────────────────────────────────────────────────────────
    mid_model, cols = models["enhanced_mid"]
    y_pred = mid_model.predict(X_test[cols])
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    mae    = mean_absolute_error(y_test, y_pred)
    direction_acc = np.mean(np.sign(y_pred) == np.sign(y_test))

    signal_mask = np.abs(y_test) > 0.01
    signal_acc  = (np.mean(np.sign(y_pred[signal_mask]) == np.sign(y_test[signal_mask]))
                   if signal_mask.sum() > 0 else direction_acc)

    # Baseline 对比
    bm, bcols  = models["baseline_mid"]
    yb_pred    = bm.predict(X_test[bcols])
    baseline_acc = np.mean(np.sign(yb_pred) == np.sign(y_test))

    print("  ── 模型评估（10日预测窗口）──")
    print("  RMSE                  : " + str(round(rmse, 4)))
    print("  MAE                   : " + str(round(mae, 4)))
    print("  Enhanced 方向准确率    : " + str(round(direction_acc * 100, 2)) + "%")
    print("  Baseline 方向准确率    : " + str(round(baseline_acc * 100, 2)) + "%")
    print("  有效信号方向准确率     : " + str(round(signal_acc * 100, 2)) +
          "%  (|涨跌|>1%, 共" + str(signal_mask.sum()) + "条)")
    print("  情感因子提升           : +" +
          str(round((direction_acc - baseline_acc) * 100, 2)) + "%")

    return models, X_test, y_test

def save_results(models, feat, all_feature_cols):
    print("正在保存模型...")
    model_dir = os.path.join(ROOT_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    model_feature_map = {}
    for name, (model, cols) in models.items():
        model.save_model(os.path.join(model_dir, name + ".json"))
        model_feature_map[name] = cols

    joblib.dump(model_feature_map, os.path.join(model_dir, "model_feature_map.pkl"))
    joblib.dump(all_feature_cols,  os.path.join(model_dir, "feature_cols.pkl"))
    feat.to_csv(os.path.join(ROOT_DIR, "data", "processed", "feature_matrix.csv"))

    mid_model, cols = models["enhanced_mid"]
    importance = pd.DataFrame({
        "feature"   : cols,
        "importance": mid_model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("  ── 特征重要性 Top 12 ──")
    print(importance.head(12).to_string(index=False))
    importance.to_csv(
        os.path.join(ROOT_DIR, "data", "processed", "feature_importance.csv"),
        index=False
    )
    print("所有模型已保存至 models/ 目录")

if __name__ == "__main__":
    df                                    = load_and_merge()
    feat, all_feature_cols, baseline_cols = build_features(df, target_col="WTI", horizon=10)
    models, X_test, y_test                = train_models(feat, all_feature_cols, baseline_cols)
    save_results(models, feat, all_feature_cols)
    print("Step 4 完成！")