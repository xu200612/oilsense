import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from xgboost import XGBRegressor
from dotenv import load_dotenv

# ── 路径设置 ───────────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

CRISIS_EVENTS = [
    {"date": "2020-03-09", "label": "新冠+油价战争(2020-03-09)", "color": "#e74c3c"},
    {"date": "2022-02-24", "label": "俄乌冲突爆发(2022-02-24)",  "color": "#e67e22"},
    {"date": "2023-10-07", "label": "以哈冲突爆发(2023-10-07)",  "color": "#9b59b6"},
    {"date": "2025-01-20", "label": "特朗普就职(2025-01-20)",    "color": "#2980b9"},
]

# ── 加载数据和模型 ─────────────────────────────────────────────────────────
def load_assets():
    print("正在加载特征矩阵和模型...")

    feat = pd.read_csv(
        os.path.join(ROOT_DIR, "data", "processed", "feature_matrix.csv"),
        index_col=0, parse_dates=True
    )

    model_feature_map = joblib.load(
        os.path.join(ROOT_DIR, "models", "model_feature_map.pkl")
    )

    models = {}
    for name in ["enhanced_low", "enhanced_mid", "enhanced_high",
                 "baseline_low", "baseline_mid", "baseline_high"]:
        m = XGBRegressor()
        m.load_model(os.path.join(ROOT_DIR, "models", name + ".json"))
        models[name] = m

    print("  加载完成，数据行数：" + str(len(feat)))
    return feat, model_feature_map, models

# ── 生成全样本预测 ─────────────────────────────────────────────────────────
def generate_predictions(feat, model_feature_map, models):
    print("正在生成全样本预测...")

    pred_df = pd.DataFrame(index=feat.index)
    pred_df["WTI_actual"]    = feat["WTI"]
    pred_df["target_actual"] = feat["target"]

    for name, model in models.items():
        cols = model_feature_map[name]
        pred_df["pred_" + name] = model.predict(feat[cols])

    print("  预测完成，共 " + str(len(pred_df)) + " 条")
    return pred_df

# ── 绘图1：油价走势 + 风险区间 + 危机标注 ─────────────────────────────────
def plot_price_with_risk(pred_df):
    print("正在绘制油价风险区间图...")

    fig, axes = plt.subplots(2, 1, figsize=(16, 12), sharex=True)

    ax1 = axes[0]
    ax1.plot(pred_df.index, pred_df["WTI_actual"],
             color="#2c3e50", linewidth=1.5, label="WTI 油价（实际）")
    ax1.set_ylabel("价格（美元/桶）", fontsize=12)
    ax1.set_title("WTI 原油价格走势与 OilSense 风险预警", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(alpha=0.3)

    y_max = pred_df["WTI_actual"].max()
    for event in CRISIS_EVENTS:
        edate = pd.Timestamp(event["date"])
        if pred_df.index.min() <= edate <= pred_df.index.max():
            ax1.axvline(x=edate, color=event["color"],
                        linewidth=1.5, linestyle="--", alpha=0.8)
            ax1.text(edate, y_max * 0.95, event["label"],
                     fontsize=8, color=event["color"], ha="center", va="top",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    ax2 = axes[1]
    ax2.fill_between(pred_df.index,
                     pred_df["pred_enhanced_low"],
                     pred_df["pred_enhanced_high"],
                     alpha=0.25, color="#3498db", label="风险区间（10%~90%）")
    ax2.plot(pred_df.index, pred_df["pred_enhanced_mid"],
             color="#3498db", linewidth=1.2, label="预测中位数（Enhanced）")
    ax2.plot(pred_df.index, pred_df["pred_baseline_mid"],
             color="#95a5a6", linewidth=1.0, linestyle="--",
             alpha=0.7, label="预测中位数（Baseline）")
    ax2.plot(pred_df.index, pred_df["target_actual"],
             color="#e74c3c", linewidth=0.8, alpha=0.6, label="实际涨跌幅")
    ax2.axhline(y=0, color="black", linewidth=0.8)

    for event in CRISIS_EVENTS:
        edate = pd.Timestamp(event["date"])
        if pred_df.index.min() <= edate <= pred_df.index.max():
            ax2.axvline(x=edate, color=event["color"],
                        linewidth=1.5, linestyle="--", alpha=0.8)

    ax2.set_ylabel("预测5日涨跌幅", fontsize=12)
    ax2.set_xlabel("日期", fontsize=12)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    plt.tight_layout()
    out_path = os.path.join(ROOT_DIR, "data", "processed", "backtest_price_risk.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print("  已保存：backtest_price_risk.png")
    plt.show()

# ── 绘图2：特征重要性 ──────────────────────────────────────────────────────
def plot_feature_importance():
    print("正在绘制特征重要性图...")

    importance = pd.read_csv(
        os.path.join(ROOT_DIR, "data", "processed", "feature_importance.csv")
    ).head(12)

    sentiment_cols = ["sentiment_score", "geopolitics_flag", "policy_flag", "news_count"]
    colors = ["#e74c3c" if f in sentiment_cols else "#3498db"
              for f in importance["feature"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(importance["feature"][::-1],
            importance["importance"][::-1],
            color=colors[::-1], edgecolor="white", height=0.7)
    ax.set_xlabel("特征重要性得分", fontsize=12)
    ax.set_title("OilSense 模型特征重要性（红色=LLM情感因子）",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#e74c3c", label="LLM 情感因子"),
        Patch(facecolor="#3498db", label="传统量化因子")
    ], loc="lower right")

    plt.tight_layout()
    out_path = os.path.join(ROOT_DIR, "data", "processed", "feature_importance.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print("  已保存：feature_importance.png")
    plt.show()

# ── 绘图3：危机窗口放大图 ──────────────────────────────────────────────────
def plot_crisis_zoom(pred_df):
    print("正在绘制危机窗口放大图...")

    valid_events = [e for e in CRISIS_EVENTS
                    if pred_df.index.min() <= pd.Timestamp(e["date"]) <= pred_df.index.max()]

    if not valid_events:
        print("  无有效危机节点，跳过")
        return

    fig, axes = plt.subplots(1, len(valid_events),
                             figsize=(6 * len(valid_events), 5))
    if len(valid_events) == 1:
        axes = [axes]

    for ax, event in zip(axes, valid_events):
        edate  = pd.Timestamp(event["date"])
        start  = edate - pd.Timedelta(days=30)
        end    = edate + pd.Timedelta(days=30)
        window = pred_df.loc[start:end]

        if len(window) < 5:
            continue

        ax.fill_between(window.index,
                        window["pred_enhanced_low"],
                        window["pred_enhanced_high"],
                        alpha=0.25, color="#3498db")
        ax.plot(window.index, window["pred_enhanced_mid"],
                color="#3498db", linewidth=1.5, label="预测中位数")
        ax.plot(window.index, window["target_actual"],
                color="#e74c3c", linewidth=1.2, alpha=0.8, label="实际涨跌幅")
        ax.axvline(x=edate, color=event["color"], linewidth=2, linestyle="--")
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_title(event["label"], fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

    plt.suptitle("OilSense 危机事件窗口回测", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(ROOT_DIR, "data", "processed", "backtest_crisis_zoom.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print("  已保存：backtest_crisis_zoom.png")
    plt.show()

def plot_model_comparison(pred_df):
    print("正在绘制模型对比图...")

    # 只取测试集（后20%）评估，避免训练集污染
    test_start = pred_df.index[int(len(pred_df) * 0.8)]
    test_df    = pred_df.loc[test_start:]

    actual  = test_df["target_actual"]
    results = {}

    for version in ["enhanced", "baseline"]:
        mid   = test_df["pred_" + version + "_mid"]
        valid = actual.notna() & mid.notna()
        acc   = np.mean(np.sign(mid[valid]) == np.sign(actual[valid]))
        results[version] = round(acc * 100, 2)

    print("  测试集方向准确率 - Baseline: " + str(results["baseline"]) + "%  Enhanced: " + str(results["enhanced"]) + "%")

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(
        ["Baseline\n（无情感因子）", "Enhanced\n（含LLM情感因子）"],
        [results["baseline"], results["enhanced"]],
        color=["#95a5a6", "#e74c3c"], width=0.5, edgecolor="white"
    )
    ax.axhline(y=50, color="black", linewidth=1,
               linestyle="--", label="随机基准（50%）")

    for bar, val in zip(bars, [results["baseline"], results["enhanced"]]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(val) + "%", ha="center", va="bottom",
                fontsize=13, fontweight="bold")

    ax.set_ylabel("方向准确率（%）", fontsize=12)
    ax.set_title("Baseline vs Enhanced 模型方向准确率对比\n（样本外测试集）",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(45, max(results.values()) + 8)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(ROOT_DIR, "data", "processed", "model_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print("  已保存：model_comparison.png")
    plt.show()

# ── 主程序 ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    feat, model_feature_map, models = load_assets()
    pred_df = generate_predictions(feat, model_feature_map, models)
    plot_price_with_risk(pred_df)
    plot_feature_importance()
    plot_crisis_zoom(pred_df)
    plot_model_comparison(pred_df)
    print("Step 5 完成！所有回测图表生成成功。")
