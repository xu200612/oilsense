import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

# ── 国家关键词映射 ────────────────────────────────────────────
COUNTRY_KEYWORDS = {
    "伊朗"    : ["Iran", "Iranian", "Tehran", "Khamenei", "IRGC", "Hormuz"],
    "俄罗斯"  : ["Russia", "Russian", "Moscow", "Kremlin", "Putin", "Gazprom"],
    "沙特阿拉伯": ["Saudi", "Riyadh", "Aramco", "MBS", "OPEC"],
    "伊拉克"  : ["Iraq", "Iraqi", "Baghdad", "Basra"],
    "美国"    : ["US ", "USA", "America", "Washington", "Biden", "Trump", "Fed"],
    "阿联酋"  : ["UAE", "Dubai", "Abu Dhabi", "ADNOC"],
    "科威特"  : ["Kuwait", "Kuwaiti"],
    "挪威"    : ["Norway", "Norwegian", "Equinor", "Statoil"],
    "哈萨克斯坦": ["Kazakhstan", "Kazakh", "Tengiz", "KazMunaiGas"],
    "尼日利亚" : ["Nigeria", "Nigerian", "Lagos", "Niger Delta"],
    "利比亚"  : ["Libya", "Libyan", "Tripoli", "Benghazi"],
    "委内瑞拉" : ["Venezuela", "Venezuelan", "Maduro", "PDVSA", "Caracas"],
    "阿尔及利亚": ["Algeria", "Algerian", "Sonatrach"],
}

# ── 静态基础风险分（0~1，越高越危险）────────────────────────────
BASE_RISK = {
    "伊朗"    : 0.90,
    "俄罗斯"  : 0.80,
    "利比亚"  : 0.75,
    "委内瑞拉" : 0.70,
    "伊拉克"  : 0.60,
    "尼日利亚" : 0.55,
    "阿尔及利亚": 0.45,
    "哈萨克斯坦": 0.35,
    "阿联酋"  : 0.20,
    "科威特"  : 0.20,
    "沙特阿拉伯": 0.25,
    "挪威"    : 0.10,
    "美国"    : 0.15,
}

def simple_sentiment(text: str) -> float:
    """
    简单关键词情感打分（-1 到 +1）
    负值=利空（风险上升），正值=利多（风险下降）
    """
    text = text.lower()
    negative = [
        "war", "attack", "strike", "sanction", "conflict", "crisis",
        "threat", "tension", "explosion", "missile", "drone", "shutdown",
        "block", "close", "halt", "disruption", "cut", "ban", "embargo",
        "escalat", "hostil", "bomb", "kill", "invasion", "coup",
    ]
    positive = [
        "peace", "deal", "agreement", "ceasefire", "negotiat", "resolve",
        "diplomacy", "cooperat", "supply", "increase", "open", "resume",
        "stabiliz", "recover", "ease", "lift", "calm",
    ]
    neg_count = sum(1 for w in negative if w in text)
    pos_count = sum(1 for w in positive if w in text)
    total = neg_count + pos_count
    if total == 0:
        return 0.0
    return (pos_count - neg_count) / total

def compute_country_risk(days_back: int = 7) -> dict:
    """
    计算各国动态风险分数
    返回 {国家名: {score, level, color, news_sentiment, news_count, ...}}
    """
    # 加载新闻数据
    news_path = os.path.join(ROOT_DIR, "data", "raw", "news_data.csv")
    news = pd.DataFrame()
    if os.path.exists(news_path):
        news = pd.read_csv(news_path)
        news["date"] = pd.to_datetime(news["date"], errors="coerce")
        # 只取最近 days_back 天
        cutoff = pd.Timestamp.now() - timedelta(days=days_back)
        news   = news[news["date"] >= cutoff]

    # 加载 GDELT 全局冲突强度
    gdelt_path = os.path.join(ROOT_DIR, "data", "raw", "gdelt_sentiment.csv")
    gdelt_conflict_global = 0.5  # 默认中性
    if os.path.exists(gdelt_path):
        gdelt = pd.read_csv(gdelt_path, index_col=0, parse_dates=True)
        if "gdelt_conflict_intensity" in gdelt.columns:
            ci       = gdelt["gdelt_conflict_intensity"].dropna()
            latest   = ci.iloc[-1]
            ci_min   = ci.quantile(0.05)
            ci_max   = ci.quantile(0.95)
            # 归一化到 0~1
            gdelt_conflict_global = float(
                np.clip((latest - ci_min) / (ci_max - ci_min + 1e-6), 0, 1)
            )

    results = {}
    for country, keywords in COUNTRY_KEYWORDS.items():
        # 提取该国相关新闻
        if len(news) > 0:
            pattern = "|".join(keywords)
            mask    = news["title"].str.contains(pattern, case=False, na=False)
            c_news  = news[mask]
        else:
            c_news = pd.DataFrame()

        # 计算新闻情感均值
        if len(c_news) > 0:
            sentiments     = c_news["title"].apply(simple_sentiment)
            news_sentiment = float(sentiments.mean())  # -1~+1
            news_count     = len(c_news)
        else:
            news_sentiment = 0.0
            news_count     = 0

        # 新闻情感转换为风险分（负面情感→高风险）
        # news_sentiment: -1(极负面)→风险1.0，+1(极正面)→风险0.0
        news_risk = (1.0 - news_sentiment) / 2.0  # 映射到 0~1

        # 综合风险分
        base     = BASE_RISK.get(country, 0.3)
        score    = (
            news_risk              * 0.55 +
            gdelt_conflict_global  * 0.15 +
            base                   * 0.30
        )
        score = float(np.clip(score, 0.0, 1.0))

        # 映射风险等级
        if score >= 0.70:
            level, color = "高风险",   "#e74c3c"
        elif score >= 0.45:
            level, color = "中等风险", "#e67e22"
        elif score >= 0.25:
            level, color = "低风险",   "#f1c40f"
        else:
            level, color = "极低风险", "#2ecc71"

        results[country] = {
            "score"          : round(score, 3),
            "level"          : level,
            "color"          : color,
            "news_sentiment" : round(news_sentiment, 3),
            "news_count"     : news_count,
            "gdelt_global"   : round(gdelt_conflict_global, 3),
            "base_risk"      : base,
            "updated_at"     : datetime.now().strftime("%Y-%m-%d %H:%M"),
        }

    return results


if __name__ == "__main__":
    print("计算国别动态风险...")
    results = compute_country_risk()
    print(f"{'国家':<10} {'风险等级':<8} {'综合分':<8} "
          f"{'新闻情感':<10} {'新闻数':<6} {'GDELT全局'}")
    print("-" * 60)
    for country, r in sorted(results.items(),
                              key=lambda x: x[1]["score"], reverse=True):
        print(f"{country:<10} {r['level']:<8} {r['score']:<8} "
              f"{r['news_sentiment']:<10} {r['news_count']:<6} "
              f"{r['gdelt_global']}")
