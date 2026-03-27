# sentiment_analysis.py
import os
import json
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime, timedelta

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

DETAIL_PATH = os.path.join(ROOT_DIR, "data", "processed", "news_sentiment_detail.csv")
FACTOR_PATH = os.path.join(ROOT_DIR, "data", "processed", "daily_sentiment.csv")


# ── 单条新闻分析 ──────────────────────────────────────────────────────────
def analyze_sentiment(title: str, description: str) -> dict:
    text   = f"Title: {title}\nDescription: {description or 'N/A'}"
    prompt = f"""You are a professional energy market analyst. Analyze the following news and assess its impact on crude oil prices.

Return a JSON object with exactly these fields:
- "sentiment": one of "bullish", "bearish", "neutral"
- "score": float from -1.0 (strongly bearish) to +1.0 (strongly bullish)
- "confidence": float from 0.0 to 1.0
- "event_type": one of "supply", "demand", "geopolitics", "policy", "macro", "other"
- "impact_duration": one of "short" (1-3 days), "medium" (1-2 weeks), "long" (1+ month)
- "key_entities": list of up to 3 key countries/organizations mentioned
- "reason": one sentence explanation in English

News: {text}

Respond with valid JSON only."""

    try:
        response = client.chat.completions.create(
            model       = "deepseek-chat",
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.1,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())
    except Exception as e:
        return {
            "sentiment"      : "neutral",
            "score"          : 0.0,
            "confidence"     : 0.0,
            "event_type"     : "other",
            "impact_duration": "short",
            "key_entities"   : [],
            "reason"         : f"analysis failed: {str(e)[:50]}",
        }


# ── 增量分析：只处理未分析过的新闻 ───────────────────────────────────────
def incremental_sentiment_analysis(max_articles=100):
    """
    只分析新增的新闻，避免重复消耗API额度
    max_articles: 每次最多处理多少条新文章
    """
    print("\n[情感分析] 增量处理...")

    # 读取原始新闻
    news_path = os.path.join(ROOT_DIR, "data", "raw", "news_data.csv")
    if not os.path.exists(news_path):
        print("  新闻数据不存在，跳过")
        return

    news_df = pd.read_csv(news_path)
    print(f"  原始新闻共 {len(news_df)} 条")

    # 读取已分析结果
    if os.path.exists(DETAIL_PATH):
        done_df       = pd.read_csv(DETAIL_PATH)
        analyzed_titles = set(done_df["title"].tolist())
        print(f"  已分析 {len(done_df)} 条，跳过")
    else:
        done_df         = pd.DataFrame()
        analyzed_titles = set()

    # 找出未分析的新闻，优先处理最新的
    pending = news_df[~news_df["title"].isin(analyzed_titles)].copy()
    pending = pending.sort_values("date", ascending=False).head(max_articles)
    print(f"  待分析 {len(pending)} 条（本次最多处理 {max_articles} 条）")

    if len(pending) == 0:
        print("  无新内容需要分析")
        return

    results = []
    for idx, (_, row) in enumerate(pending.iterrows()):
        result = analyze_sentiment(
            title       = str(row["title"]),
            description = str(row.get("description", "")),
        )
        result["date"]    = row["date"]
        result["title"]   = row["title"]
        result["source"]  = row.get("source", "")
        result["url"]     = row.get("url", "")
        result["keyword"] = row.get("keyword", "")
        results.append(result)

        if (idx + 1) % 10 == 0:
            print(f"  进度: {idx+1}/{len(pending)}")
        time.sleep(0.4)

    # 合并并保存
    new_df   = pd.DataFrame(results)
    combined = pd.concat([done_df, new_df], ignore_index=True) if len(done_df) > 0 else new_df
    combined = combined.drop_duplicates(subset=["title"])
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined = combined[combined["date"] >= datetime.today() - timedelta(days=90)]
    combined["date"] = combined["date"].dt.strftime("%Y-%m-%d")
    combined.sort_values("date", ascending=False, inplace=True)
    combined.to_csv(DETAIL_PATH, index=False, encoding="utf-8-sig")
    print(f"  情感详情已保存，共 {len(combined)} 条")

    # 重新聚合日频因子
    build_daily_sentiment_factor(combined)


# ── 聚合日频情绪因子 ──────────────────────────────────────────────────────
def build_daily_sentiment_factor(df: pd.DataFrame = None):
    if df is None:
        if not os.path.exists(DETAIL_PATH):
            print("  情感详情文件不存在")
            return
        df = pd.read_csv(DETAIL_PATH)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    def weighted_score(g):
        total_conf = g["confidence"].sum()
        if total_conf == 0:
            return 0.0
        return (g["score"] * g["confidence"]).sum() / total_conf

    daily = df.groupby("date").apply(
        lambda g: pd.Series({
            "sentiment_score": weighted_score(g),
            "news_count": len(g),
            "bullish_count": (g["sentiment"] == "bullish").sum(),
            "bearish_count": (g["sentiment"] == "bearish").sum(),
            "neutral_count": (g["sentiment"] == "neutral").sum(),
            "avg_confidence": g["confidence"].mean(),
            "geopolitics_flag": int((g["event_type"] == "geopolitics").any()),
            "policy_flag": int((g["event_type"] == "policy").any()),
            "supply_flag": int((g["event_type"] == "supply").any()),
            "top_bullish": g[g["sentiment"] == "bullish"].sort_values(
                "confidence", ascending=False
            )["title"].iloc[0] if (g["sentiment"] == "bullish").any() else "",
            "top_bearish": g[g["sentiment"] == "bearish"].sort_values(
                "confidence", ascending=False
            )["title"].iloc[0] if (g["sentiment"] == "bearish").any() else "",
        }),
        include_groups=False
    ).reset_index()

    daily.to_csv(FACTOR_PATH, index=False, encoding="utf-8-sig")
    print(f"  日频情绪因子已更新，共 {len(daily)} 天")
    return daily


# ── 获取最新情感摘要（供app.py调用）─────────────────────────────────────
def get_latest_sentiment_summary(days=7):
    """返回最近N天的情感摘要，供前端展示"""
    if not os.path.exists(DETAIL_PATH):
        return []

    df = pd.read_csv(DETAIL_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    cutoff = datetime.today() - timedelta(days=days)
    recent = df[df["date"] >= cutoff].sort_values("date", ascending=False)

    summary = []
    for _, row in recent.head(20).iterrows():
        summary.append({
            "date"           : str(row["date"])[:10],
            "title"          : row["title"],
            "source"         : row.get("source", ""),
            "url"            : row.get("url", ""),
            "sentiment"      : row.get("sentiment", "neutral"),
            "score"          : float(row.get("score", 0)),
            "confidence"     : float(row.get("confidence", 0)),
            "event_type"     : row.get("event_type", "other"),
            "impact_duration": row.get("impact_duration", "short"),
            "reason"         : row.get("reason", ""),
        })
    return summary


if __name__ == "__main__":
    incremental_sentiment_analysis(max_articles=50)
