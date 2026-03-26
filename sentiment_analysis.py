import os
import json
import time
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm

# ── 加载环境变量 ───────────────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

client = OpenAI(
    api_key  = DEEPSEEK_API_KEY,
    base_url = "https://api.deepseek.com"
)

# ── 情感分析函数 ───────────────────────────────────────────────────────────
def analyze_sentiment(title: str, description: str) -> dict:
    """
    输入一条新闻的标题和摘要，返回结构化的情感评分
    """
    text = f"Title: {title}\nDescription: {description or 'N/A'}"

    prompt = f"""You are a professional energy market analyst.
Analyze the following news article and assess its impact on crude oil prices.

Return a JSON object with exactly these fields:
- "sentiment": one of "bullish", "bearish", or "neutral"
- "score": a float from -1.0 (strongly bearish) to +1.0 (strongly bullish), 0.0 is neutral
- "confidence": a float from 0.0 to 1.0 indicating your confidence
- "event_type": one of "supply", "demand", "geopolitics", "policy", "macro", "other"
- "reason": one sentence explanation in English

News:
{text}

Respond with valid JSON only, no extra text."""

    try:
        response = client.chat.completions.create(
            model    = "deepseek-chat",
            messages = [{"role": "user", "content": prompt}],
            temperature = 0.1  # 低温度保证输出稳定
        )
        raw = response.choices[0].message.content.strip()

        # 清理可能的 markdown 代码块
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        result = json.loads(raw)
        return result

    except Exception as e:
        print(f"  分析失败: {e}")
        return {
            "sentiment"  : "neutral",
            "score"      : 0.0,
            "confidence" : 0.0,
            "event_type" : "other",
            "reason"     : "analysis failed"
        }

# ── 批量处理新闻 ───────────────────────────────────────────────────────────
def process_news_sentiment():
    print("正在加载新闻数据...")
    news_path = os.path.join(ROOT_DIR, "data", "raw", "news_data.csv")
    df = pd.read_csv(news_path)
    print(f"  共 {len(df)} 条新闻待分析")

    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="情感分析进度"):
        result = analyze_sentiment(
            title       = str(row["title"]),
            description = str(row.get("description", ""))
        )
        result["date"]    = row["date"]
        result["title"]   = row["title"]
        result["keyword"] = row["keyword"]
        results.append(result)

        # 每条请求间隔0.5秒，避免触发频率限制
        time.sleep(0.5)

    # 保存详细结果
    df_results = pd.DataFrame(results)
    detail_path = os.path.join(ROOT_DIR, "data", "processed", "news_sentiment_detail.csv")
    df_results.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"\n详细情感数据已保存：{detail_path}")

    return df_results

# ── 聚合为日频情绪因子 ─────────────────────────────────────────────────────
def build_daily_sentiment_factor(df_results: pd.DataFrame):
    print("\n正在聚合日频情绪因子...")

    df_results["date"] = pd.to_datetime(df_results["date"])

    # 用置信度加权平均情感得分
    def weighted_score(group):
        if group["confidence"].sum() == 0:
            return 0.0
        return (group["score"] * group["confidence"]).sum() / group["confidence"].sum()

    daily = df_results.groupby("date").apply(
        lambda g: pd.Series({
            "sentiment_score"     : weighted_score(g),
            "news_count"          : len(g),
            "bullish_count"       : (g["sentiment"] == "bullish").sum(),
            "bearish_count"       : (g["sentiment"] == "bearish").sum(),
            "avg_confidence"      : g["confidence"].mean(),
            "geopolitics_flag"    : int((g["event_type"] == "geopolitics").any()),
            "policy_flag"         : int((g["event_type"] == "policy").any()),
        })
    ).reset_index()

    # 保存日频因子
    factor_path = os.path.join(ROOT_DIR, "data", "processed", "daily_sentiment.csv")
    daily.to_csv(factor_path, index=False, encoding="utf-8-sig")
    print(f"日频情绪因子已保存：{factor_path}")
    print(f"\n情绪因子预览：")
    print(daily.tail(10).to_string(index=False))

    return daily

# ── 主程序 ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df_results = process_news_sentiment()
    daily      = build_daily_sentiment_factor(df_results)
    print("\nStep 3 完成！情感因子构建成功。")
