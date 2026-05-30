"""一次性脚本：对 MediaCloud 历史新闻跑情感分析，keep_days=0 保留全部历史。"""
import os, sys

SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "OilSense 原油风险智能预警系统",
    "技术文档",
    "OilSense_源代码",
)
MEDIACLOUD = os.path.join(SRC, "data", "raw", "mediacloud_news_history_batch.csv")
ROOT_HERE  = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, SRC)
os.chdir(SRC)   # ROOT_DIR in sentiment_analysis will resolve to src

from sentiment_analysis import incremental_sentiment_analysis

detail = os.path.join(ROOT_HERE, "data", "processed", "news_sentiment_detail.csv")
factor = os.path.join(ROOT_HERE, "data", "processed", "daily_sentiment.csv")

print(f"新闻源: {MEDIACLOUD}")
print(f"输出detail: {detail}")
print(f"输出factor: {factor}")
print()

incremental_sentiment_analysis(
    max_articles=2500,
    news_path=MEDIACLOUD,
    detail_path=detail,
    factor_path=factor,
    keep_days=0,
)
