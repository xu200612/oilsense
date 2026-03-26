import os
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))


def _build_normal_context(
    current_price: float,
    pred_low: float,
    pred_mid: float,
    pred_high: float,
    feature_importance: pd.DataFrame,
    recent_news: list = None,
) -> str:
    """正常市场下的上下文构建"""

    # 预测价格区间
    price_low  = round(current_price * (1 + pred_low),  2)
    price_mid  = round(current_price * (1 + pred_mid),  2)
    price_high = round(current_price * (1 + pred_high), 2)

    pred_text = (
        "悲观情景（10%分位）：{:.1f}%  → 约{:.2f}美元/桶\n"
        "基准预测（中位数）  ：{:.1f}%  → 约{:.2f}美元/桶\n"
        "乐观情景（90%分位）：{:.1f}%  → 约{:.2f}美元/桶"
    ).format(
        pred_low  * 100, price_low,
        pred_mid  * 100, price_mid,
        pred_high * 100, price_high,
    )

    # Top5驱动因子
    top5 = feature_importance.head(5)
    factors_text = "\n".join([
        "  {}. {}（重要性：{:.4f}）".format(i+1, row["feature"], row["importance"])
        for i, (_, row) in enumerate(top5.iterrows())
    ])

    # 宏观数据
    macro_path = os.path.join(ROOT_DIR, "data", "raw", "macro_data.csv")
    macro_context = ""
    if os.path.exists(macro_path):
        df_m  = pd.read_csv(macro_path, index_col=0, parse_dates=True)
        parts = []
        for label, col in [
            ("VIX恐慌指数", "VIX"), ("美元指数DXY", "DXY"),
            ("美联储利率",  "FED_RATE"), ("10年期美债", "US10Y"),
            ("全球EPU",    "GLOBAL_EPU"),
        ]:
            if col in df_m.columns:
                val  = float(df_m[col].dropna().iloc[-1])
                prev = float(df_m[col].dropna().iloc[-2])
                chg  = val - prev
                parts.append("{}：{:.2f}（{:+.3f}）".format(label, val, chg))
        macro_context = "\n".join(parts)

    # 咽喉点状态
    pw_path = os.path.join(ROOT_DIR, "data", "raw", "portwatch_chokepoints.csv")
    cp_context = ""
    if os.path.exists(pw_path):
        df_pw = pd.read_csv(pw_path, index_col=0, parse_dates=True)
        cp_map = {
            "cp6": "霍尔木兹", "cp4": "曼德海峡",
            "cp1": "苏伊士",   "cp5": "马六甲",
        }
        parts = []
        for cp, name in cp_map.items():
            col = cp + "_tanker"
            if col in df_pw.columns:
                s      = df_pw[col].dropna()
                latest = int(s.iloc[-1])
                avg    = float(s.tail(90).mean())
                ratio  = latest / avg if avg > 0 else 1.0
                parts.append(
                    "{}：{}艘/日（均值{:.1f}，比率{:.0f}%）".format(
                        name, latest, avg, ratio * 100)
                )
        cp_context = "\n".join(parts)

    # 新闻
    news_text = ""
    if recent_news:
        news_text = "\n近期关键市场新闻（最新6条）：\n"
        for n in recent_news[:6]:
            news_text += "- {}: {}\n".format(
                str(n.get("date", ""))[:10], n.get("title", ""))

    return """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【当前市场状态】正常市场（统计模型有效）
报告日期：{}
WTI现价：{:.2f}美元/桶

【模型预测（未来10日涨跌幅）】
{}

【主要驱动因子（XGBoost Top5）】
{}

【宏观指标】
{}

【关键咽喉点航运状态】
{}
{}━━━━━━━━━━━━━━━━━━━━━━━""".format(
        datetime.now().strftime("%Y年%m月%d日"),
        current_price,
        pred_text,
        factors_text,
        macro_context,
        cp_context,
        news_text,
    )


def _build_blackswan_context(signals: dict, recent_news: list = None) -> str:
    """黑天鹅期间的上下文构建"""
    from black_swan import _build_context
    return _build_context(signals, recent_news)


def _get_normal_prompt(context: str) -> str:
    return context + """

请作为花旗银行企业银行部大宗商品风险分析师，生成一份标准版WTI原油风险分析报告
（面向企业客户：航空公司、化工企业、能源贸易商，中文，专业金融语言，500字以内）：

## 一、市场概况
（2-3句，描述当前油价水平、近期走势和整体市场环境）

## 二、核心驱动因子分析
（针对Top5因子，逐一简析当前对油价的影响方向，每条1-2句）

## 三、未来10日价格展望
（基于模型预测区间，描述三种情景下的价格路径，给出具体价格数字）

## 四、行业风险提示
- 航空业：燃油成本变化及建议
- 化工/炼化：原料成本及库存策略
- 能源贸易商：市场机会及风险点

## 五、风险管理建议
（2-3条具体可执行建议，注明工具/方向/期限）

## 六、风险提示
（1句免责声明）"""


def _get_blackswan_prompt(context: str) -> str:
    return context + """

请作为花旗银行大宗商品部首席分析师，提供极端事件深度分析报告
（中文，专业金融语言，700字以内）：

**1. 事件定性与严重程度**
（结合封锁天数、油轮降幅、历史类比，评定严重程度：轻微/中等/严重/极端）

**2. 供需缺口分析**
（供应缺口规模、替代路线能力、战略储备释放、需求侧响应四个维度量化）

**3. 分情景价格路径预测（未来30日）​**
| 情景 | 触发条件 | WTI价格区间 | 发生概率 |
|------|---------|------------|---------|
| 快速缓解 | ... | $XX~$XX | XX% |
| 僵持延续 | ... | $XX~$XX | XX% |
| 全面升级 | ... | $XX~$XX | XX% |

**4. 行业冲击信号**
- 航空业、化工/炼化、能源贸易商、航运业（各1-2句含建议）

**5. 企业风险对冲操作建议**
（3条具体可执行，注明工具/方向/比例/期限）

**6. 每日关键监测指标**
（3-4个指标，注明阈值和含义）

**免责声明**（1句）"""


def generate_report(
    current_price     : float,
    pred_low          : float,
    pred_mid          : float,
    pred_high         : float,
    feature_importance: pd.DataFrame,
    is_black_swan     : bool = False,
    bs_signals        : dict = None,
    recent_news       : list = None,
) -> dict:
    """
    统一报告生成入口
    正常市场 → 标准版风险报告
    黑天鹅期间 → 极端事件深度分析报告
    """
    try:
        from openai import OpenAI
    except ImportError:
        return {"error": "openai 库未安装", "status": "error"}

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        return {"error": "DEEPSEEK_API_KEY 未配置", "status": "error"}

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    if is_black_swan and bs_signals:
        context   = _build_blackswan_context(bs_signals, recent_news)
        prompt    = _get_blackswan_prompt(context)
        mode      = "黑天鹅极端事件分析"
        max_tokens= 1500
        sys_msg   = (
            "你是花旗银行大宗商品部首席分析师，"
            "专注地缘政治风险对大宗商品价格的影响。"
            "分析需有具体数字支撑，给出可执行建议。"
        )
    else:
        context   = _build_normal_context(
            current_price, pred_low, pred_mid, pred_high,
            feature_importance, recent_news
        )
        prompt    = _get_normal_prompt(context)
        mode      = "标准市场风险报告"
        max_tokens= 1000
        sys_msg   = (
            "你是花旗银行企业银行部大宗商品风险分析师，"
            "为企业客户提供专业的油价风险分析报告。"
            "语言简洁专业，数字具体，建议可执行。"
        )

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        report_text = response.choices[0].message.content

        result = {
            "report"       : report_text,
            "mode"         : mode,
            "current_price": current_price,
            "is_black_swan": is_black_swan,
            "generated_at" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model"        : "DeepSeek-Chat",
            "status"       : "ok",
        }

        # 保存最新报告
        out_path = os.path.join(ROOT_DIR, "data", "raw", "latest_report.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result

    except Exception as e:
        return {"error": str(e), "status": "error"}


if __name__ == "__main__":
    importance = pd.read_csv(
        os.path.join(ROOT_DIR, "data", "processed", "feature_importance.csv")
    )

    # 测试正常市场报告
    print("=" * 50)
    print("测试：正常市场报告")
    print("=" * 50)
    result = generate_report(
        current_price      = 89.0,
        pred_low           = -0.05,
        pred_mid           =  0.02,
        pred_high          =  0.08,
        feature_importance = importance,
        is_black_swan      = False,
    )
    if result["status"] == "ok":
        print(result["report"])
    else:
        print("错误：", result["error"])
