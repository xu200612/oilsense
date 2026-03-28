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

请作为花旗银行企业银行部大宗商品风险分析师，根据以上数据生成一份WTI原油风险分析报告。
报告面向企业银行团队及其客户（航空、化工、能源贸易商、物流、高耗能制造），中文，专业金融语言，800字以内。

---

## 一、油价驱动因子分析
基于模型特征重要性，逐一解读当前Top5驱动因子：
- **因子名称**（重要性得分）：当前数值 → 对油价的影响方向与逻辑（1-2句）
- 最后注明：LLM情感因子 vs 传统量化因子的相对贡献，说明AI增强的价值

## 二、中短期油价预测（未来10日）
| 情景 | 涨跌幅区间 | WTI价格区间 | 发生概率 | 核心假设 |
|------|-----------|------------|---------|---------|
| 悲观（P10） | XX% | $XX~$XX | ~10% | ... |
| 基准（P50） | XX% | $XX | ~80% | ... |
| 乐观（P90） | XX% | $XX~$XX | ~10% | ... |
说明：价格区间来自XGBoost量化分位数模型，已在样本外测试集完成回测验证（方向准确率约58%）。

## 三、风险信号等级
当前综合风险等级：【高/中/低/极低】
信号依据：（2句，说明触发该等级的核心指标及阈值）
模型置信度：（高/中/低，结合当前波动率和情感信号强度说明）

## 四、行业冲击信号与企业应对建议
- **航空业**：燃油成本变动对每座公里成本的影响估算，套保工具建议（期货锁价 or 期权保护）
- **化工/炼化**：原料成本传导压力，库存补货策略（超前 or 推迟）
- **能源贸易商**：价差机会或风险，头寸方向建议
- **物流/高耗能制造**：燃料成本占比预警，融资安排建议

## 五、可执行风险管理操作建议
（3条，每条注明：对冲工具 | 操作方向 | 建议比例 | 期限 | 止损触发条件）

## 六、风险提示
本报告由OilSense AI系统自动生成，基于公开市场数据与机器学习模型，仅供参考，不构成投资建议。"""


def _get_blackswan_prompt(context: str) -> str:
    return context + """

请作为花旗银行大宗商品部首席分析师，根据以上极端事件数据，生成一份深度风险分析报告。
面向企业银行团队及高风险敞口客户，中文，专业金融语言，1000字以内。

---

**1. 极端事件定性与风险评级**
结合封锁持续天数、油轮通过量降幅（对比历史均值）、历史类比案例，
给出风险评级：轻微 / 中等 / 严重 / 极端，并量化说明理由（需引用具体数据）。

**2. 供需缺口量化分析**
| 维度 | 当前状态 | 1-2周预判 | 1个月预判 |
|------|---------|----------|---------|
| 供应中断规模（万桶/日） | ... | ... | ... |
| 替代路线能力（好望角绕行） | ... | ... | ... |
| 战略石油储备可释放量 | ... | ... | ... |
| 需求侧弹性响应 | ... | ... | ... |

**3. AI历史情景匹配结论**
（基于余弦相似度匹配的最相似历史事件，说明：相似度、当时价格路径、与当前情况的关键差异、对本次预判的参考价值）

**4. 分情景油价路径预测（未来30日）**
| 情景 | 触发条件 | WTI价格区间 | 概率 | 历史参照 |
|------|---------|------------|-----|---------|
| 快速缓解 | ... | $XX~$XX | XX% | ... |
| 僵持延续 | ... | $XX~$XX | XX% | ... |
| 全面升级 | ... | $XX~$XX | XX% | ... |

**5. 行业冲击信号与紧急应对**
- **航空业**：燃油成本冲击幅度（$/座位）、建议立即执行的套保操作
- **化工/炼化**：原料成本上涨传导时间、安全库存天数建议
- **能源贸易商**：价差扩大机会、绕行运费成本增量测算
- **物流/高耗能制造**：燃料成本敏感性分析、融资需求预警

**6. 企业银行客户风险对冲紧急操作建议**
（3条，格式：对冲工具 | 方向 | 建议比例 | 期限 | 止损/平仓触发条件）

**7. 每日关键监测指标**
（4-5个指标，格式：指标名 | 数据来源 | 当前值 | 预警阈值 | 超阈值含义）

**8. 模型信号说明**
（说明：当前统计模型已切换至极端情景匹配模式的依据，历史情景相似度，以及该模式的局限性）

**免责声明**：本报告由OilSense AI系统在极端事件期间自动生成，不确定性显著高于正常市场，仅供参考，不构成任何投融资建议。"""


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
        max_tokens= 2500
        sys_msg   = (
            "你是花旗银行大宗商品部首席分析师，专注地缘政治风险对大宗商品价格的影响。"
            "报告需包含具体数字支撑，量化分析供需缺口，给出可执行的企业银行客户对冲建议。"
            "格式严格按照提示词的表格和章节结构输出，不得省略任何章节。"
        )
    else:
        context   = _build_normal_context(
            current_price, pred_low, pred_mid, pred_high,
            feature_importance, recent_news
        )
        prompt    = _get_normal_prompt(context)
        mode      = "标准市场风险报告"
        max_tokens= 1800
        sys_msg   = (
            "你是花旗银行企业银行部大宗商品风险分析师，为航空、化工、能源贸易商等企业客户提供油价风险报告。"
            "报告需基于模型数据输出具体价格数字，因子分析需量化，行业建议需可执行。"
            "格式严格按照提示词的表格和章节结构输出，不得省略任何章节。"
        )
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