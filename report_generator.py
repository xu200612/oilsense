import os
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))
LEGACY_ENV_PATH = os.path.join(
    os.path.dirname(ROOT_DIR),
    "OilSense 原油风险智能预警系统",
    "技术文档",
    "OilSense_源代码",
    ".env",
)
if os.path.exists(LEGACY_ENV_PATH):
    load_dotenv(dotenv_path=LEGACY_ENV_PATH)


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
            # 用拼接替代format，避免标题里的{}被误解析
            title = str(n.get("title", "")).replace("{", "{{").replace("}", "}}")
            news_text += "- " + str(n.get("date", ""))[:10] + ": " + title + "\n"

    return (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "【当前市场状态】正常市场（统计模型有效）\n"
        "报告日期：" + datetime.now().strftime("%Y年%m月%d日") + "\n"
        "WTI现价：" + str(round(current_price, 2)) + "美元/桶\n\n"
        "【模型预测（未来10日涨跌幅）】\n" + pred_text + "\n\n"
        "【主要驱动因子（XGBoost Top5）】\n" + factors_text + "\n\n"
        "【宏观指标】\n" + macro_context + "\n\n"
        "【关键咽喉点航运状态】\n" + cp_context + "\n"
        + news_text +
        "━━━━━━━━━━━━━━━━━━━━━━━"
    )


def _build_blackswan_context(signals: dict, recent_news: list = None) -> str:
    """黑天鹅期间的上下文构建"""
    from black_swan import _build_context
    return _build_context(signals, recent_news)


def _get_normal_prompt(context: str) -> str:
    return context + """

你是 OilSense 的企业银行风险分析助手，正在为“花旗杯”企业银行风控场景生成一份油价风险简报。请把报告写成参赛项目可展示的专业分析，而不是交易员喊单；不要声称自己是花旗员工，也不要代表花旗出具意见。

【可使用的数据来源】
- 价格与市场：WTI/Brent、期限价差、历史收益率、波动率。
- 宏观金融：美元指数、美国CPI/PPI、联邦基金利率、10年期美债、VIX、EPU等。
- 新闻与情绪：NewsAPI/RSS新闻、DeepSeek情绪打分、GDELT地缘政治冲突强度、Goldstein、Tone、mentions。
- 供应链与航运：PortWatch咽喉点油轮通行、AIS快照、霍尔木兹/曼德/苏伊士等关键通道状态。
- 模型输出：XGBoost P10/P50/P90未来10日风险区间、Top特征贡献、历史情景匹配结果。

【写作目标】
报告要回答评委最关心的三个问题：油价为什么这样动、企业银行客户会受到什么影响、银行可以怎样管理风险敞口。请用中文，800-1000字以内，专业但不要堆术语。

【结构要求】
1. 市场判断：用1-2句话先给出当前油价风险状态和未来10日核心判断，必须引用上下文中的WTI现价和P10/P50/P90区间。
2. 驱动因素：围绕模型Top5因子解释传导链。每个因子要写清楚“数据变化 → 经济含义 → 供需/风险溢价/金融条件 → 油价影响”，不要只说“影响油价”。
3. 新闻/情绪增量：说明新闻情绪、GDELT或航运数据补充了什么传统宏观因子看不到的信息；如果上下文没有新闻或情绪数据，明确写“当前数据不足”。
4. 企业银行敞口：分别讨论航空/航运、化工炼化、能源生产商、能源贸易商、大宗商品进口商等客户的信用风险、现金流压力、授信/保证金/贸易融资需求。
5. 风险管理建议：给出2-3条可执行建议，注明工具或动作（套保、授信监控、保证金、情景压力测试、库存/采购节奏）、适用客户、触发条件。
6. 限制说明：最后简短说明模型输出是XGBoost分位数风险区间，方向命中率、MAE/RMSE、区间覆盖率以页面回测模块为准；不得自行编造模型成绩。

【硬性约束】
只使用上下文提供的数据。不要编造最新新闻、API来源、价格、概率或模型准确率。若数据缺失，直接说明“当前数据不足”。末尾加一句“本报告由OilSense AI辅助生成，仅供风险管理参考，不构成投资建议。”"""


def _get_blackswan_prompt(context: str) -> str:
    return context + """

【霍尔木兹封锁分析必读背景知识——请在报告中体现这些事实】
1. 霍尔木兹海峡每日通过约1700-2100万桶原油，占全球海运贸易约20%，但占全球总供应约17%（因部分供应走陆路）。
2. 沙特阿拉伯有东西管道（Petroline）替代出口路线，连接波斯湾到红海延布港，日输送能力约500万桶，封锁时可绕开霍尔木兹。
3. 阿联酋有阿布扎比到富查伊拉港的管道，日产能约150万桶，可绕开霍尔木兹直接进入阿曼湾。
4. 伊朗、伊拉克、科威特没有替代出口路线，完全依赖霍尔木兹。
5. 因此，霍尔木兹封锁的实际供应中断规模不是通过量的全部，要扣除沙特约500万桶/日和阿联酋约150万桶/日的替代出口能力。
6. 好望角绕行会增加约15-20天航程，运费成本约增加5-10美元/桶。
7. IEA成员国战略储备约16亿桶，协调释放峰值约200-300万桶/日，但这是短期工具，无法替代持续供应。

你是 OilSense 的企业银行风险分析助手，正在为“花旗杯”企业银行风险管理场景生成极端事件情景报告。请把报告写成参赛项目可展示的“地缘政治冲击—油价—企业敞口—银行动作”闭环分析；不要声称自己是花旗员工，也不要代表花旗出具意见。

【核心原则】
这不是简单预测油价涨跌，而是解释极端事件如何通过供应缺口、航运绕行、风险溢价、库存与金融条件传导到企业客户。分析霍尔木兹时必须区分“通过量受扰”和“净供应中断”，扣除沙特、阿联酋替代管道、库存释放、绕行和需求萎缩，不得把20%海运贸易直接写成20%全球供应永久中断。

【必须使用的数据线索】
- 黑天鹅触发信号：封锁天数、油轮通行降幅、咽喉点状态、VIX、波动率、GDELT/新闻情绪。
- 模型输出：当前WTI、P10/P50/P90未来10日风险区间、情景匹配事件、相似度、历史10日/30日涨跌。
- 银行敞口：航空/航运、化工炼化、能源贸易商、能源生产商、大宗商品进口商的信用风险、保证金压力和贸易融资需求。

【报告结构】
1. 事件定性：用一段话说明当前是轻微/中等/严重/极端哪一级，判断依据必须来自上下文信号。
2. 传导链：触发事件 → 实际净供应缺口/航运绕行 → 市场风险溢价和库存反应 → WTI价格区间。不要只列数字，要解释机制。
3. 三情景路径：快速缓解、僵持延续、全面升级。每个情景都要写触发条件、价格区间或涨跌方向、对企业现金流的影响。
4. 历史类比：基于上下文给出的相似事件，说明“相似在哪里、不相似在哪里”，不要机械套用历史涨幅。
5. 企业银行动作：列出银行应重点监控的客户、授信调整、保证金/信用证、套保、压力测试和每日预警指标。
6. 限制说明：说明这是XGBoost风险区间+历史情景匹配+AI分析的辅助判断，模型表现以页面回测为准。

【硬性约束】
只使用上下文提供的数据。不要编造最新新闻、价格、概率、API来源或模型准确率；缺失则写“当前数据不足”。全文1000-1200字以内，结尾加“本报告由OilSense AI辅助生成，仅供风险管理参考，不构成投资建议。”"""


def generate_report(
    current_price     : float,
    pred_low          : float,
    pred_mid          : float,
    pred_high         : float,
    feature_importance: pd.DataFrame,
    is_black_swan     : bool = False,
    bs_signals        : dict = None,
    recent_news       : list = None,
    similar_events    : list = None,
) -> dict:
    # 强制重新加载 .env，避免Streamlit缓存旧Key
    load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"), override=True)
    report_run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    refresh_instruction = (
        "\n\n【本次生成控制】\n"
        "报告生成时间：" + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n"
        "报告编号：" + report_run_id + "\n"
        "请基于同一批数据重新组织分析表达和段落侧重点。核心判断必须忠于数据，"
        "但不要复用上一版报告的标题、开头句、段落顺序和整段措辞。\n"
    )

    try:
        from openai import OpenAI
    except ImportError:
        return {"error": "openai 库未安装", "status": "error"}

    api_key = os.getenv("CLAUDE_API_KEY", "") or os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        return {"error": "DEEPSEEK_API_KEY 未配置", "status": "error"}

    client = OpenAI(api_key=api_key, base_url="https://api.whatai.cc/v1")

    # 构建历史情景匹配附加上下文
    similar_events_text = ""
    if similar_events:
        similar_events_text = "\n\n【AI历史情景匹配结果（余弦相似度）】\n"
        for ev in similar_events[:3]:
            ret_10d = "{:+.1%}".format(ev["actual_return"]) if ev.get("actual_return") is not None else "N/A"
            ret_30d = "{:+.1%}".format(ev["return_30d"]) if ev.get("return_30d") is not None else "N/A"
            # 用字符串拼接替代.format()，避免description里的{}被误解析
            similar_events_text += (
                "- " + str(ev.get("event", "")) +
                "（相似度" + str(round(ev.get("similarity", 0) * 100)) + "%，严重程度：" +
                str(ev.get("severity", "")) + "）\n" +
                "  触发日：" + str(ev.get("trigger_date", "")) +
                "  10日实际涨跌：" + ret_10d +
                "  30日实际涨跌：" + ret_30d + "\n" +
                "  描述：" + str(ev.get("description", "")) + "\n"
            )

    if is_black_swan and bs_signals:
        context   = _build_blackswan_context(bs_signals, recent_news)
        context  += similar_events_text
        context  += refresh_instruction
        prompt    = _get_blackswan_prompt(context)
        mode      = "黑天鹅极端事件分析"
        max_tokens= 5000
        sys_msg   = (
            "你是OilSense企业银行风险分析助手，模拟企业银行大宗商品风险分析框架，专注地缘政治风险对大宗商品价格的影响。"
            "报告的核心是解释油价波动的驱动逻辑和因果传导链，而非单纯预测价格涨跌。"
            "每个因子必须说清楚完整的传导路径。银行风险敞口评估需要从企业银行视角出发，"
            "关注客户信用风险变化和融资需求，而非只给交易建议。"
            "格式严格按照提示词的表格和章节结构输出，不得省略任何章节。"
            "注意：不要在报告中添加任何版权声明（如© Citigroup）、内部密级标注（如'企业银行内部参阅'）"
            "或'本报告不得对外传播'等字样。不要声称自己是花旗员工或代表花旗出具意见。"
            "只使用上下文给出的数据；缺失的数据写明'当前数据不足'，不要编造最新价格、日期、API来源或模型表现。"
            "免责声明只需说明这是OilSense AI辅助生成的分析报告，仅供参考，不构成投资建议。"
        )
    else:
        context   = _build_normal_context(
            current_price, pred_low, pred_mid, pred_high,
            feature_importance, recent_news
        )
        context  += similar_events_text   # 正常报告也注入情景匹配数据
        context  += refresh_instruction
        prompt    = _get_normal_prompt(context)
        mode      = "标准市场风险报告"
        max_tokens= 3000
        sys_msg   = (
            "你是OilSense企业银行风险分析助手，模拟企业银行大宗商品风险分析框架，为航空、化工、能源贸易商等企业客户提供油价风险报告。"
            "报告的核心是解释油价波动的驱动逻辑和因果传导链，而非单纯预测价格涨跌。"
            "每个驱动因子必须说清楚完整的传导路径（A→B→C→对油价的影响）。"
            "银行风险敞口评估要从企业银行视角出发，关注客户信用风险和融资需求变化。"
            "注意：不要在报告中添加任何版权声明、内部密级标注或传播限制字样。"
            "不要声称自己是花旗员工或代表花旗出具意见。"
            "只使用上下文给出的数据；缺失的数据写明'当前数据不足'，不要编造最新价格、日期、API来源或模型表现。"
            "免责声明只需说明这是OilSense AI系统生成的分析报告，仅供参考，不构成投资建议。"
        )

    try:
        response = client.chat.completions.create(
            model="claude-sonnet-4-6",
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.35,
        )
        report_text = response.choices[0].message.content

        result = {
            "report"       : report_text,
            "mode"         : mode,
            "current_price": current_price,
            "is_black_swan": is_black_swan,
            "generated_at" : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model"        : "Claude Sonnet 4.6",
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
