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

你是花旗银行企业银行部大宗商品风险分析师，请根据以上数据，用你自己的专业判断写一份油价风险分析报告。
面向花旗企业银行团队及企业客户，中文，专业金融语言，800字以内。

【核心要求】这份报告的价值不是预测油价涨跌，而是解释油价为什么会这样动，以及这对花旗企业银行的客户意味着什么。
评委是花旗企业银行专业人士，他们要看到结构化的市场逻辑，而不是价格数字。

请围绕三个核心问题展开，用你认为最合适的方式组织报告结构，不必强行套模板：

问题一：当前油价由什么在驱动？
基于模型Top5因子，每个因子说清楚完整的传导链（例："美联储利率↑ → 美元走强 → 以美元计价原油相对昂贵 → 需求承压 → 油价下行压力"）。
不要只说这个因子"影响油价"，要说清楚中间每一步的逻辑。
特别说明情感因子这次捕捉到了什么传统量化数据捕捉不到的市场信号。

问题二：当前市场的主要矛盾是什么？
供给、需求、金融条件三者现在哪个在主导，哪个在对冲，油价处于什么均衡状态，这个均衡稳不稳定。

问题三：这对花旗的企业客户意味着什么？
不同行业传导路径和时滞不同——航空、化工、能源贸易商、物流各自受到油价的影响机制是什么，
哪些行业客户的信用风险在当前价格水平下值得银行提高关注，哪些客户的融资需求可能激增。

风险量化区间（附在末尾作为辅助参考，非报告主体）：
根据上方上下文中的模型预测数据，列出P10/P50/P90对应的价格区间。
注明这是XGBoost分位数模型输出，样本外方向准确率约58%，价格区间仅供参考，核心价值在于以上逻辑分析。

最后给出2-3条可执行对冲建议，注明工具、方向、期限、止损条件。
加一行免责声明结尾。"""


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

你是花旗银行大宗商品部首席分析师，请根据以上数据和背景知识，写一份极端事件深度风险分析报告。
面向花旗企业银行团队及高风险敞口客户，中文，专业金融语言，1000字以内。

【核心要求】报告要体现真实的市场逻辑，不要照搬数字，要有自己的判断。
特别注意：分析供应中断规模时，必须考虑沙特和阿联酋的替代出口管道，不能把霍尔木兹通过量全部算作中断。

请围绕以下问题展开，用你认为最合适的方式组织报告，不必强行套模板：

**一、这次事件的运行逻辑是什么？**
完整的因果链：触发原因 → 供应中断的实际规模（考虑替代路线后的净缺口）→ 市场传导机制 → 价格响应
给出风险评级（轻微/中等/严重/极端）并量化支撑理由。

**二、供需缺口有多大，市场如何再平衡？**
实际净供应缺口（扣除替代路线后）是多少，好望角绕行能弥补多少，IEA储备释放能覆盖多少，
需求侧在高价下会有多少自然萎缩，市场再平衡的时间窗口大概是多长。

**三、历史情景给我们什么启示？**
基于AI匹配的历史事件，重点说逻辑上的类比而非价格上的类比——
当时供需失衡的机制与现在有什么相似和不同，市场是如何找到新均衡的，
对当前最有参考价值的经验是什么。

**四、分情景价格路径**
给出快速缓解、僵持延续、全面升级三种情景的价格区间和核心假设，
每个情景说清楚触发条件和市场再平衡逻辑，不要只给数字。

**五、对花旗企业银行客户的影响**
不同行业的传导路径和时滞各不同，哪些客户信用风险上升，哪些融资需求激增，
银行应该主动关注什么、提供什么服务。

**六、可执行对冲建议**
3条，注明工具、方向、比例、期限、止损/平仓触发条件。

**七、每日监测指标**
3-4个最关键的指标，注明当前值、预警阈值和阈值的含义。

末尾加免责声明。"""


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
        prompt    = _get_blackswan_prompt(context)
        mode      = "黑天鹅极端事件分析"
        max_tokens= 5000
        sys_msg   = (
            "你是花旗银行大宗商品部首席分析师，专注地缘政治风险对大宗商品价格的影响。"
            "报告的核心是解释油价波动的驱动逻辑和因果传导链，而非单纯预测价格涨跌。"
            "每个因子必须说清楚完整的传导路径。银行风险敞口评估需要从企业银行视角出发，"
            "关注客户信用风险变化和融资需求，而非只给交易建议。"
            "格式严格按照提示词的表格和章节结构输出，不得省略任何章节。"
            "注意：不要在报告中添加任何版权声明（如© Citigroup）、内部密级标注（如'企业银行内部参阅'）"
            "或'本报告不得对外传播'等字样。免责声明只需说明这是AI辅助生成的分析报告仅供参考即可。，你也不是花旗银行的分析师，你开头只要写报告就行，不要加戏"
        )
    else:
        context   = _build_normal_context(
            current_price, pred_low, pred_mid, pred_high,
            feature_importance, recent_news
        )
        context  += similar_events_text   # 正常报告也注入情景匹配数据
        prompt    = _get_normal_prompt(context)
        mode      = "标准市场风险报告"
        max_tokens= 3000
        sys_msg   = (
            "你是花旗银行企业银行部大宗商品风险分析师，为航空、化工、能源贸易商等企业客户提供油价风险报告。"
            "报告的核心是解释油价波动的驱动逻辑和因果传导链，而非单纯预测价格涨跌。"
            "每个驱动因子必须说清楚完整的传导路径（A→B→C→对油价的影响）。"
            "银行风险敞口评估要从企业银行视角出发，关注客户信用风险和融资需求变化。"
            "注意：不要在报告中添加任何版权声明、内部密级标注或传播限制字样。"
            "免责声明只需说明这是OilSense AI系统生成的分析报告，仅供参考，不构成投资建议。你也不是花旗银行的分析师，你开头只要写报告就行，不要加戏"
        )

    try:
        response = client.chat.completions.create(
            model="claude-sonnet-4-6",
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