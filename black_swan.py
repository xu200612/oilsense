import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))


# ── 黑天鹅触发与退出条件 ─────────────────────────────────────────────────
def detect_black_swan():
    signals  = {}
    triggers = []

    pw_path = os.path.join(ROOT_DIR, "data", "raw", "portwatch_chokepoints.csv")
    if os.path.exists(pw_path):
        df = pd.read_csv(pw_path, index_col=0, parse_dates=True)

        if "hormuz_blocked" in df.columns:
            cp6          = df["cp6_tanker"].dropna()
            latest       = int(cp6.iloc[-1])
            avg_90d      = round(float(cp6.tail(90).mean()), 1)
            latest_flag  = int(df["hormuz_blocked"].dropna().iloc[-1])
            signals["hormuz_blocked"]       = latest_flag
            signals["hormuz_tanker_latest"] = latest
            signals["hormuz_tanker_avg90d"] = avg_90d
            signals["hormuz_ratio"]         = round(latest / avg_90d, 3) if avg_90d > 0 else 0
            if latest_flag == 1:
                triggers.append(
                    "霍尔木兹海峡油轮通过量低于历史均值50%"
                    "（当前{}艘/日均{}艘）".format(latest, avg_90d)
                )

        if "mandeb_blocked" in df.columns:
            cp4         = df["cp4_tanker"].dropna()
            m_latest    = int(cp4.iloc[-1])
            m_avg       = round(float(cp4.tail(90).mean()), 1)
            m_flag      = int(df["mandeb_blocked"].dropna().iloc[-1])
            signals["mandeb_blocked"]       = m_flag
            signals["mandeb_tanker_latest"] = m_latest
            signals["mandeb_tanker_avg90d"] = m_avg
            if m_flag == 1:
                triggers.append(
                    "曼德海峡油轮通过量低于历史均值50%"
                    "（当前{}艘/日均{}艘）".format(m_latest, m_avg)
                )

        if "cp11_tanker" in df.columns:
            cp11 = df["cp11_tanker"].dropna()
            if len(cp11) > 90:
                z = float((cp11.iloc[-1] - cp11.tail(90).mean()) /
                          (cp11.tail(90).std() + 1e-6))
                signals["taiwan_zscore"] = round(z, 2)
                if abs(z) > 2.5:
                    triggers.append(
                        "台湾海峡船舶通过量异常（z-score={:.2f}）".format(z)
                    )

    macro_path = os.path.join(ROOT_DIR, "data", "raw", "macro_data.csv")
    if os.path.exists(macro_path):
        macro = pd.read_csv(macro_path, index_col=0, parse_dates=True)
        if "VIX" in macro.columns:
            vix = float(macro["VIX"].dropna().iloc[-1])
            signals["VIX"] = round(vix, 2)
            if vix > 45:
                triggers.append("VIX极端恐慌（当前{:.1f} > 45）".format(vix))

    gdelt_path = os.path.join(ROOT_DIR, "data", "raw", "gdelt_sentiment.csv")
    if os.path.exists(gdelt_path):
        gdelt = pd.read_csv(gdelt_path, index_col=0, parse_dates=True)
        if "gdelt_conflict_intensity" in gdelt.columns:
            ci        = gdelt["gdelt_conflict_intensity"].dropna()
            ci_latest = float(ci.iloc[-1])
            ci_p95    = float(ci.quantile(0.95))
            signals["gdelt_conflict_intensity"] = round(ci_latest, 4)
            signals["gdelt_conflict_p95"]       = round(ci_p95, 4)
            if ci_latest > ci_p95:
                triggers.append(
                    "GDELT冲突强度超过历史95分位"
                    "（当前{:.4f} > p95={:.4f}）".format(ci_latest, ci_p95)
                )

    # ── 退出条件：油轮数量连续3天回升超过均值30% ─────────────────────────
    exit_signals = []
    if os.path.exists(pw_path):
        df_exit = pd.read_csv(pw_path, index_col=0, parse_dates=True)
        if "cp6_tanker" in df_exit.columns:
            cp6_exit  = df_exit["cp6_tanker"].dropna()
            avg_exit  = float(cp6_exit.tail(90).mean())
            recent_3d = cp6_exit.iloc[-3:]
            if len(recent_3d) == 3 and all(recent_3d > avg_exit * 0.3):
                exit_signals.append(
                    "霍尔木兹油轮通过量已连续3日回升至均值30%以上"
                    "（最近3日：{}艘，90日均值：{:.1f}艘）".format(
                        list(recent_3d.astype(int)), avg_exit
                    )
                )

    signals["exit_signals"]  = exit_signals
    signals["auto_exit"]     = len(exit_signals) > 0
    signals["triggers"]      = triggers
    signals["trigger_count"] = len(triggers)
    signals["detected_at"]   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    is_black_swan = len(triggers) >= 1 and not signals["auto_exit"]
    return is_black_swan, signals


# ── 构建丰富上下文 ────────────────────────────────────────────────────────
def _build_context(signals: dict, recent_news: list = None) -> str:
    pw_path   = os.path.join(ROOT_DIR, "data", "raw", "portwatch_chokepoints.csv")
    oil_path  = os.path.join(ROOT_DIR, "data", "raw", "oil_prices.csv")
    macro_path= os.path.join(ROOT_DIR, "data", "raw", "macro_data.csv")

    # 封锁持续天数
    blockade_days  = 0
    hormuz_history = ""
    cape_text      = ""
    mandeb_text    = ""
    if os.path.exists(pw_path):
        df_pw = pd.read_csv(pw_path, index_col=0, parse_dates=True)
        if "hormuz_blocked" in df_pw.columns:
            blocked = df_pw["hormuz_blocked"].dropna()
            if int(blocked.iloc[-1]) == 1:
                # 找到最近一次连续封锁的起始日（从后往前找第一个0之后的1）
                unblocked_idx = blocked[blocked == 0].index
                if len(unblocked_idx) > 0:
                    last_normal = unblocked_idx[-1]
                    start_date = blocked[blocked.index > last_normal][blocked == 1].index[0]
                else:
                    start_date = blocked[blocked == 1].index[0]
                blockade_days = (blocked.index[-1] - start_date).days + 1
            cp6 = df_pw["cp6_tanker"].dropna()
            # 动态封锁起始日，不硬编码
            _blockade_start = start_date if int(blocked.iloc[-1]) == 1 and "start_date" in dir() else cp6.index[-30]
            pre_mean  = float(cp6[cp6.index < _blockade_start].tail(30).mean()) if len(cp6[cp6.index < _blockade_start]) >= 5 else float(cp6.head(30).mean())
            post_mean = float(cp6[cp6.index >= _blockade_start].mean()) if len(cp6[cp6.index >= _blockade_start]) > 0 else float(cp6.iloc[-1])
            drop_pct  = (1 - post_mean / pre_mean) * 100 if pre_mean > 0 else 0
            hormuz_history = (
                "封锁前30日均值：{:.1f}艘/日，"
                "封锁后均值：{:.1f}艘/日，降幅：{:.1f}%".format(
                    pre_mean, post_mean, drop_pct)
            )
        if "cape_reroute_signal" in df_pw.columns:
            cape_val  = float(df_pw["cape_reroute_signal"].dropna().iloc[-1])
            cape_text = "好望角绕行信号：{:.2f}（>1.0表示绕行流量高于正常水平）".format(cape_val)
        if "cp4_tanker" in df_pw.columns:
            cp4        = df_pw["cp4_tanker"].dropna()
            m_latest   = int(cp4.iloc[-1])
            m_avg      = float(cp4.tail(90).mean())
            mandeb_text= "曼德海峡：当前{}艘/日，均值{:.1f}艘/日".format(m_latest, m_avg)

    # 油价走势
    oil_context = ""
    if os.path.exists(oil_path):
        df_oil = pd.read_csv(oil_path, index_col=0, parse_dates=True)
        if "WTI" in df_oil.columns:
            wti    = df_oil["WTI"].dropna()
            latest = float(wti.iloc[-1])
            w1ago  = float(wti.iloc[-6])  if len(wti) > 6  else latest
            m1ago  = float(wti.iloc[-22]) if len(wti) > 22 else latest
            # 动态找封锁前基准价，不硬编码日期
            _pw_path = os.path.join(ROOT_DIR, "data", "raw", "portwatch_chokepoints.csv")
            _blockade_dt = None
            if os.path.exists(_pw_path):
                _df_pw2 = pd.read_csv(_pw_path, index_col=0, parse_dates=True)
                if "hormuz_blocked" in _df_pw2.columns:
                    _bl = _df_pw2["hormuz_blocked"].dropna()
                    if len(_bl) > 0 and int(_bl.iloc[-1]) == 1:
                        _unbl = _bl[_bl == 0]
                        if len(_unbl) > 0:
                            _blockade_dt = _bl[_bl.index > _unbl.index[-1]].index[0]
                        else:
                            _blockade_dt = _bl[_bl == 1].index[0]
            if _blockade_dt is not None:
                _pre_wti = wti[wti.index < _blockade_dt]
                pre_b = float(_pre_wti.iloc[-1]) if len(_pre_wti) > 0 else latest
                pre_label = _blockade_dt.strftime("%m月%d日") + "封锁前"
            else:
                pre_b = float(wti.iloc[-30]) if len(wti) > 30 else latest
                pre_label = "30日前"
            oil_context = (
                "WTI最新价：{:.2f}美元/桶\n"
                "基准价格（{}）：{:.2f}美元/桶\n"
                "区间涨幅：{:.1f}%\n"
                "1周前：{:.2f}，1个月前：{:.2f}".format(
                    latest, pre_label, pre_b,
                    (latest / pre_b - 1) * 100 if pre_b > 0 else 0,
                    w1ago, m1ago)
            )

    # 宏观数据
    macro_context = ""
    if os.path.exists(macro_path):
        df_m  = pd.read_csv(macro_path, index_col=0, parse_dates=True)
        parts = []
        for label, col in [("VIX", "VIX"), ("美元指数", "DXY"),
                            ("美联储利率", "FED_RATE"), ("10年期美债", "US10Y")]:
            if col in df_m.columns:
                val = float(df_m[col].dropna().iloc[-1])
                parts.append("{}：{:.2f}".format(label, val))
        macro_context = "、".join(parts)

    # 新闻
    news_text = ""
    if recent_news:
        news_text = "\n近期关键新闻（最新8条）：\n"
        for n in recent_news[:8]:
            news_text += "- {}: {}\n".format(
                str(n.get("date", ""))[:10], n.get("title", ""))

    triggers_text = "\n".join(["- " + t for t in signals.get("triggers", [])])

    context = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【触发信号】
{triggers}

【霍尔木兹封锁详情】
- 封锁持续时间：约{days}天（自2026年3月2日起）
- {hormuz}
- {cape}
- {mandeb}

【油价走势】
{oil}

【宏观环境】
{macro}

【历史类比案例】
- 1990年海湾战争：油价从$17涨至$46（+170%），持续约6个月
- 2019年沙特阿美袭击：油价单日暴涨15%，2周内回落50%
- 2022年俄乌冲突：油价从$90涨至$130（+44%），持续约3个月
- 当前霍尔木兹封锁：全球约20%石油供应受阻，规模历史罕见
{news}━━━━━━━━━━━━━━━━━━━━━━━""".format(
        triggers=triggers_text,
        days=blockade_days,
        hormuz=hormuz_history,
        cape=cape_text,
        mandeb=mandeb_text,
        oil=oil_context,
        macro=macro_context,
        news=news_text,
    )
    return context


# ── DeepSeek 深度分析 ─────────────────────────────────────────────────────
def run_deepseek_analysis(signals: dict, recent_news: list = None) -> dict:
    try:
        from openai import OpenAI
    except ImportError:
        return {"error": "openai 库未安装"}

    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        return {"error": "DEEPSEEK_API_KEY 未配置"}

    client  = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    context = _build_context(signals, recent_news)

    prompt = context + """

请作为花旗银行大宗商品部首席分析师，提供以下深度分析
（中文，专业金融语言，总字数不超过700字）：

**1. 事件定性与严重程度**
（结合封锁天数、油轮降幅、历史类比，评定严重程度：轻微/中等/严重/极端，并给出理由）

**2. 供需缺口分析**
（从供应缺口规模、替代路线能力、战略储备释放、需求侧响应四个维度量化分析）

**3. 分情景价格路径预测（未来30日）​**
请用表格呈现：
| 情景 | 触发条件 | WTI价格区间 | 发生概率 |
|------|---------|------------|---------|
| 快速缓解 | ... | $XX~$XX | XX% |
| 僵持延续 | ... | $XX~$XX | XX% |
| 全面升级 | ... | $XX~$XX | XX% |

**4. 行业冲击信号**
- 航空业：燃油成本影响及建议
- 化工/炼化：原料成本及库存策略
- 能源贸易商：套利机会及头寸建议
- 航运业：绕行成本及运费走势

**5. 企业风险对冲操作建议**
（3条具体可执行建议，注明工具类型/方向/比例/期限）

**6. 每日关键监测指标**
（3-4个指标，注明阈值和含义）"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system",
                 "content": (
                     "你是花旗银行大宗商品部首席分析师，"
                     "专注地缘政治风险对大宗商品价格的影响。"
                     "分析需有具体数字支撑，避免空泛，给出可执行建议。"
                 )},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.2,
        )
        analysis_text = response.choices[0].message.content
        return {
            "analysis"    : analysis_text,
            "model"       : "deepseek-chat",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "signals"     : signals,
            "status"      : "ok",
        }
    except Exception as e:
        return {"error": str(e), "signals": signals, "status": "error"}


# ── 主入口 ────────────────────────────────────────────────────────────────
def get_black_swan_report(recent_news: list = None) -> dict:
    is_black_swan, signals = detect_black_swan()

    if not is_black_swan:
        return {
            "is_black_swan": False,
            "signals"      : signals,
            "message"      : "当前未检测到极端事件信号，模型预测正常运行"
        }

    print("黑天鹅信号触发：" + str(signals["triggers"]))
    analysis = run_deepseek_analysis(signals, recent_news)

    report = {
        "is_black_swan": True,
        "signals"      : signals,
        "analysis"     : analysis,
    }
    out_path = os.path.join(ROOT_DIR, "data", "raw", "black_swan_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return report


if __name__ == "__main__":
    print("正在检测黑天鹅信号...")
    is_bs, signals = detect_black_swan()
    print("黑天鹅状态：" + ("是" if is_bs else "否"))
    print("触发信号：")
    for t in signals.get("triggers", []):
        print("  - " + t)
    print("退出信号：")
    for e in signals.get("exit_signals", []):
        print("  - " + e)
    print()

    if is_bs:
        print("正在调用 DeepSeek 进行深度情景分析...")
        report = get_black_swan_report()
        if "analysis" in report and report["analysis"].get("status") == "ok":
            print()
            print("── DeepSeek 深度分析结果 ──")
            print(report["analysis"]["analysis"])
        else:
            print("分析失败：", report.get("analysis", {}).get("error", "未知错误"))
    else:
        if signals.get("auto_exit"):
            print("黑天鹅已自动解除：", signals["exit_signals"])
        else:
            print("当前无黑天鹅事件")
