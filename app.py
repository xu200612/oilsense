import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from xgboost import XGBRegressor
from dotenv import load_dotenv
from datetime import datetime


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))

st.set_page_config(
    page_title="OilSense | 原油风险智能预警系统",
    page_icon="🛢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── 产油国数据（静态，来源EIA 2024年报）─────────────────────────────────────
OIL_COUNTRIES = {
    "美国": {"lat": 38.0, "lon": -97.0, "code": "US", "prod": 12.9, "share": 13.2},
    "俄罗斯": {"lat": 61.0, "lon": 90.0, "code": "RS", "prod": 10.1, "share": 10.3},
    "沙特阿拉伯": {"lat": 24.0, "lon": 45.0, "code": "SA", "prod": 9.6, "share": 9.8},
    "伊拉克": {"lat": 33.0, "lon": 44.0, "code": "IZ", "prod": 4.2, "share": 4.3},
    "伊朗": {"lat": 32.0, "lon": 53.0, "code": "IR", "prod": 3.4, "share": 3.5},
    "阿联酋": {"lat": 24.0, "lon": 54.0, "code": "AE", "prod": 3.2, "share": 3.3},
    "科威特": {"lat": 29.0, "lon": 47.0, "code": "KU", "prod": 2.7, "share": 2.8},
    "挪威": {"lat": 60.0, "lon": 10.0, "code": "NO", "prod": 1.8, "share": 1.8},
    "哈萨克斯坦": {"lat": 48.0, "lon": 68.0, "code": "KZ", "prod": 1.8, "share": 1.8},
    "尼日利亚": {"lat": 9.0, "lon": 8.0, "code": "NG", "prod": 1.5, "share": 1.5},
    "利比亚": {"lat": 27.0, "lon": 17.0, "code": "LY", "prod": 1.2, "share": 1.2},
    "委内瑞拉": {"lat": 8.0, "lon": -66.0, "code": "VE", "prod": 0.9, "share": 0.9},
    "阿尔及利亚": {"lat": 28.0, "lon": 2.0, "code": "AL", "prod": 0.9, "share": 0.9},
}

# 地缘政治重点事件（用于地球上的感叹号标注）
GEO_EVENTS = [
    {"lat": 32.0, "lon": 53.0, "label": "美伊紧张局势", "severity": "high"},
    {"lat": 24.0, "lon": 45.0, "label": "OPEC+减产协议", "severity": "medium"},
    {"lat": 61.0, "lon": 90.0, "label": "俄罗斯能源制裁", "severity": "high"},
    {"lat": 31.5, "lon": 34.8, "label": "中东冲突外溢", "severity": "high"},
    {"lat": 60.0, "lon": 10.0, "label": "北欧天然气供应", "severity": "low"},
]
@st.cache_data(ttl=3600)
def get_country_risk():
    try:
        from country_risk import compute_country_risk
        return compute_country_risk(days_back=7)
    except Exception as e:
        # 降级到静态风险
        return {}


# ── 加载数据和模型 ─────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
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
    importance = pd.read_csv(
        os.path.join(ROOT_DIR, "data", "processed", "feature_importance.csv")
    )
    sentiment = pd.read_csv(
        os.path.join(ROOT_DIR, "data", "processed", "daily_sentiment.csv"),
        parse_dates=["date"]
    )
    news = pd.DataFrame()
    news_path = os.path.join(ROOT_DIR, "data", "raw", "news_data.csv")
    if os.path.exists(news_path):
        news = pd.read_csv(news_path)
    return feat, model_feature_map, models, importance, sentiment, news
@st.cache_data(ttl=3600)  # 每小时刷新
def load_portwatch():
    """加载 PortWatch 咽喉点数据"""
    pw_path = os.path.join(ROOT_DIR, "data", "raw", "portwatch_chokepoints.csv")
    if not os.path.exists(pw_path):
        return None
    df = pd.read_csv(pw_path, index_col=0, parse_dates=True)
    return df

@st.cache_data(ttl=3600)
def get_black_swan_status():
    """检测黑天鹅状态，优先读缓存报告"""
    try:
        from black_swan import detect_black_swan, get_black_swan_report
        is_bs, signals = detect_black_swan()
        return is_bs, signals
    except Exception as e:
        return False, {"error": str(e)}

@st.cache_data(ttl=3600)
def get_chokepoint_status():
    """获取各咽喉点当前状态"""
    df = load_portwatch()
    if df is None:
        return {}

    cp_map = {
        "cp6" : {"name": "霍尔木兹", "importance": "全球20%石油"},
        "cp4" : {"name": "曼德海峡", "importance": "红海通道"},
        "cp1" : {"name": "苏伊士运河", "importance": "欧亚航线"},
        "cp5" : {"name": "马六甲", "importance": "亚洲石油"},
        "cp3" : {"name": "博斯普鲁斯", "importance": "俄油出口"},
        "cp11": {"name": "台湾海峡", "importance": "亚太航运"},
        "cp7" : {"name": "好望角", "importance": "绕行路线"},
    }

    status = {}
    for cp, info in cp_map.items():
        col = f"{cp}_tanker"
        if col not in df.columns:
            continue
        series   = df[col].dropna()
        if len(series) < 2:
            continue
        latest   = series.iloc[-1]
        avg_90d  = series.tail(90).mean()
        ratio    = latest / avg_90d if avg_90d > 0 else 1.0
        last_date = str(series.index[-1].date())

        if ratio < 0.3:
            risk, color = "极高风险", "#e74c3c"
        elif ratio < 0.6:
            risk, color = "高风险",   "#e67e22"
        elif ratio < 0.85:
            risk, color = "偏低",     "#f1c40f"
        else:
            risk, color = "正常",     "#2ecc71"

        status[cp] = {
            "name"          : info["name"],
            "importance"    : info["importance"],
            "latest"        : int(latest),
            "avg_90d"       : round(avg_90d, 1),
            "ratio"         : round(ratio * 100, 1),
            "risk"          : risk,
            "color"         : color,
            "last_date"     : last_date,
        }
    return status

@st.cache_data(ttl=300)  # 每5分钟刷新一次
def get_realtime_price():
    """从 Yahoo Finance 拉取实时 WTI 价格"""
    try:
        import requests
        url     = "https://query1.finance.yahoo.com/v8/finance/chart/CL=F"
        headers = {"User-Agent": "Mozilla/5.0"}
        r       = requests.get(url, headers=headers, timeout=10)
        data    = r.json()
        meta    = data["chart"]["result"][0]["meta"]
        price   = meta["regularMarketPrice"]
        ts      = meta["regularMarketTime"]
        date    = datetime.fromtimestamp(ts).strftime("%Y-%m-%d")
        return price, date, True
    except Exception as e:
        # 失败则回退到 feature_matrix 里的最新价格
        fallback_price = feat["WTI"].iloc[-1]
        fallback_date  = str(feat.index[-1].date())
        return fallback_price, fallback_date, False

feat, model_feature_map, models, importance, sentiment, news = load_assets()


@st.cache_data
def get_predictions():
    pred_df = pd.DataFrame(index=feat.index)
    pred_df["WTI"] = feat["WTI"]
    pred_df["target"] = feat["target"]
    for name, model in models.items():
        cols = model_feature_map[name]
        pred_df["pred_" + name] = model.predict(feat[cols])
    return pred_df


pred_df = get_predictions()


def get_risk_level(low, mid, high):
    spread = high - low
    if spread > 0.15:
        return "极高风险", "#e74c3c", 5
    elif spread > 0.10:
        return "高风险", "#e67e22", 4
    elif spread > 0.07:
        return "中等风险", "#f1c40f", 3
    elif spread > 0.04:
        return "低风险", "#2ecc71", 2
    else:
        return "极低风险", "#27ae60", 1


def get_country_news(country_name, n=3):
    """从新闻数据里提取该国相关新闻"""
    if len(news) == 0:
        return []
    keywords = {
        "伊朗": ["Iran", "Iranian"],
        "沙特阿拉伯": ["Saudi", "Riyadh", "Aramco"],
        "俄罗斯": ["Russia", "Russian", "Moscow", "Kremlin"],
        "伊拉克": ["Iraq", "Iraqi", "Baghdad"],
        "美国": ["US", "America", "Washington", "Biden", "Trump"],
        "阿联酋": ["UAE", "Dubai", "Abu Dhabi"],
        "挪威": ["Norway", "Norwegian", "Equinor"],
        "委内瑞拉": ["Venezuela", "Maduro"],
        "利比亚": ["Libya", "Libyan"],
        "尼日利亚": ["Nigeria", "Nigerian"],
        "科威特": ["Kuwait"],
        "哈萨克斯坦": ["Kazakhstan"],
        "阿尔及利亚": ["Algeria"],
    }
    kws = keywords.get(country_name, [country_name])
    mask = news["title"].str.contains("|".join(kws), case=False, na=False)
    filtered = news[mask].head(n)
    return filtered[["date", "title", "source", "url"]].to_dict("records")


# ── 侧边栏 ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛢 OilSense")
    st.caption("原油风险智能预警系统")
    st.divider()
    page = st.radio(
        "导航",
        ["全球能源地图", "市场概览", "风险预测", "历史回测"],
        label_visibility="collapsed"
    )
    st.divider()
    last_low = pred_df["pred_enhanced_low"].iloc[-1]
    last_mid = pred_df["pred_enhanced_mid"].iloc[-1]
    last_high = pred_df["pred_enhanced_high"].iloc[-1]
    risk_label, risk_color, _ = get_risk_level(last_low, last_mid, last_high)

    rt_price, rt_date, rt_live = get_realtime_price()
    st.metric(
        "WTI 现价",
        "$" + str(round(rt_price, 2)),
        "实时" if rt_live else "延迟数据"
    )
    st.metric("价格日期", rt_date)
    if rt_live:
        st.caption("🟢 实时行情（Yahoo Finance）")
    else:
        st.caption("🟡 使用缓存数据")
    st.markdown(
        "<div style='background:" + risk_color + ";padding:8px;border-radius:6px;"
                                                 "text-align:center;color:white;font-weight:bold;'>"
        + risk_label + "</div>", unsafe_allow_html=True
    )
    st.divider()
    st.caption("数据来源：FRED / EIA / NewsAPI / GDELT / DeepSeek")
    st.caption("注：价格数据存在约2周官方发布延迟")

# ══════════════════════════════════════════════════════════════════════════
# 页面一：全球能源地图（3D地球）
# ══════════════════════════════════════════════════════════════════════════
# ── 黑天鹅预警横幅（全页面显示）─────────────────────────────────────────
is_black_swan, bs_signals = get_black_swan_status()

if is_black_swan:
    triggers_text = " ｜ ".join(bs_signals.get("triggers", []))
    st.markdown(
        """
        <div style='background:linear-gradient(90deg,#7b0000,#c0392b);
        padding:14px 20px;border-radius:8px;margin-bottom:16px;
        border:1px solid #e74c3c;'>
        <span style='font-size:20px;'>🚨</span>
        <span style='color:white;font-weight:bold;font-size:16px;margin-left:8px;'>
        极端事件预警：统计模型已暂停，当前由 AI 情景分析接管
        </span><br>
        <span style='color:#ffcccc;font-size:13px;margin-left:28px;'>
        """ + triggers_text + """
        </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 展开显示 DeepSeek 分析
    with st.expander("查看 AI 情景分析详情", expanded=True):
        report_path = os.path.join(ROOT_DIR, "data", "raw", "black_swan_report.json")
        if os.path.exists(report_path):
            import json
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            analysis = report.get("analysis", {}).get("analysis", "")
            if analysis:
                st.markdown(analysis)
                gen_time = report.get("analysis", {}).get("generated_at", "")
                st.caption("AI 分析生成时间：" + gen_time +
                           " ｜ 模型：DeepSeek-Chat ｜ 数据源：IMF PortWatch")
            else:
                st.warning("分析报告生成中，请稍后刷新")
        else:
            # 实时生成
            with st.spinner("正在调用 DeepSeek 进行情景分析..."):
                try:
                    from black_swan import get_black_swan_report
                    report = get_black_swan_report()
                    analysis = report.get("analysis", {}).get("analysis", "")
                    if analysis:
                        st.markdown(analysis)
                    else:
                        st.warning("分析生成失败，请检查 DEEPSEEK_API_KEY")
                except Exception as e:
                    st.error("黑天鹅分析调用失败：" + str(e))

if page == "全球能源地图":
    st.title("全球能源地图")
    st.caption("点击产油国查看详细分析 · 感叹号标注为当前地缘政治热点")

    # ── 构建地球图 ─────────────────────────────────────────────────────────
    fig = go.Figure()

    # 产油国圆圈（大小=产量，颜色=风险等级）
    country_names = list(OIL_COUNTRIES.keys())
    lats = [v["lat"] for v in OIL_COUNTRIES.values()]
    lons = [v["lon"] for v in OIL_COUNTRIES.values()]
    prods = [v["prod"] for v in OIL_COUNTRIES.values()]
    shares = [v["share"] for v in OIL_COUNTRIES.values()]

    country_risk_data = get_country_risk()

    # 动态风险颜色，降级到静态
    static_high = ["伊朗", "俄罗斯", "利比亚", "委内瑞拉"]
    static_mid = ["伊拉克", "尼日利亚", "阿尔及利亚"]
    colors = []
    high_risk = []
    mid_risk = []

    for name in country_names:
        if name in country_risk_data:
            color = country_risk_data[name]["color"]
            level = country_risk_data[name]["level"]
            if level == "高风险":
                high_risk.append(name)
            elif level in ["中等风险"]:
                mid_risk.append(name)
        else:
            # 降级静态
            if name in static_high:
                color = "#e74c3c"
                high_risk.append(name)
            elif name in static_mid:
                color = "#e67e22"
                mid_risk.append(name)
            else:
                color = "#2ecc71"
        colors.append(color)

    hover_texts = []
    for name in country_names:
        info = OIL_COUNTRIES[name]
        country_news = get_country_news(name, n=2)
        news_str = ""
        for n_item in country_news:
            news_str += "<br>📰 " + str(n_item["title"])[:45] + "..."
        risk_info = country_risk_data.get(name, {})
        risk_str = ""
        if risk_info:
            risk_str = (
                    "风险等级：" + risk_info["level"] +
                    "（" + str(risk_info["score"]) + "）<br>" +
                    "新闻情感：" + str(risk_info["news_sentiment"]) +
                    " / 新闻数：" + str(risk_info["news_count"]) + "<br>"
            )
        hover_texts.append(
            "<b>" + name + "</b><br>" +
            "日产量：" + str(info["prod"]) + " 百万桶/天<br>" +
            "全球占比：" + str(info["share"]) + "%<br>" +
            risk_str +
            news_str
        )

    fig.add_trace(go.Scattergeo(
        lat=lats,
        lon=lons,
        text=hover_texts,
        hoverinfo="text",
        name="产油国",
        marker=dict(
            size=[p * 4 + 8 for p in prods],
            color=colors,
            opacity=0.85,
            line=dict(color="white", width=1.5),
            sizemode="diameter",
        ),
        customdata=country_names,
    ))

    # 国家名称标签
    fig.add_trace(go.Scattergeo(
        lat=lats,
        lon=lons,
        text=country_names,
        mode="text",
        textfont=dict(size=9, color="white"),
        hoverinfo="skip",
        name="标签",
        showlegend=False,
    ))

    # 地缘政治事件感叹号标注
    event_colors = {"high": "#e74c3c", "medium": "#f1c40f", "low": "#2ecc71"}
    for event in GEO_EVENTS:
        fig.add_trace(go.Scattergeo(
            lat=[event["lat"]],
            lon=[event["lon"]],
            text=["⚠ " + event["label"]],
            mode="text+markers",
            marker=dict(
                symbol="triangle-up",
                size=14,
                color=event_colors[event["severity"]],
                opacity=0.9,
            ),
            textfont=dict(size=8, color=event_colors[event["severity"]]),
            textposition="top center",
            hovertext=event["label"],
            hoverinfo="text",
            name=event["label"],
            showlegend=False,
        ))

    fig.update_layout(
        height=580,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#0e1117",
        geo=dict(
            projection_type="orthographic",
            showland=True,
            landcolor="#1a2332",
            showocean=True,
            oceancolor="#0d1b2a",
            showlakes=False,
            showcountries=True,
            countrycolor="#2d3748",
            showcoastlines=True,
            coastlinecolor="#4a5568",
            bgcolor="#0e1117",
            projection_rotation=dict(lon=45, lat=20, roll=0),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(color="white", size=10),
            x=0.01, y=0.99
        )
    )

    st.plotly_chart(fig, use_container_width=True)
    # ── 咽喉点航运状态面板 ────────────────────────────────────────────────
    st.divider()
    st.subheader("全球关键咽喉点实时状态")
    st.caption("数据来源：IMF PortWatch · 每周二更新 · 油轮通过量与90日均值对比")

    cp_status = get_chokepoint_status()
    if cp_status:
        cols = st.columns(len(cp_status))
        for i, (cp, s) in enumerate(cp_status.items()):
            with cols[i]:
                st.markdown(
                    f"""
                    <div style='background:#1a2332;border:1px solid {s["color"]};
                    border-radius:8px;padding:10px;text-align:center;'>
                    <div style='color:{s["color"]};font-weight:bold;font-size:13px;'>
                    {s["name"]}</div>
                    <div style='color:white;font-size:22px;font-weight:bold;margin:4px 0;'>
                    {s["latest"]}</div>
                    <div style='color:#aaa;font-size:11px;'>油轮/日（均值{s["avg_90d"]}）</div>
                    <div style='background:{s["color"]};color:white;border-radius:4px;
                    padding:2px 6px;font-size:11px;margin-top:6px;display:inline-block;'>
                    {s["risk"]} {s["ratio"]}%</div>
                    <div style='color:#666;font-size:10px;margin-top:4px;'>
                    {s["importance"]}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.caption("PortWatch 数据未加载，运行 python fetch_portwatch.py 生成")

    # ── 图例说明 ───────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        high_names = "、".join(high_risk) if high_risk else "无"
        st.markdown("🔴 **高风险产油国**")
        st.caption(high_names)
    with col2:
        mid_names = "、".join(mid_risk) if mid_risk else "无"
        st.markdown("🟠 **中等风险产油国**")
        st.caption(mid_names)
    with col3:
        low_names = "、".join([n for n in country_names
                              if n not in high_risk and n not in mid_risk])
        st.markdown("🟢 **低风险产油国**")
        st.caption(low_names)
    with col4:
        st.markdown("⚠ **地缘政治热点**")
        st.caption("当前活跃冲突/制裁/协议区域")

    st.divider()

    # ── 点击国家后展示详情 ─────────────────────────────────────────────────
    st.subheader("产油国详细分析")
    selected = st.selectbox(
        "选择国家查看详情",
        options=list(OIL_COUNTRIES.keys()),
        index=2  # 默认沙特
    )

    info = OIL_COUNTRIES[selected]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("日产量", str(info["prod"]) + " 百万桶/天")
    with c2:
        st.metric("全球占比", str(info["share"]) + "%")
    with c3:
        if selected in country_risk_data:
            r = country_risk_data[selected]
            if r["level"] == "高风险":
                icon = "🔴"
            elif r["level"] == "中等风险":
                icon = "🟠"
            elif r["level"] == "低风险":
                icon = "🟡"
            else:
                icon = "🟢"
            risk_status = icon + " " + r["level"] + "（动态）"
        else:
            if selected in static_high:
                risk_status = "⚠ 高风险"
            elif selected in static_mid:
                risk_status = "⚡ 中等风险"
            else:
                risk_status = "✅ 低风险"

    with c4:
        st.metric("所在地区",
                  "中东" if info["lon"] > 30 and info["lat"] < 40 and info["lon"] < 60
                  else "欧洲/中亚" if info["lon"] > 0 and info["lat"] > 40
                  else "美洲" if info["lon"] < 0
                  else "非洲" if info["lat"] < 20
                  else "其他")

    # 该国相关新闻
    country_news = get_country_news(selected, n=5)
    if country_news:
        st.markdown("#### 相关新闻信号")
        for item in country_news:
            source_badge = "📡 " + str(item.get("source", ""))
            url = item.get("url", "")
            title = str(item.get("title", ""))
            date = str(item.get("date", ""))
            if url:
                st.markdown(
                    "- **​" + date + "​** &nbsp; [" + title + "](" + url + ")  `" +
                    source_badge + "`"
                )
            else:
                st.markdown(
                    "- **​" + date + "​** &nbsp;" + title + "  `" + source_badge + "`"
                )
    else:
        st.caption("暂无该国相关新闻数据")

    st.divider()

    # ── 全球产量分布饼图 ───────────────────────────────────────────────────
    st.subheader("全球原油产量分布")
    col1, col2 = st.columns([1, 1])

    with col1:
        fig_pie = go.Figure(go.Pie(
            labels=country_names,
            values=prods,
            marker=dict(colors=colors),
            hole=0.4,
            textinfo="label+percent",
            textfont=dict(size=11),
        ))
        fig_pie.update_layout(
            height=380,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="#0e1117",
            font=dict(color="white"),
            showlegend=False,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # 产量排名条形图
        sorted_countries = sorted(OIL_COUNTRIES.items(),
                                  key=lambda x: x[1]["prod"], reverse=True)
        fig_bar = go.Figure(go.Bar(
            x=[v["prod"] for _, v in sorted_countries],
            y=[k for k, _ in sorted_countries],
            orientation="h",
            marker_color=[
                "#e74c3c" if k in high_risk else
                "#e67e22" if k in mid_risk else "#2ecc71"
                for k, _ in sorted_countries
            ],
            text=[str(v["prod"]) + " Mb/d" for _, v in sorted_countries],
            textposition="outside",
        ))
        fig_bar.update_layout(
            height=380,
            margin=dict(l=0, r=30, t=10, b=0),
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font=dict(color="white"),
            xaxis=dict(gridcolor="#2d3748", title="百万桶/天"),
            yaxis=dict(gridcolor="#2d3748"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════
# 页面二：市场概览
# ══════════════════════════════════════════════════════════════════════════
elif page == "市场概览":
    st.title("市场概览")
    st.caption("基于最新数据的油市关键指标速览")

    last_low = pred_df["pred_enhanced_low"].iloc[-1]
    last_mid = pred_df["pred_enhanced_mid"].iloc[-1]
    last_high = pred_df["pred_enhanced_high"].iloc[-1]
    risk_label, risk_color, _ = get_risk_level(last_low, last_mid, last_high)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        rt_price, rt_date, rt_live = get_realtime_price()
        wti_prev = feat["WTI"].dropna().iloc[-1]  # 用历史最新做delta参考
        delta = round(rt_price - wti_prev, 2)
        st.metric(
            "WTI 现价（美元/桶）",
            "${:.2f}".format(rt_price),
            "{:+.2f}".format(delta),
        )
        if rt_live:
            st.caption("🟢 实时行情（Yahoo Finance）")
        else:
            st.caption("🟡 缓存数据")
    with c2:
        st.metric("10日预测中位涨跌",
                  str(round(last_mid * 100, 2)) + "%",
                  "区间 [" + str(round(last_low * 100, 1)) + "%, " +
                  str(round(last_high * 100, 1)) + "%]")
    with c3:
        st.metric("当前风险等级", risk_label)
    with c4:
        if len(sentiment) > 0:
            latest_sent = sentiment.sort_values("date").iloc[-1]
            st.metric("最新情绪评分",
                      str(round(latest_sent["sentiment_score"], 3)),
                      "新闻数：" + str(int(latest_sent["news_count"])))
        else:
            st.metric("最新情绪评分", "N/A")

    st.divider()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("近90日 WTI 油价走势")
        recent = pred_df.tail(90).copy()
        rt_price_chart, _, rt_live_chart = get_realtime_price()

        # 把实时价格作为最新一行补进去
        today = pd.Timestamp.now().normalize()
        if today not in recent.index:
            new_row = pd.Series({"WTI": rt_price_chart}, name=today)
            recent = pd.concat([recent, new_row.to_frame().T])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent.index, y=recent["WTI"],
            name="WTI 油价",
            line=dict(color="#3498db", width=2),
            connectgaps=True,
        ))
        # 实时价格标注
        fig.add_trace(go.Scatter(
            x=[today],
            y=[rt_price_chart],
            mode="markers+text",
            name="实时价格",
            marker=dict(color="#e74c3c", size=12, symbol="star"),
            text=["实时 ${:.2f}".format(rt_price_chart)],
            textposition="top center",
            textfont=dict(color="#e74c3c", size=11),
        ))
        # 数据延迟分界线
        last_data_date = pred_df.index[-1]
        fig.add_vline(
            x=last_data_date.timestamp() * 1000,
            line_dash="dash",
            line_color="#666",
            annotation_text="数据截止",
            annotation_font_color="#666",
            annotation_font_size=10,
        )
        fig.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="white"),
            xaxis=dict(gridcolor="#2d3748"),
            yaxis=dict(gridcolor="#2d3748", title="美元/桶"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("关键宏观指标")
        macro_items = {
            "联邦基金利率": ("FED_RATE", "%"),
            "美元指数": ("DXY", ""),
            "VIX 恐慌指数": ("VIX", ""),
            "美国CPI": ("US_CPI", ""),
            "全球EPU": ("GLOBAL_EPU", ""),
            "美国10Y国债": ("US10Y", "%"),
        }
        for label, (col, unit) in macro_items.items():
            if col in feat.columns:
                series = feat[col].dropna()
                val = series.iloc[-1]
                # 找最近一次有变化的值做delta
                changed = series[series != val]
                prev = changed.iloc[-1] if len(changed) > 0 else val
                delta = round(val - prev, 3)
                delta_str = "{:+.3f}".format(delta) if delta != 0 else "—"
                st.metric(
                    label,
                    "{:.2f}{}".format(val, unit),
                    delta_str
                )

    st.divider()

    # 最新新闻
    if len(news) > 0:
        st.subheader("最新市场新闻")
        latest_news = news.sort_values("date", ascending=False).head(8)
        for _, row in latest_news.iterrows():
            url = row.get("url", "")
            title = str(row.get("title", ""))
            src = str(row.get("source", ""))
            date = str(row.get("date", ""))
            if url:
                st.markdown("- **​" + date + "​** &nbsp; [" + title + "](" + url + ")  `" + src + "`")
            else:
                st.markdown("- **​" + date + "​** &nbsp;" + title + "  `" + src + "`")

# ══════════════════════════════════════════════════════════════════════════
# 页面三：风险预测
# ══════════════════════════════════════════════════════════════════════════
elif page == "风险预测":
    st.title("风险预测")
    st.caption("未来10日 WTI 原油价格涨跌幅风险区间预测")

    last_low = pred_df["pred_enhanced_low"].iloc[-1]
    last_mid = pred_df["pred_enhanced_mid"].iloc[-1]
    last_high = pred_df["pred_enhanced_high"].iloc[-1]
    risk_label, risk_color, _ = get_risk_level(last_low, last_mid, last_high)

    # ── 黑天鹅期间：显示 AI 情景分析，隐藏统计模型预测 ──────────────────
    # 手动覆盖开关（侧边栏）
    with st.sidebar:
        st.divider()
        manual_override = st.toggle(
            "手动关闭黑天鹅模式",
            value=False,
            help="当您判断极端事件已解除时，可手动关闭黑天鹅预警"
        )
        if manual_override:
            is_black_swan = False

    if is_black_swan:
        st.error(
            "当前处于极端地缘政治事件期间（霍尔木兹封锁），"
            "统计模型预测已暂停。以下为 AI 情景分析结果。"
        )

        rt_price, rt_date, rt_live = get_realtime_price()

        # 情景价格区间可视化
        scenarios = {
            "缓解情景\n（外交解决）": {"low": 85, "high": 95, "color": "#2ecc71"},
            "基准情景\n（封锁持续）": {"low": 95, "high": 110, "color": "#f1c40f"},
            "升级情景\n（冲突扩大）": {"low": 110, "high": 120, "color": "#e74c3c"},
        }

        fig_bs = go.Figure()

        # 当前价格基准线
        fig_bs.add_hline(
            y=rt_price,
            line_dash="dash", line_color="white", line_width=2,
            annotation_text=f"当前价格 ${rt_price}",
            annotation_font_color="white",
        )

        for i, (label, s) in enumerate(scenarios.items()):
            fig_bs.add_trace(go.Bar(
                x=[label],
                y=[s["high"] - s["low"]],
                base=[s["low"]],
                marker_color=s["color"],
                marker_opacity=0.75,
                name=label,
                text=f"${s['low']} ~ ${s['high']}",
                textposition="inside",
                textfont=dict(color="white", size=13),
                width=0.4,
            ))

        fig_bs.update_layout(
            height=420,
            title="未来1-2周 WTI 价格情景区间（AI 生成）",
            margin=dict(l=0, r=0, t=40, b=0),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="white"),
            yaxis=dict(
                gridcolor="#2d3748",
                title="WTI 美元/桶",
                range=[75, 130],
            ),
            xaxis=dict(gridcolor="#2d3748"),
            showlegend=False,
            barmode="overlay",
        )
        st.plotly_chart(fig_bs, use_container_width=True)

        # DeepSeek 分析文字
        report_path = os.path.join(ROOT_DIR, "data", "raw", "black_swan_report.json")
        if os.path.exists(report_path):
            import json

            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            analysis = report.get("analysis", {}).get("analysis", "")
            if analysis:
                with st.expander("查看完整 AI 情景分析", expanded=False):
                    st.markdown(analysis)
                    st.caption("生成时间：" +
                               report.get("analysis", {}).get("generated_at", ""))

        st.divider()
        st.subheader("封锁期间油价走势")
        # 显示封锁前后的实际价格走势
        recent = pred_df["2026-01-01":]
        fig_wti = go.Figure()
        fig_wti.add_trace(go.Scatter(
            x=recent.index, y=recent["WTI"],
            name="WTI 实际价格",
            line=dict(color="#3498db", width=2)
        ))
        import pandas as pd

        fig_wti.add_vline(
            x=pd.Timestamp("2026-03-02").timestamp() * 1000,
            line_dash="dash", line_color="#e74c3c",
            annotation_text="霍尔木兹封锁",
            annotation_font_color="#e74c3c",
        )

        fig_wti.update_layout(
            height=320,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font=dict(color="white"),
            xaxis=dict(gridcolor="#2d3748"),
            yaxis=dict(gridcolor="#2d3748", title="美元/桶"),
        )
        st.plotly_chart(fig_wti, use_container_width=True)
        st.divider()
        st.subheader("极端事件深度分析报告")
        st.caption("由 DeepSeek 生成 · 包含情景价格预测、行业冲击信号、对冲操作建议")

        col_btn1, col_btn2 = st.columns([1, 3])
        with col_btn1:
            gen_bs_report = st.button(
                "生成极端事件报告",
                type="primary",
                use_container_width=True
            )
        with col_btn2:
            report_path = os.path.join(ROOT_DIR, "data", "raw", "latest_report.json")
            if os.path.exists(report_path):
                show_bs_cache = st.button("查看上次报告", use_container_width=True)
            else:
                show_bs_cache = False

        if gen_bs_report:
            with st.spinner("正在生成极端事件深度分析，约需15-25秒..."):
                try:
                    from report_generator import generate_report
                    recent_news_list = news.to_dict("records") if len(news) > 0 else []
                    result = generate_report(
                        current_price      = rt_price,
                        pred_low           = last_low,
                        pred_mid           = last_mid,
                        pred_high          = last_high,
                        feature_importance = importance,
                        is_black_swan      = True,
                        bs_signals         = bs_signals,
                        recent_news        = recent_news_list,
                    )
                    if result["status"] == "ok":
                        st.success("报告生成成功 ｜ " + result["generated_at"])
                        st.markdown(result["report"])
                        st.download_button(
                            label     = "下载报告（TXT）",
                            data      = result["report"].encode("utf-8"),
                            file_name = "OilSense_BlackSwan_{}.txt".format(
                                datetime.now().strftime("%Y%m%d_%H%M")
                            ),
                            mime      = "text/plain",
                        )
                    else:
                        st.error("生成失败：" + result.get("error", ""))
                except Exception as e:
                    st.error("调用失败：" + str(e))

        elif os.path.exists(report_path) and show_bs_cache:
            import json
            with open(report_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            st.markdown(cached.get("report", ""))


    else:
        # ── 正常市场：显示统计模型预测 ──────────────────────────────────
        col1, col2, col3 = st.columns(3)
        with col1:
          st.markdown("#### 悲观情景（10%分位）")
          st.markdown("<h2 style='color:#e74c3c'>" +
                    str(round(last_low * 100, 2)) + "%</h2>", unsafe_allow_html=True)
          st.caption("有10%概率跌幅超过此值")
        with col2:
          st.markdown("#### 基准预测（中位数）")
          color = "#e74c3c" if last_mid < 0 else "#2ecc71"
          st.markdown("<h2 style='color:" + color + "'>" +
                    str(round(last_mid * 100, 2)) + "%</h2>", unsafe_allow_html=True)
          st.caption("模型预测最可能的涨跌幅")
        with col3:
          st.markdown("#### 乐观情景（90%分位）")
          st.markdown("<h2 style='color:#2ecc71'>" +
                    str(round(last_high * 100, 2)) + "%</h2>", unsafe_allow_html=True)
          st.caption("有10%概率涨幅超过此值")
    st.divider()
    st.subheader("企业银行分析报告")
    st.caption("由 DeepSeek 生成 · 面向航空、化工、能源贸易商等企业客户")

    # 检查是否有缓存报告
    report_path = os.path.join(ROOT_DIR, "data", "raw", "latest_report.json")
    has_cache = os.path.exists(report_path)
    if has_cache:
        import json

        with open(report_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        st.caption(
            "上次生成：" + cached.get("generated_at", "") +
            " ｜ 模式：" + cached.get("mode", "") +
            " ｜ 模型：" + cached.get("model", "")
        )

    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        gen_report = st.button(
            "生成最新报告",
            type="primary",
            use_container_width=True,
            key="btn_gen_normal"
        )
    with col_btn2:
        if has_cache:
            show_cache = st.button(
                "查看上次报告",
                use_container_width=True,
                key="btn_cache_normal"
            )
        else:
            show_cache = False

    if gen_report:
        with st.spinner("正在调用 DeepSeek 生成分析报告，约需10-20秒..."):
            try:
                from report_generator import generate_report

                recent_news_list = news.to_dict("records") if len(news) > 0 else []
                result = generate_report(
                    current_price=rt_price,
                    pred_low=last_low,
                    pred_mid=last_mid,
                    pred_high=last_high,
                    feature_importance=importance,
                    is_black_swan=is_black_swan,
                    bs_signals=bs_signals if is_black_swan else None,
                    recent_news=recent_news_list,
                )
                if result["status"] == "ok":
                    st.success(
                        "报告生成成功 ｜ 模式：" + result["mode"] +
                        " ｜ 生成时间：" + result["generated_at"]
                    )
                    st.markdown(result["report"])
                    # 提供下载
                    st.download_button(
                        label="下载报告（TXT）",
                        data=result["report"].encode("utf-8"),
                        file_name="OilSense_Report_{}.txt".format(
                            datetime.now().strftime("%Y%m%d_%H%M")
                        ),
                        mime="text/plain",
                        key="btn_download_normal"
                    )

                else:
                    st.error("报告生成失败：" + result.get("error", "未知错误"))
            except Exception as e:
                st.error("调用失败：" + str(e))

    elif has_cache and show_cache:
        st.download_button(
            label="下载报告（TXT）",
            data=cached.get("report", "").encode("utf-8"),
            file_name="OilSense_Report_cached.txt",
            mime="text/plain",
            key="btn_download_cache_normal"
        )

    st.divider()

    st.subheader("近60日预测区间走势")
    recent = pred_df.tail(60)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(recent.index) + list(recent.index[::-1]),
        y=list(recent["pred_enhanced_high"]) + list(recent["pred_enhanced_low"][::-1]),
        fill="toself", fillcolor="rgba(52,152,219,0.2)",
        line=dict(color="rgba(0,0,0,0)"), name="风险区间（10%~90%）"
    ))
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent["pred_enhanced_mid"],
        name="预测中位数（Enhanced）",
        line=dict(color="#3498db", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent["pred_baseline_mid"],
        name="预测中位数（Baseline）",
        line=dict(color="#95a5a6", width=1.5, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent["target"],
        name="实际涨跌幅",
        line=dict(color="#e74c3c", width=1), opacity=0.7
    ))
    fig.add_hline(y=0, line_dash="solid", line_color="white", line_width=0.8)
    fig.update_layout(
        height=420, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"),
        xaxis=dict(gridcolor="#2d3748"),
        yaxis=dict(gridcolor="#2d3748", title="10日涨跌幅"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # 模型对比
    st.subheader("模型性能对比（样本外测试集）")
    test_start = pred_df.index[int(len(pred_df) * 0.8)]
    test_df = pred_df.loc[test_start:]
    actual = test_df["target"]

    results = {}
    for version in ["enhanced", "baseline"]:
        mid = test_df["pred_" + version + "_mid"]
        valid = actual.notna() & mid.notna()
        acc = np.mean(np.sign(mid[valid]) == np.sign(actual[valid]))
        results[version] = round(acc * 100, 2)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Enhanced 方向准确率", str(results["enhanced"]) + "%",
                  str(round(results["enhanced"] - 50, 2)) + "% vs 随机基准")
    with col2:
        st.metric("Baseline 方向准确率", str(results["baseline"]) + "%",
                  str(round(results["baseline"] - 50, 2)) + "% vs 随机基准")

    with col3:
        st.metric("测试集样本数", str(len(test_df)) + " 条")

    st.divider()

    # 特征重要性
    st.subheader("模型特征重要性")
    top_imp = importance.head(12)
    sentiment_cols = ["sentiment_score", "geopolitics_flag", "policy_flag",
                      "news_count", "gdelt_goldstein", "gdelt_tone",
                      "gdelt_conflict_cnt", "gdelt_coop_cnt",
                      "gdelt_conflict_intensity", "gdelt_mentions",
                      "gdelt_goldstein_chg", "gdelt_conflict_ma5", "gdelt_tone_chg"]
    colors = ["#e74c3c" if f in sentiment_cols else "#3498db"
              for f in top_imp["feature"]]

    fig2 = go.Figure(go.Bar(
        x=top_imp["importance"][::-1].values,
        y=top_imp["feature"][::-1].values,
        orientation="h",
        marker_color=colors[::-1],
        text=[str(round(v, 4)) for v in top_imp["importance"][::-1].values],
        textposition="outside",
    ))
    fig2.update_layout(
        height=420, margin=dict(l=0, r=60, t=10, b=0),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"),
        xaxis=dict(gridcolor="#2d3748", title="重要性得分"),
        yaxis=dict(gridcolor="#2d3748"),
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("🔴 红色 = LLM情感/GDELT地缘政治因子　🔵 蓝色 = 传统量化因子")

# ══════════════════════════════════════════════════════════════════════════
# 页面四：历史回测
# ══════════════════════════════════════════════════════════════════════════
elif page == "历史回测":
    st.title("历史回测")
    st.caption("2020年至今 WTI 油价走势与模型预测表现")

    CRISIS_EVENTS = [
        {"date": "2020-03-09", "label": "新冠+油价战争", "color": "#e74c3c"},
        {"date": "2022-02-24", "label": "俄乌冲突爆发",  "color": "#e67e22"},
        {"date": "2023-10-07", "label": "以哈冲突爆发",  "color": "#9b59b6"},
        {"date": "2025-01-20", "label": "特朗普就职",    "color": "#2980b9"},
    ]

    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始日期", value=pd.Timestamp("2020-01-01"))
    with col2:
        end_date = st.date_input("结束日期", value=pred_df.index[-1])

    filtered = pred_df.loc[str(start_date):str(end_date)]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.05)

    fig.add_trace(go.Scatter(
        x=filtered.index, y=filtered["WTI"],
        name="WTI 油价", line=dict(color="#3498db", width=1.5)
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=list(filtered.index) + list(filtered.index[::-1]),
        y=list(filtered["pred_enhanced_high"]) + list(filtered["pred_enhanced_low"][::-1]),
        fill="toself", fillcolor="rgba(52,152,219,0.2)",
        line=dict(color="rgba(0,0,0,0)"), name="风险区间"
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=filtered.index, y=filtered["pred_enhanced_mid"],
        name="预测中位数（Enhanced）",
        line=dict(color="#3498db", width=1.5)
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=filtered.index, y=filtered["pred_baseline_mid"],
        name="预测中位数（Baseline）",
        line=dict(color="#95a5a6", width=1, dash="dash")
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=filtered.index, y=filtered["target"],
        name="实际涨跌幅",
        line=dict(color="#e74c3c", width=0.8), opacity=0.6
    ), row=2, col=1)

    # 危机事件标注
    for event in CRISIS_EVENTS:
        edate = pd.Timestamp(event["date"])
        if pd.Timestamp(start_date) <= edate <= pd.Timestamp(end_date):
            fig.add_vline(
                x=edate.timestamp() * 1000,
                line_dash="dash",
                line_color=event["color"], line_width=1.5,
                annotation_text=event["label"],
                annotation_position="top",
                annotation_font_size=10,
                annotation_font_color=event["color"]
            )

    fig.update_layout(
        height=620, margin=dict(l=0, r=0, t=30, b=0),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
    )
    fig.update_yaxes(gridcolor="#2d3748")
    fig.update_xaxes(gridcolor="#2d3748")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # 危机窗口详细分析
    st.subheader("危机事件窗口分析")
    selected_event = st.selectbox(
        "选择事件",
        ["俄乌冲突爆发(2022-02-24)",
         "以哈冲突爆发(2023-10-07)",
         "特朗普就职(2025-01-20)"]
    )

    event_map = {
        "俄乌冲突爆发(2022-02-24)": ("2022-02-24", "#e67e22"),
        "以哈冲突爆发(2023-10-07)": ("2023-10-07", "#9b59b6"),
        "特朗普就职(2025-01-20)":   ("2025-01-20", "#2980b9"),
    }
    ev_date, ev_color = event_map[selected_event]
    ev_ts    = pd.Timestamp(ev_date)
    ev_start = ev_ts - pd.Timedelta(days=30)
    ev_end   = ev_ts + pd.Timedelta(days=45)
    ev_data  = pred_df.loc[ev_start:ev_end]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=list(ev_data.index) + list(ev_data.index[::-1]),
        y=list(ev_data["pred_enhanced_high"]) + list(ev_data["pred_enhanced_low"][::-1]),
        fill="toself", fillcolor="rgba(52,152,219,0.25)",
        line=dict(color="rgba(0,0,0,0)"), name="风险区间"
    ))
    fig3.add_trace(go.Scatter(
        x=ev_data.index, y=ev_data["pred_enhanced_mid"],
        name="预测中位数", line=dict(color="#3498db", width=2)
    ))
    fig3.add_trace(go.Scatter(
        x=ev_data.index, y=ev_data["target"],
        name="实际涨跌幅", line=dict(color="#e74c3c", width=1.5)
    ))
    fig3.add_vline(
        x=pd.Timestamp(ev_date).timestamp() * 1000,
        line_dash="dash",
        line_color=ev_color, line_width=2
    )
    fig3.add_hline(y=0, line_color="white", line_width=0.8)
    fig3.update_layout(
        height=350, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"),
        xaxis=dict(gridcolor="#2d3748"),
        yaxis=dict(gridcolor="#2d3748", title="涨跌幅"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    bgcolor="rgba(0,0,0,0)")
    )
    st.plotly_chart(fig3, use_container_width=True)
