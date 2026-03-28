import os
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components
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

# ── 全局CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }

/* ── 全局背景 ── */
[data-testid="stAppViewContainer"] {
    background: #080c14 !important;
}
[data-testid="stMain"] {
    background: transparent !important;
}

/* ════════════════════════════════
   侧边栏
════════════════════════════════ */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #07090f 0%, #0a0d18 60%, #07090f 100%) !important;
    border-right: 1px solid rgba(180,120,40,0.18) !important;
    box-shadow: 4px 0 24px rgba(0,0,0,0.6) !important;
}
[data-testid="stSidebar"] section[data-testid="stSidebarContent"] {
    padding: 2rem 1.2rem 1.5rem !important;
}

/* Logo */
[data-testid="stSidebar"] h1 {
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #c8922a 0%, #e8c97a 50%, #c8922a 100%) !important;
    background-size: 200% auto !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    animation: shimmer 4s linear infinite !important;
    letter-spacing: -0.02em !important;
    margin-bottom: 0.2rem !important;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    font-size: 0.72rem !important;
    color: #4a3f28 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    margin-bottom: 1.8rem !important;
}

/* 导航分组标题 */
[data-testid="stSidebar"] h3 {
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    color: #4a3f28 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    margin: 1.5rem 0 0.6rem 0.4rem !important;
    border-bottom: none !important;
}

/* 隐藏折叠按钮文字 */
[data-testid="stSidebarCollapseButton"] span,
[data-testid="collapsedControl"] span {
    display: none !important;
}
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="collapsedControl"] svg {
    color: rgba(180,120,40,0.4) !important;
}

[data-testid="stSidebar"] .stRadio > div {
    gap: 3px !important;
}
/* 覆盖radio选中圆点颜色 */
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] div[class*="checked"],
[data-testid="stSidebar"] [data-baseweb="radio"] > div:first-child {
    background-color: #c8922a !important;
    border-color: #e8c97a !important;
}
[data-testid="stSidebar"] [data-baseweb="radio"] {
    accent-color: #c8922a !important;
}
[data-testid="stSidebar"] input[type="radio"]:checked + div,
[data-testid="stSidebar"] input[type="radio"] + div {
    border-color: #c8922a !important;
}
[data-testid="stSidebar"] input[type="radio"]:checked + div::before {
    background: #c8922a !important;
}

/* 导航项 */
[data-testid="stSidebar"] .stRadio label {
    display: flex !important;
    align-items: center !important;
    position: relative !important;
    padding: 9px 14px 9px 16px !important;
    border-radius: 8px !important;
    font-size: 13.5px !important;
    font-weight: 500 !important;
    color: #7a6a4a !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    border: 1px solid transparent !important;
    margin: 1px 0 !important;
    letter-spacing: 0.01em !important;
}

/* 左侧竖线指示器 */
[data-testid="stSidebar"] .stRadio label::before {
    content: '' !important;
    position: absolute !important;
    left: 0px !important;
    top: 18% !important;
    height: 64% !important;
    width: 2.5px !important;
    border-radius: 2px !important;
    background: transparent !important;
    transition: all 0.2s ease !important;
}

/* hover */
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(180,120,40,0.08) !important;
    color: #c9a96e !important;
    border-color: rgba(180,120,40,0.18) !important;
}

/* 选中 */
[data-testid="stSidebar"] .stRadio label[data-checked="true"] {
    background: linear-gradient(90deg,
        rgba(180,120,40,0.2) 0%,
        rgba(180,120,40,0.04) 100%) !important;
    color: #e8c97a !important;
    border-color: rgba(180,120,40,0.4) !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] .stRadio label[data-checked="true"]::before {
    background: linear-gradient(180deg, #c8922a, #e8c97a) !important;
    box-shadow: 0 0 8px rgba(232,201,122,0.6) !important;
}

[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
    margin: 0 !important;
    font-size: 13.5px !important;
}

/* 侧边栏指标卡片 */
[data-testid="stSidebar"] [data-testid="stMetric"] {
    background: rgba(180,120,40,0.06) !important;
    border: 1px solid rgba(180,120,40,0.15) !important;
    border-radius: 10px !important;
    padding: 12px 14px !important;
    margin-bottom: 8px !important;
    backdrop-filter: none !important;
}
[data-testid="stSidebar"] [data-testid="stMetric"]:hover {
    border-color: rgba(180,120,40,0.35) !important;
    background: rgba(180,120,40,0.1) !important;
    transform: none !important;
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    font-size: 1.25rem !important;
    color: #e8c97a !important;
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    font-size: 0.72rem !important;
    color: #5a4a2a !important;
}

/* 侧边栏分割线 */
[data-testid="stSidebar"] hr {
    border-color: rgba(180,120,40,0.12) !important;
    margin: 1.2rem 0 !important;
}

/* ════════════════════════════════
   主内容区
════════════════════════════════ */

/* 指标卡片 */
[data-testid="stMetric"] {
    background: rgba(16,20,30,0.75) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(180,120,40,0.18) !important;
    border-radius: 14px !important;
    padding: 18px 20px !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35) !important;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(180,120,40,0.45) !important;
    box-shadow: 0 6px 32px rgba(180,120,40,0.12) !important;
    transform: translateY(-2px) !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.65rem !important;
    font-weight: 700 !important;
    color: #e8c97a !important;
    letter-spacing: -0.02em !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: #7a6a4a !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* 标题 */
h1 {
    font-size: 2rem !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #c8922a 0%, #e8c97a 50%, #c8922a 100%) !important;
    background-size: 200% auto !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    animation: shimmer 4s linear infinite !important;
    letter-spacing: -0.03em !important;
}
h2 {
    font-size: 1.25rem !important;
    font-weight: 700 !important;
    color: #c9a96e !important;
    border-bottom: 1px solid rgba(180,120,40,0.18) !important;
    padding-bottom: 8px !important;
    margin-top: 1.5rem !important;
}
h3 {
    font-size: 1rem !important;
    font-weight: 600 !important;
    color: #a89060 !important;
}
@keyframes shimmer {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
}

/* 按钮 */
.stButton > button {
    background: rgba(16,20,30,0.8) !important;
    border: 1px solid rgba(180,120,40,0.35) !important;
    border-radius: 8px !important;
    color: #c9a96e !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: rgba(180,120,40,0.15) !important;
    border-color: rgba(180,120,40,0.6) !important;
    color: #e8c97a !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(180,120,40,0.2) !important;
}
.stButton > button:active {
    background: rgba(180,120,40,0.25) !important;
    border-color: #e8c97a !important;
    color: #e8c97a !important;
    transform: translateY(0px) !important;
    box-shadow: inset 0 2px 8px rgba(0,0,0,0.3),
                0 0 12px rgba(232,201,122,0.3) !important;
}
.stButton > button:focus:not(:active) {
    border-color: rgba(180,120,40,0.4) !important;
    box-shadow: 0 0 0 2px rgba(180,120,40,0.2) !important;
}

/* 隐藏按钮图标文字泄露 */
button span.material-symbols-rounded {
    font-size: 0 !important;
    color: transparent !important;
}

/* selectbox / date_input */
[data-testid="stSelectbox"] > div > div,
[data-testid="stDateInput"] input {
    background: rgba(16,20,30,0.75) !important;
    border: 1px solid rgba(180,120,40,0.25) !important;
    border-radius: 8px !important;
    color: #c9a96e !important;
}

/* 分割线 */
hr { border-color: rgba(180,120,40,0.15) !important; }

/* caption */
[data-testid="stCaptionContainer"] p {
    color: #6a5a3a !important;
    font-size: 0.78rem !important;
}

/* ════════════════════════════════
   粒子背景
════════════════════════════════ */
.oil-drop {
    position: fixed;
    top: -20px;
    border-radius: 50% 50% 50% 50% / 60% 60% 40% 40%;
    background: radial-gradient(ellipse at 40% 35%,
        rgba(220,160,50,0.55) 0%,
        rgba(180,100,20,0.35) 40%,
        rgba(100,60,10,0.1) 100%);
    animation: dropFall linear infinite;
    pointer-events: none;
    z-index: 0;
}
@keyframes dropFall {
    0%   { transform: translateY(-20px) scaleX(0.85); opacity: 0; }
    5%   { opacity: 1; }
    90%  { opacity: 0.6; }
    100% { transform: translateY(100vh) scaleX(0.85); opacity: 0; }
}

/* ════════════════════════════════
   移动端
════════════════════════════════ */
@media (max-width: 768px) {
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.1rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.2rem !important; }
    [data-testid="stMetric"] { padding: 12px 14px !important; }
}
</style>
""", unsafe_allow_html=True)

# ── 粒子节点 ──────────────────────────────────────────────────────────────
st.markdown("""
<div aria-hidden="true">
<div class="oil-drop" style="left:5%;  width:5px; height:9px;  animation-duration:6.2s; animation-delay:0s;"></div>
<div class="oil-drop" style="left:10%; width:3px; height:6px;  animation-duration:8.1s; animation-delay:1.2s;"></div>
<div class="oil-drop" style="left:16%; width:6px; height:11px; animation-duration:7.4s; animation-delay:0.5s;"></div>
<div class="oil-drop" style="left:22%; width:4px; height:7px;  animation-duration:9.0s; animation-delay:2.1s;"></div>
<div class="oil-drop" style="left:28%; width:5px; height:9px;  animation-duration:6.8s; animation-delay:3.3s;"></div>
<div class="oil-drop" style="left:34%; width:3px; height:5px;  animation-duration:10.2s;animation-delay:0.8s;"></div>
<div class="oil-drop" style="left:40%; width:6px; height:10px; animation-duration:7.1s; animation-delay:1.9s;"></div>
<div class="oil-drop" style="left:46%; width:4px; height:8px;  animation-duration:8.5s; animation-delay:4.0s;"></div>
<div class="oil-drop" style="left:52%; width:5px; height:9px;  animation-duration:6.5s; animation-delay:2.7s;"></div>
<div class="oil-drop" style="left:58%; width:3px; height:6px;  animation-duration:9.3s; animation-delay:0.3s;"></div>
<div class="oil-drop" style="left:64%; width:6px; height:11px; animation-duration:7.8s; animation-delay:3.6s;"></div>
<div class="oil-drop" style="left:70%; width:4px; height:7px;  animation-duration:8.9s; animation-delay:1.5s;"></div>
<div class="oil-drop" style="left:76%; width:5px; height:9px;  animation-duration:6.3s; animation-delay:5.1s;"></div>
<div class="oil-drop" style="left:82%; width:3px; height:5px;  animation-duration:10.5s;animation-delay:2.4s;"></div>
<div class="oil-drop" style="left:88%; width:6px; height:10px; animation-duration:7.6s; animation-delay:0.9s;"></div>
<div class="oil-drop" style="left:93%; width:4px; height:8px;  animation-duration:8.2s; animation-delay:3.8s;"></div>
<div class="oil-drop" style="left:97%; width:5px; height:9px;  animation-duration:6.9s; animation-delay:1.1s;"></div>
</div>
""", unsafe_allow_html=True)

# ── 产油国数据 ────────────────────────────────────────────────────────────
OIL_COUNTRIES = {
    "美国":      {"lat": 38.0, "lon": -97.0, "code": "US", "prod": 12.9, "share": 13.2},
    "俄罗斯":    {"lat": 61.0, "lon":  90.0, "code": "RS", "prod": 10.1, "share": 10.3},
    "沙特阿拉伯":{"lat": 24.0, "lon":  45.0, "code": "SA", "prod":  9.6, "share":  9.8},
    "伊拉克":    {"lat": 33.0, "lon":  44.0, "code": "IZ", "prod":  4.2, "share":  4.3},
    "伊朗":      {"lat": 32.0, "lon":  53.0, "code": "IR", "prod":  3.4, "share":  3.5},
    "阿联酋":    {"lat": 24.0, "lon":  54.0, "code": "AE", "prod":  3.2, "share":  3.3},
    "科威特":    {"lat": 29.0, "lon":  47.0, "code": "KU", "prod":  2.7, "share":  2.8},
    "挪威":      {"lat": 60.0, "lon":  10.0, "code": "NO", "prod":  1.8, "share":  1.8},
    "哈萨克斯坦":{"lat": 48.0, "lon":  68.0, "code": "KZ", "prod":  1.8, "share":  1.8},
    "尼日利亚":  {"lat":  9.0, "lon":   8.0, "code": "NG", "prod":  1.5, "share":  1.5},
    "利比亚":    {"lat": 27.0, "lon":  17.0, "code": "LY", "prod":  1.2, "share":  1.2},
    "委内瑞拉":  {"lat":  8.0, "lon": -66.0, "code": "VE", "prod":  0.9, "share":  0.9},
    "阿尔及利亚":{"lat": 28.0, "lon":   2.0, "code": "AL", "prod":  0.9, "share":  0.9},
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


@st.cache_data(ttl=86400)
def auto_update_data():
    try:
        from update_daily import (
            update_oil_prices,
            update_macro_data,
            update_news_rss,      # RSS很快
        )
        update_oil_prices()       # FRED拉增量，通常0条，很快
        update_macro_data()       # 同上
        update_news_rss()         # RSS抓取，约10秒
        return datetime.now().strftime("%Y-%m-%d %H:%M")
    except Exception as e:
        return f"更新失败: {e}"

with st.spinner("正在检查数据更新..."):
    auto_update_data()

@st.cache_data(ttl=84600)
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
@st.cache_data(ttl=3600)
def get_chokepoint_status():
    """获取各咽喉点当前状态，AIS有覆盖时优先用AIS数据"""
    import json

    df = load_portwatch()
    if df is None:
        return {}

    # 加载AIS快照
    ais_data = {}
    ais_path = os.path.join(ROOT_DIR, "data", "raw", "ais_snapshot.json")
    if os.path.exists(ais_path):
        with open(ais_path, "r", encoding="utf-8") as f:
            ais_data = json.load(f)

    cp_map = {
        "cp6" : {"name": "霍尔木兹", "importance": "全球20%石油",  "ais_name": "霍尔木兹海峡"},
        "cp4" : {"name": "曼德海峡", "importance": "红海通道",     "ais_name": "曼德海峡"},
        "cp1" : {"name": "苏伊士运河","importance": "欧亚航线",    "ais_name": "苏伊士运河"},
        "cp5" : {"name": "马六甲",   "importance": "亚洲石油",     "ais_name": "马六甲海峡"},
        "cp3" : {"name": "博斯普鲁斯","importance": "俄油出口",    "ais_name": "博斯普鲁斯海峡"},
        "cp11": {"name": "台湾海峡", "importance": "亚太航运",     "ais_name": None},
        "cp7" : {"name": "好望角",   "importance": "绕行路线",     "ais_name": None},
    }

    status = {}
    for cp, info in cp_map.items():
        col = f"{cp}_tanker"
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) < 2:
            continue

        # PortWatch基础数据
        pw_latest   = int(series.iloc[-1])
        avg_90d     = series.tail(90).mean()
        pw_ratio    = pw_latest / avg_90d if avg_90d > 0 else 1.0
        last_date   = str(series.index[-1].date())

        def _risk_color(ratio):
            if ratio < 0.3:   return "极高风险", "#e74c3c"
            elif ratio < 0.6: return "高风险",   "#e67e22"
            elif ratio < 0.85:return "偏低",     "#f1c40f"
            else:             return "正常",     "#2ecc71"

        pw_risk, pw_color = _risk_color(pw_ratio)

        # AIS数据融合
        ais_entry    = ais_data.get(info["ais_name"]) if info["ais_name"] else None
        has_ais      = ais_entry and ais_entry.get("ais_coverage", False)

        if has_ais:
            ais_count    = ais_entry["count"]
            ais_normal   = ais_entry["normal_count"]
            ais_ratio    = ais_count / ais_normal if ais_normal > 0 else 1.0
            ais_risk, ais_color = _risk_color(ais_ratio)
            ais_ratio_pct = round(ais_ratio * 100, 1)
            ais_timestamp = ais_entry.get("timestamp", "")

            # 取两者中风险更高的作为主显示
            risk_priority = {"极高风险": 4, "高风险": 3, "偏低": 2, "正常": 1}
            if risk_priority.get(ais_risk, 0) >= risk_priority.get(pw_risk, 0):
                main_risk  = ais_risk
                main_color = ais_color
            else:
                main_risk  = pw_risk
                main_color = pw_color
        else:
            ais_count     = None
            ais_ratio_pct = None
            ais_risk      = None
            ais_color     = None
            ais_timestamp = None
            main_risk     = pw_risk
            main_color    = pw_color

        status[cp] = {
            "name"         : info["name"],
            "importance"   : info["importance"],
            # PortWatch
            "latest"       : pw_latest,
            "avg_90d"      : round(avg_90d, 1),
            "ratio"        : round(pw_ratio * 100, 1),
            "risk"         : pw_risk,
            "color"        : pw_color,
            "last_date"    : last_date,
            # AIS
            "has_ais"      : has_ais,
            "ais_count"    : ais_count,
            "ais_ratio"    : ais_ratio_pct,
            "ais_risk"     : ais_risk,
            "ais_color"    : ais_color,
            "ais_timestamp": ais_timestamp,
            # 综合
            "main_risk"    : main_risk,
            "main_color"   : main_color,
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

def get_country_news(country_name, n=5):
    """从情感分析结果里提取该国相关新闻，带情感标签"""
    # 优先读情感分析结果
    sentiment_path = os.path.join(ROOT_DIR, "data", "processed", "news_sentiment_detail.csv")
    news_path      = os.path.join(ROOT_DIR, "data", "raw", "news_data.csv")

    if os.path.exists(sentiment_path):
        df = pd.read_csv(sentiment_path)
    elif os.path.exists(news_path):
        df = pd.read_csv(news_path)
    else:
        return []

    if len(df) == 0:
        return []

    keywords = {
        "伊朗"    : ["Iran", "Iranian"],
        "沙特阿拉伯": ["Saudi", "Riyadh", "Aramco"],
        "俄罗斯"  : ["Russia", "Russian", "Moscow", "Kremlin"],
        "伊拉克"  : ["Iraq", "Iraqi", "Baghdad"],
        "美国"    : ["US", "America", "Washington", "Biden", "Trump"],
        "阿联酋"  : ["UAE", "Dubai", "Abu Dhabi"],
        "挪威"    : ["Norway", "Norwegian", "Equinor"],
        "委内瑞拉": ["Venezuela", "Maduro"],
        "利比亚"  : ["Libya", "Libyan"],
        "尼日利亚": ["Nigeria", "Nigerian"],
        "科威特"  : ["Kuwait"],
        "哈萨克斯坦": ["Kazakhstan"],
        "阿尔及利亚": ["Algeria"],
    }

    kws  = keywords.get(country_name, [country_name])
    mask = df["title"].str.contains("|".join(kws), case=False, na=False)
    filtered = df[mask].sort_values("date", ascending=False).head(n)

    results = []
    for _, row in filtered.iterrows():
        results.append({
            "date"           : str(row.get("date", ""))[:10],
            "title"          : str(row.get("title", "")),
            "source"         : str(row.get("source", "")),
            "url"            : str(row.get("url", "")),
            "sentiment"      : str(row.get("sentiment", "neutral")),
            "score"          : float(row.get("score", 0)) if "score" in row else 0.0,
            "confidence"     : float(row.get("confidence", 0)) if "confidence" in row else 0.0,
            "event_type"     : str(row.get("event_type", "other")) if "event_type" in row else "other",
            "impact_duration": str(row.get("impact_duration", "short")) if "impact_duration" in row else "short",
            "reason"         : str(row.get("reason", "")) if "reason" in row else "",
        })
    return results


with st.sidebar:
    st.title("🛢 OilSense")
    st.caption("原油风险智能预警系统")
    st.divider()
    st.markdown("### 导航")
    page = st.radio(
        "导航",
        ["全球能源地图", "市场概览", "风险预测", "历史回测"],
        label_visibility="visible"
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
    if st.button("查看AI情景分析详情", key="bs_report_btn", use_container_width=True):
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
                    import black_swan
                    black_swan.ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

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


    st.plotly_chart(fig, use_container_width=True,key="chart_pred")
    # ── 咽喉点航运状态面板 ────────────────────────────────────────────────
    st.divider()
    st.subheader("全球关键咽喉点实时状态")
    st.caption("PortWatch 历史趋势（90日均值对比）· AIS实时信号融合")

    cp_status = get_chokepoint_status()

    if cp_status:
        cols = st.columns(len(cp_status))
        for i, (cp, s) in enumerate(cp_status.items()):
            with cols[i]:
                if s["has_ais"] and s["ais_count"] is not None:
                    ais_badge = (
                        "<div style=\"background:rgba(52,152,219,0.15);"
                        "border:1px solid #3498db;border-radius:4px;"
                        "padding:2px 6px;font-size:9px;color:#3498db;"
                        "font-weight:600;margin-bottom:4px;display:inline-block;\">"
                        f"● AIS实时 · {s['ais_count']}艘 · {s['ais_ratio']}%"
                        "</div>"
                    )
                else:
                    ais_badge = (
                        "<div style=\"background:rgba(100,100,100,0.1);"
                        "border:1px solid #444;border-radius:4px;"
                        "padding:2px 6px;font-size:9px;color:#555;"
                        "font-weight:600;margin-bottom:4px;display:inline-block;\">"
                        "PortWatch"
                        "</div>"
                    )

                if s["has_ais"] and s["ais_timestamp"]:
                    ts_text = f"AIS · {s['ais_timestamp'][:16]} | PW · {s['last_date']}"
                    ts_color = "#3498db"
                else:
                    ts_text = f"PortWatch · {s['last_date']}"
                    ts_color = "#c9a96e"

                card_html = (
                        "<div style=\""
                        f"background:linear-gradient(135deg,#0f1520 0%,#1a2332 100%);"
                        f"border:1px solid {s['main_color']};"
                        "border-radius:10px;padding:12px 10px;text-align:center;"
                        "box-shadow:0 4px 16px rgba(0,0,0,0.4);margin-bottom:8px;\">"

                        f"<div style=\"color:{s['main_color']};font-weight:700;"
                        "font-size:12px;letter-spacing:0.05em;margin-bottom:4px;\">"
                        f"{s['name']}</div>"

                        + ais_badge +

                        "<div style=\"color:#e8c97a;font-size:22px;font-weight:800;"
                        "letter-spacing:-0.02em;margin:4px 0 2px;\">"
                        f"{s['latest']}</div>"

                        "<div style=\"color:#6a5a3a;font-size:10px;margin-bottom:6px;\">"
                        f"油轮/日 · 90日均值 {s['avg_90d']}</div>"

                        f"<div style=\"background:{s['main_color']};color:white;"
                        "border-radius:4px;padding:2px 8px;font-size:11px;"
                        "font-weight:600;display:inline-block;margin-bottom:6px;\">"
                        f"{s['main_risk']} · {s['ratio']}%</div>"

                        "<div style=\"color:#4a3f28;font-size:9px;"
                        "margin-top:4px;line-height:1.3;\">"
                        f"{s['importance']}</div>"

                        "<div style=\"margin-top:6px;padding-top:6px;"
                        "border-top:1px solid rgba(255,255,255,0.05);"
                        f"color:{ts_color};font-size:9px;font-weight:500;\">"
                        f"{ts_text}</div>"

                        "</div>"
                )
                st.markdown(card_html, unsafe_allow_html=True)
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

        SENTIMENT_CFG = {
            "bullish": {"label": "看涨", "color": "#2ecc71", "bg": "rgba(46,204,113,0.1)", "icon": "▲"},
            "bearish": {"label": "看跌", "color": "#e74c3c", "bg": "rgba(231,76,60,0.1)", "icon": "▼"},
            "neutral": {"label": "中性", "color": "#95a5a6", "bg": "rgba(149,165,166,0.1)", "icon": "●"},
        }
        EVENT_LABEL = {
            "geopolitics": "地缘政治",
            "supply": "供应",
            "demand": "需求",
            "policy": "政策",
            "macro": "宏观",
            "other": "其他",
        }
        DURATION_LABEL = {
            "short": "短期",
            "medium": "中期",
            "long": "长期",
        }

        for item in country_news:
            sent = item.get("sentiment", "neutral")
            cfg = SENTIMENT_CFG.get(sent, SENTIMENT_CFG["neutral"])
            score = item.get("score", 0.0)
            conf = item.get("confidence", 0.0)
            etype = EVENT_LABEL.get(item.get("event_type", "other"), "其他")
            duration = DURATION_LABEL.get(item.get("impact_duration", "short"), "短期")
            reason = item.get("reason", "")
            url = item.get("url", "")
            title = item.get("title", "")
            date = item.get("date", "")
            source = item.get("source", "")

            # 评分条宽度
            bar_width = int(abs(score) * 100)
            bar_color = cfg["color"]

            title_html = (
                f'<a href="{url}" target="_blank" style="color:#e8c97a;'
                f'text-decoration:none;font-weight:600;font-size:13px;line-height:1.4;">'
                f'{title}</a>'
                if url else
                f'<span style="color:#e8c97a;font-weight:600;font-size:13px;">{title}</span>'
            )

            card = (
                    f"<div style=\"background:{cfg['bg']};border:1px solid {cfg['color']};"
                    "border-radius:10px;padding:12px 14px;margin-bottom:10px;\">"

                    # 标题行
                    f"<div style=\"margin-bottom:8px;\">{title_html}</div>"

                    # 标签行
                    "<div style=\"display:flex;gap:6px;flex-wrap:wrap;margin-bottom:8px;align-items:center;\">"

                    # 情感标签
                    f"<span style=\"background:{cfg['color']};color:white;border-radius:4px;"
                    f"padding:2px 8px;font-size:11px;font-weight:700;\">"
                    f"{cfg['icon']} {cfg['label']}</span>"

                    # 事件类型
                    f"<span style=\"background:rgba(200,146,42,0.2);color:#c8922a;border-radius:4px;"
                    f"padding:2px 8px;font-size:11px;font-weight:600;\">{etype}</span>"

                    # 影响周期
                    f"<span style=\"background:rgba(255,255,255,0.05);color:#888;border-radius:4px;"
                    f"padding:2px 8px;font-size:11px;\">{duration}</span>"

                    # 来源和日期
                    f"<span style=\"color:#555;font-size:11px;margin-left:auto;\">"
                    f"{source} · {date}</span>"

                    "</div>"

                    # 评分条
                    "<div style=\"margin-bottom:6px;\">"
                    "<div style=\"background:rgba(255,255,255,0.05);border-radius:4px;height:4px;\">"
                    f"<div style=\"background:{bar_color};width:{bar_width}%;height:4px;"
                    "border-radius:4px;\"></div>"
                    "</div>"
                    f"<div style=\"color:#555;font-size:9px;margin-top:2px;\">"
                    f"影响强度 {score:+.2f} · 置信度 {int(conf * 100)}%</div>"
                    "</div>"

                    # 分析原因
                    + (
                        f"<div style=\"color:#777;font-size:11px;font-style:italic;"
                        f"border-top:1px solid rgba(255,255,255,0.05);padding-top:6px;\">"
                        f"{reason}</div>"
                        if reason and reason != "analysis failed" else ""
                    ) +

                    "</div>"
            )
            st.markdown(card, unsafe_allow_html=True)
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

        # 优先读情感分析结果
        sentiment_path = os.path.join(ROOT_DIR, "data", "processed", "news_sentiment_detail.csv")
        if os.path.exists(sentiment_path):
            news_display = pd.read_csv(sentiment_path)
            has_sentiment = True
        else:
            news_display = news.copy()
            has_sentiment = False

        news_display = news_display.sort_values("date", ascending=False).head(8)

        SENTIMENT_CFG = {
            "bullish": {"label": "看涨", "color": "#2ecc71", "bg": "rgba(46,204,113,0.08)", "icon": "▲",
                        "bar": "#2ecc71"},
            "bearish": {"label": "看跌", "color": "#e74c3c", "bg": "rgba(231,76,60,0.08)", "icon": "▼",
                        "bar": "#e74c3c"},
            "neutral": {"label": "中性", "color": "#95a5a6", "bg": "rgba(149,165,166,0.06)", "icon": "●",
                        "bar": "#95a5a6"},
        }
        EVENT_LABEL = {
            "geopolitics": "地缘政治", "supply": "供应",
            "demand": "需求", "policy": "政策",
            "macro": "宏观", "other": "其他",
        }

        for _, row in news_display.iterrows():
            url = str(row.get("url", ""))
            title = str(row.get("title", ""))
            source = str(row.get("source", ""))
            date = str(row.get("date", ""))[:10]
            desc = str(row.get("description", ""))[:120] if row.get("description") else ""

            if has_sentiment:
                sent = str(row.get("sentiment", "neutral"))
                cfg = SENTIMENT_CFG.get(sent, SENTIMENT_CFG["neutral"])
                score = float(row.get("score", 0)) if pd.notna(row.get("score")) else 0.0
                conf = float(row.get("confidence", 0)) if pd.notna(row.get("confidence")) else 0.0
                etype = EVENT_LABEL.get(str(row.get("event_type", "other")), "其他")
                reason = str(row.get("reason", ""))
                bar_w = int(abs(score) * 100)
            else:
                cfg = SENTIMENT_CFG["neutral"]
                score = 0.0
                conf = 0.0
                etype = "其他"
                reason = ""
                bar_w = 0

            title_html = (
                f'<a href="{url}" target="_blank" style="color:#e8c97a;'
                f'text-decoration:none;font-weight:600;font-size:13px;line-height:1.5;">'
                f'{title}</a>'
                if url else
                f'<span style="color:#e8c97a;font-weight:600;font-size:13px;">{title}</span>'
            )

            desc_html = (
                f'<div style="color:#666;font-size:11px;margin:4px 0 6px;'
                f'line-height:1.4;">{desc}...</div>'
                if desc else ""
            )

            reason_html = (
                f'<div style="color:#666;font-size:11px;font-style:italic;'
                f'border-top:1px solid rgba(255,255,255,0.04);padding-top:6px;margin-top:4px;">'
                f'{reason}</div>'
                if reason and reason not in ["analysis failed", "nan"] else ""
            )

            card = (
                    f'<div style="background:{cfg["bg"]};'
                    f'border-left:3px solid {cfg["color"]};'
                    'border-radius:0 8px 8px 0;'
                    'padding:12px 14px;margin-bottom:8px;">'

                    f'<div style="margin-bottom:6px;">{title_html}</div>'

                    + desc_html +

                    '<div style="display:flex;gap:6px;flex-wrap:wrap;'
                    'align-items:center;margin-bottom:6px;">'

                    + (
                        f'<span style="background:{cfg["color"]};color:white;'
                        f'border-radius:4px;padding:1px 7px;font-size:11px;font-weight:700;">'
                        f'{cfg["icon"]} {cfg["label"]}</span>'
                        f'<span style="background:rgba(200,146,42,0.15);color:#c8922a;'
                        f'border-radius:4px;padding:1px 7px;font-size:11px;">{etype}</span>'
                        if has_sentiment else ""
                    ) +

                    f'<span style="color:#555;font-size:11px;margin-left:auto;">'
                    f'{source} · {date}</span>'
                    '</div>'

                    + (
                        f'<div style="background:rgba(255,255,255,0.04);'
                        f'border-radius:3px;height:3px;margin-bottom:3px;">'
                        f'<div style="background:{cfg["bar"]};width:{bar_w}%;'
                        f'height:3px;border-radius:3px;"></div></div>'
                        f'<div style="color:#444;font-size:9px;">'
                        f'影响强度 {score:+.2f} · 置信度 {int(conf * 100)}%</div>'
                        if has_sentiment and bar_w > 0 else ""
                    ) +

                    reason_html +

                    '</div>'
            )
            st.markdown(card, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# 页面三：风险预测
# ══════════════════════════════════════════════════════════════════════════
elif page == "风险预测":
    st.title("风险预测")
    st.caption("未来10日 WTI 原油价格涨跌幅预测")

    rt_price, rt_date, rt_live = get_realtime_price()

    # ── 第一层：统计模型预测 ──────────────────────────────────────────
    try:
        last_low   = pred_df["pred_enhanced_low"].iloc[-1]
        last_mid   = pred_df["pred_enhanced_mid"].iloc[-1]
        last_high  = pred_df["pred_enhanced_high"].iloc[-1]
        model_mode = "统计模型（Enhanced XGBoost）"
    except:
        last_low, last_mid, last_high = -0.05, 0.01, 0.05
        model_mode = "默认预测"

    # ── 第二层：极端情景匹配（extreme_scenario.py）────────────────────
    extreme_active = False
    extreme_result = None
    similar_events = []
    scenarios_30d  = {}
    trigger_type   = "none"
    vol_ratio      = 1.0
    vix            = 20.0

    try:
        from extreme_scenario import get_extreme_prediction
        latest_feat_path = os.path.join(ROOT_DIR, "data", "processed", "latest_features.csv")
        latest_feat      = pd.read_csv(latest_feat_path, index_col=0, parse_dates=True)
        latest_row       = latest_feat.iloc[-1]
        vol_ratio        = float(latest_row.get("vol_ratio", 1))
        vix              = float(latest_row.get("VIX", 20))

        extreme_result = get_extreme_prediction(latest_row, last_low, last_mid, last_high)

        if extreme_result["activated"] or is_black_swan:
            extreme_active = True
            last_low       = extreme_result["pred_low"]
            last_mid       = extreme_result["pred_mid"]
            last_high      = extreme_result["pred_high"]
            model_mode     = "极端情景匹配模式（第二层）"
            similar_events = extreme_result.get("similar_events", [])
            scenarios_30d  = extreme_result.get("scenarios_30d", {})
            trigger_type   = extreme_result.get("trigger_type", "geopolitics")
    except Exception as e:
        # 降级：简单放大
        if is_black_swan:
            extreme_active = True
            last_low       = last_low  * 2.5
            last_high      = last_high * 3.0
            model_mode     = "极端事件放大模式（降级）"

    risk_label, risk_color, _ = get_risk_level(last_low, last_mid, last_high)

    # ── 侧边栏开关 ────────────────────────────────────────────────────
    with st.sidebar:
        st.divider()
        manual_override = st.toggle(
            "手动关闭黑天鹅模式", value=False,
            help="当您判断极端事件已解除时，可手动关闭黑天鹅预警"
        )
        if manual_override:
            is_black_swan = False

    # ── 状态提示 ──────────────────────────────────────────────────────
    if is_black_swan:
        st.error("当前处于极端地缘政治事件期间（霍尔木兹封锁）· 统计模型已切换至极端情景模式")
    elif extreme_active:
        st.warning(f"极端市场环境（波动率 {vol_ratio:.1f}x · VIX {vix:.1f}）· 置信区间已放大")
    else:
        st.success(f"市场处于正常波动区间 · {model_mode}")

    # ── 核心预测数字 ──────────────────────────────────────────────────
    col0, col1, col2, col3 = st.columns(4)
    with col0:
        st.metric("当前 WTI", f"${rt_price:.2f}", rt_date)
    with col1:
        st.markdown("#### 悲观情景 (P10)")
        st.markdown(f"<h2 style='color:#e74c3c'>{last_low*100:.1f}%</h2>", unsafe_allow_html=True)
        st.caption("10%概率跌幅超过此值")
    with col2:
        mid_color = "#2ecc71" if last_mid > 0 else "#e74c3c"
        st.markdown("#### 基准预测 (P50)")
        st.markdown(f"<h2 style='color:{mid_color}'>{last_mid*100:.1f}%</h2>", unsafe_allow_html=True)
        st.caption("最可能的10日涨跌幅")
    with col3:
        st.markdown("#### 乐观情景 (P90)")
        st.markdown(f"<h2 style='color:#2ecc71'>{last_high*100:.1f}%</h2>", unsafe_allow_html=True)
        st.caption("10%概率涨幅超过此值")

    price_low  = rt_price * (1 + last_low)
    price_mid  = rt_price * (1 + last_mid)
    price_high = rt_price * (1 + last_high)
    st.caption(
        f"对应价格区间：${price_low:.1f} ~ ${price_high:.1f}"
        f"（中位数 ${price_mid:.1f}）· 未来10日 · {model_mode}"
    )

    st.divider()

    # ── 黑天鹅：AI情景分析图 ─────────────────────────────────────────
    if is_black_swan:
        st.subheader("AI 情景价格区间")

        # 优先用 extreme_scenario 的匹配结果，否则降级到固定比例
        if scenarios_30d:
            vals   = list(scenarios_30d.values())
            labels = list(scenarios_30d.keys())
            colors = ["#2ecc71", "#f1c40f", "#e74c3c"]
            scenarios = {
                labels[0]: {"low": round(rt_price * (1 + vals[0] * 0.7), 1),
                             "high": round(rt_price * (1 + vals[0] * 1.3), 1),
                             "color": colors[0]},
                labels[1]: {"low": round(rt_price * (1 + vals[1] * 0.8), 1),
                             "high": round(rt_price * (1 + vals[1] * 1.2), 1),
                             "color": colors[1]},
                labels[2]: {"low": round(rt_price * (1 + vals[2] * 0.9), 1),
                             "high": round(rt_price * (1 + vals[2] * 1.5), 1),
                             "color": colors[2]},
            }
        else:
            # 降级：固定比例
            scenarios = {
                "缓解情景（外交解决）": {
                    "low": round(rt_price * 0.88, 1),
                    "high": round(rt_price * 0.98, 1),
                    "color": "#2ecc71",
                },
                "基准情景（当前延续）": {
                    "low": round(rt_price * 0.95, 1),
                    "high": round(rt_price * 1.12, 1),
                    "color": "#f1c40f",
                },
                "升级情景（冲突扩大）": {
                    "low": round(rt_price * 1.10, 1),
                    "high": round(rt_price * 1.28, 1),
                    "color": "#e74c3c",
                },
            }
        fig_bs = go.Figure()
        fig_bs.add_hline(
            y=rt_price, line_dash="dash", line_color="white", line_width=2,
            annotation_text=f"当前 ${rt_price:.1f}",
            annotation_font_color="white",
        )
        for label, s in scenarios.items():
            fig_bs.add_trace(go.Bar(
                x=[label], y=[s["high"] - s["low"]], base=[s["low"]],
                marker_color=s["color"], marker_opacity=0.75, name=label,
                text=f"${s['low']} ~ ${s['high']}",
                textposition="inside",
                textfont=dict(color="white", size=12), width=0.4,
            ))
        fig_bs.update_layout(
            height=400, margin=dict(l=0, r=0, t=30, b=0),
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="white"),
            yaxis=dict(gridcolor="#2d3748", title="WTI 美元/桶"),
            xaxis=dict(gridcolor="#2d3748"),
            showlegend=False, barmode="overlay",
            title="未来1-2周 WTI 价格情景区间",
        )
        st.plotly_chart(fig_bs, use_container_width=True, key="chart_bs")

        # ── 最相似历史事件 ────────────────────────────────────────────
        if similar_events:
            st.subheader("历史情景匹配")
            st.caption("基于波动率、VIX、航运数据、地缘情感的加权余弦相似度匹配")
            SEVERITY_COLOR = {
                "extreme" : "#e74c3c",
                "severe"  : "#e67e22",
                "moderate": "#f1c40f",
            }
            SEVERITY_LABEL = {
                "extreme" : "极端",
                "severe"  : "严重",
                "moderate": "中等",
            }
            TRIGGER_LABEL = {
                "geopolitics"    : "地缘政治",
                "macro"          : "宏观冲击",
                "supply_policy"  : "供应政策",
                "demand_recovery": "需求恢复",
            }
            for ev in similar_events:
                sev_color = SEVERITY_COLOR.get(ev.get("severity", "moderate"), "#f1c40f")
                sev_label = SEVERITY_LABEL.get(ev.get("severity", "moderate"), "中等")
                tri_label = TRIGGER_LABEL.get(ev.get("trigger", ""), ev.get("trigger", ""))
                ret_10d   = ev.get("actual_return")
                ret_30d   = ev.get("return_30d") or ev.get("typical_30d")
                ret_10d_str = f"{ret_10d*100:+.1f}%" if ret_10d is not None else "N/A"
                ret_30d_str = f"{ret_30d*100:+.1f}%" if ret_30d is not None else "N/A"
                sim_pct     = int(ev.get("similarity", 0) * 100)

                st.markdown(
                    f"<div style='background:rgba(255,255,255,0.04);border-left:3px solid {sev_color};"
                    f"border-radius:0 8px 8px 0;padding:12px 14px;margin-bottom:8px;'>"
                    f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                    f"<span style='color:#e8c97a;font-weight:700;font-size:14px;'>{ev['event']}</span>"
                    f"<span style='color:#3498db;font-size:13px;font-weight:600;'>相似度 {sim_pct}%</span>"
                    f"</div>"
                    f"<div style='color:#888;font-size:12px;margin:4px 0;'>{ev.get('description','')}</div>"
                    f"<div style='display:flex;gap:8px;margin-top:6px;flex-wrap:wrap;'>"
                    f"<span style='background:{sev_color};color:white;border-radius:4px;padding:2px 8px;font-size:11px;font-weight:700;'>{sev_label}</span>"
                    f"<span style='background:rgba(200,146,42,0.2);color:#c8922a;border-radius:4px;padding:2px 8px;font-size:11px;'>{tri_label}</span>"
                    f"<span style='color:#666;font-size:11px;'>触发日：{ev.get('trigger_date','')}</span>"
                    f"<span style='color:#2ecc71;font-size:11px;font-weight:600;'>10日实际：{ret_10d_str}</span>"
                    f"<span style='color:#3498db;font-size:11px;font-weight:600;'>30日实际：{ret_30d_str}</span>"
                    f"</div></div>",
                    unsafe_allow_html=True
                )

        report_path = os.path.join(ROOT_DIR, "data", "raw", "black_swan_report.json")
        if os.path.exists(report_path):
            import json
            with open(report_path, "r", encoding="utf-8") as f:
                bs_report = json.load(f)
            analysis = bs_report.get("analysis", {}).get("analysis", "")
            if analysis:
                with st.expander("查看完整 AI 情景分析", expanded=False):
                    st.markdown(analysis)
                    st.caption("生成时间：" + bs_report.get("analysis", {}).get("generated_at", ""))

        st.divider()
        st.subheader("封锁期间油价走势")
        recent_bs = pred_df["2026-01-01":]
        fig_wti   = go.Figure()
        fig_wti.add_trace(go.Scatter(
            x=recent_bs.index, y=recent_bs["WTI"],
            name="WTI 实际价格", line=dict(color="#3498db", width=2)
        ))
        fig_wti.add_vline(
            x=pd.Timestamp("2026-03-02").timestamp() * 1000,
            line_dash="dash", line_color="#e74c3c",
            annotation_text="霍尔木兹封锁", annotation_font_color="#e74c3c",
        )
        fig_wti.update_layout(
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font=dict(color="white"),
            xaxis=dict(gridcolor="#2d3748"),
            yaxis=dict(gridcolor="#2d3748", title="美元/桶"),
        )
        st.plotly_chart(fig_wti, use_container_width=True, key="chart_wti")
        st.divider()

    # ── 报告生成 ──────────────────────────────────────────────────────
    st.subheader("企业银行分析报告")
    st.caption("由 DeepSeek 生成 · 面向航空、化工、能源贸易商等企业客户")

    report_path = os.path.join(ROOT_DIR, "data", "raw", "latest_report.json")
    has_cache   = os.path.exists(report_path)
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
            "生成最新报告", type="primary",
            use_container_width=True, key="btn_gen_report"
        )
    with col_btn2:
        show_cache = st.button(
            "查看上次报告", use_container_width=True,
            key="btn_cache_report"
        ) if has_cache else False

    if gen_report:
        with st.spinner("正在调用 DeepSeek 生成分析报告，约需10-20秒..."):
            try:
                from report_generator import generate_report
                recent_news_list = news.to_dict("records") if len(news) > 0 else []
                result = generate_report(
                    current_price      = rt_price,
                    pred_low           = last_low,
                    pred_mid           = last_mid,
                    pred_high          = last_high,
                    feature_importance = importance,
                    is_black_swan      = is_black_swan,
                    bs_signals         = bs_signals if is_black_swan else None,
                    recent_news        = recent_news_list,
                )
                if result["status"] == "ok":
                    st.success(
                        "报告生成成功 ｜ 模式：" + result["mode"] +
                        " ｜ 生成时间：" + result["generated_at"]
                    )
                    st.markdown(result["report"])
                    st.download_button(
                        label="下载报告（TXT）",
                        data=result["report"].encode("utf-8"),
                        file_name=f"OilSense_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                        mime="text/plain", key="btn_dl_report"
                    )
                else:
                    st.error("报告生成失败：" + result.get("error", ""))
            except Exception as e:
                st.error(f"调用失败：{e}")
    elif has_cache and show_cache:
        st.markdown(cached.get("report", ""))
        st.download_button(
            label="下载报告（TXT）",
            data=cached.get("report", "").encode("utf-8"),
            file_name="OilSense_Report_cached.txt",
            mime="text/plain", key="btn_dl_cache"
        )

    st.divider()

    # ── 近60日预测区间走势 ────────────────────────────────────────────
    st.subheader("近60日预测区间走势")
    st.caption("红色阴影区域为黑天鹅事件期间，统计模型预测仅供参考")

    recent = pred_df.tail(60)
    fig    = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(recent.index) + list(recent.index[::-1]),
        y=list(recent["pred_enhanced_high"]) + list(recent["pred_enhanced_low"][::-1]),
        fill="toself", fillcolor="rgba(52,152,219,0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="风险区间 P10~P90"
    ))
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent["pred_enhanced_mid"],
        name="Enhanced预测（P50）", line=dict(color="#3498db", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent["pred_baseline_mid"],
        name="Baseline预测（P50）", line=dict(color="#95a5a6", width=1.5, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent["target"],
        name="实际涨跌幅", line=dict(color="#e74c3c", width=1.5), opacity=0.8
    ))
    fig.add_hline(y=0, line_dash="solid", line_color="rgba(255,255,255,0.3)", line_width=1)
    fig.add_vrect(
        x0="2026-03-02", x1=str(recent.index[-1].date()),
        fillcolor="rgba(231,76,60,0.08)", line_width=0,
        annotation_text="黑天鹅期间", annotation_position="top left",
        annotation_font_color="#e74c3c",
    )
    fig.update_layout(
        height=420, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font=dict(color="white"),
        xaxis=dict(gridcolor="#2d3748"),
        yaxis=dict(gridcolor="#2d3748", title="10日涨跌幅"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            bgcolor="rgba(0,0,0,0)", font=dict(color="white")
        )
    )
    st.plotly_chart(fig, use_container_width=True, key="chart_hist")

    st.divider()

    # ── 模型性能对比（两套模型整合评估）────────────────────────────────
    st.subheader("模型性能对比（样本外测试集）")

    test_start = pred_df.index[int(len(pred_df) * 0.8)]
    test_df    = pred_df.loc[test_start:]
    actual     = test_df["target"]

    # 黑天鹅期间（极端情景模型负责）
    BS_START = pd.Timestamp("2026-03-02")
    normal_df  = test_df[test_df.index < BS_START]
    extreme_df = test_df[test_df.index >= BS_START]

    # 正常期间：XGBoost Enhanced 准确率
    perf = {}
    for version in ["enhanced", "baseline"]:
        mid   = normal_df[f"pred_{version}_mid"]
        act   = normal_df["target"]
        valid = act.notna() & mid.notna()
        acc   = np.mean(np.sign(mid[valid]) == np.sign(act[valid])) if valid.sum() > 0 else 0.5
        perf[version] = round(acc * 100, 2)

    # 极端期间：用情景匹配模型的方向（weighted_return_10d 的符号 vs 实际）
    extreme_acc = None
    if len(extreme_df) > 0 and extreme_result is not None and extreme_result.get("activated"):
        pred_dir  = np.sign(extreme_result.get("weighted_return_10d", 0))
        act_ext   = extreme_df["target"].dropna()
        if len(act_ext) > 0:
            extreme_acc = round(
                float(np.mean(np.sign(act_ext) == pred_dir)) * 100, 2
            )

    # 综合准确率（加权平均）
    n_normal  = normal_df["target"].notna().sum()
    n_extreme = extreme_df["target"].notna().sum()
    n_total   = n_normal + n_extreme
    if extreme_acc is not None and n_total > 0:
        hybrid_acc = round(
            (perf["enhanced"] * n_normal + extreme_acc * n_extreme) / n_total, 2
        )
    else:
        hybrid_acc = perf["enhanced"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("综合准确率（双模型）", f"{hybrid_acc}%",
                  f"+{round(hybrid_acc-50,2)}% vs 随机基准")
    with col2:
        st.metric("Enhanced 准确率（正常期）", f"{perf['enhanced']}%",
                  f"+{round(perf['enhanced']-50,2)}%")
    with col3:
        if extreme_acc is not None:
            st.metric("情景匹配准确率（极端期）", f"{extreme_acc}%",
                      f"+{round(extreme_acc-50,2)}%")
        else:
            st.metric("情景匹配准确率（极端期）", "暂无数据",
                      "极端期间样本不足")
    with col4:
        st.metric("测试集样本数", f"{len(test_df)} 条",
                  f"正常{n_normal}条 极端{n_extreme}条")

    st.caption("💡 正常市场由 Enhanced XGBoost 负责，极端事件期间自动切换至历史情景匹配模型")

    st.divider()

    # ── 特征重要性 ────────────────────────────────────────────────────
    st.subheader("模型特征重要性 Top 12")
    top_imp = importance.head(12)
    sentiment_cols = [
        "sentiment_score", "geopolitics_flag", "policy_flag", "news_count",
        "gdelt_goldstein", "gdelt_tone", "gdelt_conflict_cnt", "gdelt_coop_cnt",
        "gdelt_conflict_intensity", "gdelt_mentions",
        "gdelt_goldstein_chg", "gdelt_conflict_ma5", "gdelt_tone_chg",
    ]
    bar_colors = ["#e74c3c" if f in sentiment_cols else "#3498db"
                  for f in top_imp["feature"]]
    fig2 = go.Figure(go.Bar(
        x=top_imp["importance"][::-1].values,
        y=top_imp["feature"][::-1].values,
        orientation="h",
        marker_color=bar_colors[::-1],
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
    st.plotly_chart(fig2, use_container_width=True, key="chart_importance")
    st.caption("🔴 红色 = LLM情感/GDELT地缘政治因子  🔵 蓝色 = 传统量化因子")

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