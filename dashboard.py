"""
SmartContainer Risk Engine - Streamlit Dashboard
=================================================
Run:  streamlit run dashboard.py
"""

import warnings, os, joblib, io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    confusion_matrix, f1_score, recall_score,
    accuracy_score, classification_report,
)
import shap

warnings.filterwarnings("ignore")

# ---------------------------------------------
# PAGE CONFIG
# ---------------------------------------------
st.set_page_config(
    page_title="SmartContainer Risk Engine",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------
# THEME / CSS
# ---------------------------------------------
st.markdown("""
<style>
/* Import font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark background */
.stApp { background-color: #0d1117; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
}
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }

/* Main text */
.stMarkdown, .stText, p, h1, h2, h3, h4, label { color: #e6edf3 !important; }

/* KPI Card */
.kpi-card {
    background: linear-gradient(135deg, #161b22, #1c2128);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
}
.kpi-label  { font-size: 12px; font-weight: 500; color: #8b949e !important; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 8px; }
.kpi-value  { font-size: 36px; font-weight: 700; margin: 0; }
.kpi-critical { color: #f85149 !important; }
.kpi-lowrisk  { color: #d29922 !important; }
.kpi-clear    { color: #3fb950 !important; }
.kpi-blue     { color: #58a6ff !important; }
.kpi-purple   { color: #bc8cff !important; }
.kpi-teal     { color: #39d353 !important; }

/* Section header */
.section-header {
    border-left: 4px solid #58a6ff;
    padding-left: 14px;
    margin: 32px 0 20px 0;
}
.section-header h2 { font-size: 20px; font-weight: 600; color: #e6edf3 !important; margin: 0; }
.section-header p  { font-size: 13px; color: #8b949e !important; margin: 4px 0 0 0; }

/* Table styling */
.dataframe thead th { background-color: #161b22 !important; color: #e6edf3 !important; }
.dataframe tbody td { background-color: #0d1117 !important; color: #e6edf3 !important; }

/* Metric override */
div[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px;
}
div[data-testid="metric-container"] label { color: #8b949e !important; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #e6edf3 !important; }

/* Divider */
hr { border-color: #21262d !important; }

/* Selectbox / multiselect */
.stSelectbox > div, .stMultiSelect > div { background: #161b22 !important; border-color: #30363d !important; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------
# CONSTANTS
# ---------------------------------------------
HIST_PATH  = "Historical Data.csv"
PRED_PATH  = "my_test_results.csv"
MODEL_PATH = "risk_model.pkl"

LABEL_MAP  = {"Clear": 0, "Low Risk": 1, "Critical": 2}
RISK_COLORS = {
    "Critical": "#f85149",
    "Low Risk": "#d29922",
    "Clear":    "#3fb950",
}

PLOTLY_DARK = dict(
    paper_bgcolor="#0d1117",
    plot_bgcolor="#161b22",
    font_color="#e6edf3",
    colorway=["#58a6ff", "#3fb950", "#f85149", "#d29922", "#bc8cff", "#39d353"],
)
_AXIS_BASE = dict(gridcolor="#21262d", linecolor="#30363d")

def dark_layout(**extra):
    """Return PLOTLY_DARK merged with per-call axis/extra overrides."""
    kw = {**PLOTLY_DARK}
    # Merge axis overrides on top of the base axis style
    for ax in ("xaxis", "yaxis"):
        if ax in extra:
            kw[ax] = {**_AXIS_BASE, **extra.pop(ax)}
        else:
            kw[ax] = _AXIS_BASE
    kw.update(extra)
    return kw

FEATURES = [
    "weight_diff_abs","weight_diff_pct","weight_mismatch_flag",
    "weight_mismatch_severe","weight_ratio",
    "log_declared_value","log_declared_weight","log_measured_weight",
    "log_dwell_time","value_per_kg","log_value_per_kg","zero_value_flag",
    "hour_of_day","day_of_week","month","year",
    "hs_chapter","is_transit",
    "importer_risk_rate","exporter_risk_rate","country_risk_rate",
    "port_risk_rate","shipping_line_risk_rate","dest_country_risk_rate",
    "route_risk_rate","entity_risk_combined",
    "anomaly_score","is_anomaly",
    "Origin_Country_enc","Destination_Country_enc","Destination_Port_enc",
    "Shipping_Line_enc","Importer_ID_enc","Exporter_ID_enc",
]

FEATURE_LABELS = {
    "weight_diff_abs":        "Weight Difference (Abs)",
    "weight_diff_pct":        "Weight Mismatch %",
    "weight_mismatch_flag":   "Weight Mismatch >5%",
    "weight_mismatch_severe": "Weight Mismatch >15%",
    "weight_ratio":           "Measured/Declared Ratio",
    "log_declared_value":     "Declared Value (log)",
    "log_declared_weight":    "Declared Weight (log)",
    "log_measured_weight":    "Measured Weight (log)",
    "log_dwell_time":         "Dwell Time (log)",
    "value_per_kg":           "Value per kg",
    "log_value_per_kg":       "Value per kg (log)",
    "zero_value_flag":        "Zero Declared Value",
    "hour_of_day":            "Declaration Hour",
    "day_of_week":            "Day of Week",
    "month":                  "Month",
    "year":                   "Year",
    "hs_chapter":             "HS Code Chapter",
    "is_transit":             "Transit Regime",
    "importer_risk_rate":     "Importer Risk Rate",
    "exporter_risk_rate":     "Exporter Risk Rate",
    "country_risk_rate":      "Origin Country Risk Rate",
    "port_risk_rate":         "Destination Port Risk Rate",
    "shipping_line_risk_rate":"Shipping Line Risk Rate",
    "dest_country_risk_rate": "Dest. Country Risk Rate",
    "route_risk_rate":        "Route Risk Rate",
    "entity_risk_combined":   "Entity Risk (Combined)",
    "anomaly_score":          "Anomaly Score",
    "is_anomaly":             "Anomaly Flag",
    "Origin_Country_enc":     "Origin Country (enc)",
    "Destination_Country_enc":"Destination Country (enc)",
    "Destination_Port_enc":   "Destination Port (enc)",
    "Shipping_Line_enc":      "Shipping Line (enc)",
    "Importer_ID_enc":        "Importer ID (enc)",
    "Exporter_ID_enc":        "Exporter ID (enc)",
}

# ---------------------------------------------
# DATA LOADING  (cached)
# ---------------------------------------------
@st.cache_data(show_spinner=False)
def load_hist():
    """Load & clean historical source data (cached separately from predictions)."""
    hist = pd.read_csv(HIST_PATH)
    hist.columns = [c.strip() for c in hist.columns]
    hist.rename(columns={
        "Declaration_Date (YYYY-MM-DD)":            "Declaration_Date",
        "Trade_Regime (Import / Export / Transit)": "Trade_Regime",
    }, inplace=True)
    hist["Declaration_Date"] = pd.to_datetime(hist["Declaration_Date"], errors="coerce")
    return hist

def build_df(pred_df=None):
    """Merge historical data with predictions. Uses uploaded pred_df if supplied,
    otherwise falls back to the default predictions CSV on disk."""
    hist = load_hist()
    if pred_df is not None:
        pred      = pred_df
        join_type = "inner"   # only keep containers that have predictions
    else:
        pred      = pd.read_csv(PRED_PATH)
        join_type = "left"    # keep all historical rows (original behaviour)
    df = hist.merge(pred, on="Container_ID", how=join_type)
    df["Weight_Difference"]     = (df["Measured_Weight"] - df["Declared_Weight"]).abs()
    df["Value_to_Weight_Ratio"] = df["Declared_Value"] / (df["Declared_Weight"] + 0.001)
    df["Weight_Diff_Pct"]       = (df["Weight_Difference"] / (df["Declared_Weight"].abs() + 0.001) * 100)
    return df

# Backward-compat alias kept so nothing else breaks
def load_data():
    return build_df()

@st.cache_resource(show_spinner=False)
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

def section_header(icon, title, subtitle=""):
    st.markdown(f"""
    <div class="section-header">
        <h2>{icon} {title}</h2>
        {"<p>" + subtitle + "</p>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------
# IN-DASHBOARD PREDICTION HELPERS
# ---------------------------------------------
_PRED_FEATURES = [
    "weight_diff_abs", "weight_diff_pct", "weight_mismatch_flag",
    "weight_mismatch_severe", "weight_ratio",
    "log_declared_value", "log_declared_weight", "log_measured_weight",
    "log_dwell_time", "value_per_kg", "log_value_per_kg", "zero_value_flag",
    "hour_of_day", "day_of_week", "month", "year",
    "hs_chapter", "is_transit",
    "importer_risk_rate", "exporter_risk_rate", "country_risk_rate",
    "port_risk_rate", "shipping_line_risk_rate", "dest_country_risk_rate",
    "route_risk_rate", "entity_risk_combined",
    "anomaly_score", "is_anomaly",
    "Origin_Country_enc", "Destination_Country_enc", "Destination_Port_enc",
    "Shipping_Line_enc", "Importer_ID_enc", "Exporter_ID_enc",
]

_ISO_FEATURES = [
    "log_declared_value", "log_declared_weight", "log_measured_weight",
    "weight_diff_pct", "weight_ratio", "log_dwell_time", "log_value_per_kg",
    "hour_of_day", "hs_chapter", "importer_risk_rate", "exporter_risk_rate",
    "country_risk_rate", "is_transit"
]

_FEAT_LABELS = {
    "weight_diff_abs": "weight difference (abs)",
    "weight_diff_pct": "weight mismatch %",
    "weight_mismatch_flag": "weight mismatch >5%",
    "weight_mismatch_severe": "weight mismatch >15%",
    "weight_ratio": "measured/declared weight ratio",
    "log_declared_value": "declared value",
    "log_declared_weight": "declared weight",
    "log_measured_weight": "measured weight",
    "log_dwell_time": "dwell time",
    "value_per_kg": "value per kg",
    "log_value_per_kg": "value-per-kg (log)",
    "zero_value_flag": "zero declared value",
    "hour_of_day": "declaration hour",
    "day_of_week": "day of week",
    "month": "month",
    "year": "year",
    "hs_chapter": "HS code chapter",
    "is_transit": "transit regime",
    "importer_risk_rate": "importer risk rate",
    "exporter_risk_rate": "exporter risk rate",
    "country_risk_rate": "origin country risk rate",
    "port_risk_rate": "destination port risk rate",
    "shipping_line_risk_rate": "shipping line risk rate",
    "dest_country_risk_rate": "destination country risk rate",
    "route_risk_rate": "origin-to-destination route risk",
    "entity_risk_combined": "entity risk (combined)",
    "anomaly_score": "anomaly score",
    "is_anomaly": "anomaly flag",
    "Origin_Country_enc": "origin country",
    "Destination_Country_enc": "destination country",
    "Destination_Port_enc": "destination port",
    "Shipping_Line_enc": "shipping line",
    "Importer_ID_enc": "importer ID",
    "Exporter_ID_enc": "exporter ID",
}

def _engineer_features(df, encoders, rate_tables):
    """Mirror of predict.py engineer_features — inline to avoid import issues."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns={
        "Declaration_Date (YYYY-MM-DD)": "Declaration_Date",
        "Trade_Regime (Import / Export / Transit)": "Trade_Regime",
    }, inplace=True)

    df["Declaration_Date"] = pd.to_datetime(df["Declaration_Date"], errors="coerce")
    df["hour_of_day"] = pd.to_datetime(df["Declaration_Time"], format="%H:%M:%S", errors="coerce").dt.hour.fillna(12)
    df["day_of_week"] = df["Declaration_Date"].dt.dayofweek.fillna(0)
    df["month"]       = df["Declaration_Date"].dt.month.fillna(1)
    df["year"]        = df["Declaration_Date"].dt.year.fillna(2020)

    df["weight_diff_abs"]        = df["Measured_Weight"] - df["Declared_Weight"]
    df["weight_diff_pct"]        = df["weight_diff_abs"] / (df["Declared_Weight"].abs() + 0.001) * 100
    df["weight_mismatch_flag"]   = (df["weight_diff_pct"].abs() > 5).astype(int)
    df["weight_mismatch_severe"] = (df["weight_diff_pct"].abs() > 15).astype(int)
    df["weight_ratio"]           = df["Measured_Weight"] / (df["Declared_Weight"] + 0.001)

    df["value_per_kg"]       = df["Declared_Value"] / (df["Declared_Weight"] + 0.001)
    df["log_declared_value"] = np.log1p(df["Declared_Value"].clip(lower=0))
    df["log_declared_weight"]= np.log1p(df["Declared_Weight"].clip(lower=0))
    df["log_measured_weight"]= np.log1p(df["Measured_Weight"].clip(lower=0))
    df["log_dwell_time"]     = np.log1p(df["Dwell_Time_Hours"].clip(lower=0))
    df["log_value_per_kg"]   = np.log1p(df["value_per_kg"].clip(lower=0))
    df["zero_value_flag"]    = (df["Declared_Value"] == 0).astype(int)

    df["hs_chapter"] = (df["HS_Code"] // 10000).astype(int)
    df["is_transit"] = (df["Trade_Regime"].str.strip() == "Transit").astype(int)

    global_rate = 0.01
    df["route"] = df["Origin_Country"].astype(str) + "_" + df["Destination_Country"].astype(str)
    df["importer_risk_rate"]     = df["Importer_ID"].map(rate_tables.get("importer_risk_rate", {})).fillna(global_rate)
    df["exporter_risk_rate"]     = df["Exporter_ID"].map(rate_tables.get("exporter_risk_rate", {})).fillna(global_rate)
    df["country_risk_rate"]      = df["Origin_Country"].map(rate_tables.get("country_risk_rate", {})).fillna(global_rate)
    df["port_risk_rate"]         = df["Destination_Port"].map(rate_tables.get("port_risk_rate", {})).fillna(global_rate)
    df["shipping_line_risk_rate"]= df["Shipping_Line"].map(rate_tables.get("shipping_line_risk_rate", {})).fillna(global_rate)
    df["dest_country_risk_rate"] = df["Destination_Country"].map(rate_tables.get("dest_country_risk_rate", {})).fillna(global_rate)
    df["route_risk_rate"]        = df["route"].map(rate_tables.get("route_risk_rate", {})).fillna(global_rate)

    df["entity_risk_combined"] = (
        0.30 * df["importer_risk_rate"] +
        0.20 * df["exporter_risk_rate"] +
        0.15 * df["country_risk_rate"] +
        0.10 * df["port_risk_rate"] +
        0.10 * df["shipping_line_risk_rate"] +
        0.10 * df["dest_country_risk_rate"] +
        0.05 * df["route_risk_rate"]
    )

    cat_cols = ["Origin_Country", "Destination_Country", "Destination_Port",
                "Shipping_Line", "Importer_ID", "Exporter_ID", "Trade_Regime"]
    for col in cat_cols:
        le = encoders.get(col)
        if le is None:
            df[col + "_enc"] = 0
            continue
        known = set(le.classes_)
        df[col + "_enc"] = df[col].astype(str).apply(
            lambda x: le.transform([x])[0] if x in known else -1
        )
    return df


def _run_prediction(df_raw, model, encoders, iso_model, rate_tables,
                    critical_thresh=55.0, low_risk_thresh=22.0):
    """Run end-to-end prediction on an uploaded dataframe."""
    df = _engineer_features(df_raw, encoders, rate_tables)
    df["anomaly_score"] = -iso_model.score_samples(df[_ISO_FEATURES])
    df["is_anomaly"]    = (iso_model.predict(df[_ISO_FEATURES]) == -1).astype(int)

    X = df[_PRED_FEATURES]
    proba      = model.predict_proba(X)
    risk_raw   = 0.35 * proba[:, 1] + 1.0 * proba[:, 2]
    risk_score = (risk_raw * 100).clip(0, 100).round(2)

    def _level(s):
        if s >= critical_thresh:  return "Critical"
        if s >= low_risk_thresh:  return "Low Risk"
        return "Clear"

    df["Risk_Score"] = risk_score
    df["Risk_Level"] = [_level(s) for s in risk_score]

    # SHAP
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    pred_class  = model.predict(X)
    sv = np.array(shap_values)
    if sv.ndim == 3:
        shap_pred = np.array([sv[i, :, c] for i, c in enumerate(pred_class)])
    else:
        shap_pred = np.array([sv[c][i] for i, c in enumerate(pred_class)])

    _PLAIN_ENGLISH = {
        ("weight_diff_abs",        "HIGH"): "Declared weight differs significantly from physically measured weight",
        ("weight_diff_pct",        "HIGH"): "Declared weight is more than 5% off from the measured weight",
        ("weight_diff_pct",        "LOW" ): "Declared and measured weights are closely aligned",
        ("weight_mismatch_flag",   "HIGH"): "Weight discrepancy exceeds the 5% tolerance threshold",
        ("weight_mismatch_severe", "HIGH"): "Severe weight discrepancy — more than 15% difference detected",
        ("weight_ratio",           "HIGH"): "Measured weight is much higher than what was declared",
        ("weight_ratio",           "LOW" ): "Measured weight is much lower than what was declared",
        ("log_declared_value",     "HIGH"): "Declared value is unusually high for this type of goods",
        ("log_declared_value",     "LOW" ): "Declared value appears suspiciously low",
        ("value_per_kg",           "HIGH"): "Value per kilogram is abnormally high — possible mis-classification of goods",
        ("value_per_kg",           "LOW" ): "Value per kilogram is abnormally low — possible under-valuation",
        ("log_value_per_kg",       "HIGH"): "Value-to-weight ratio is unusually high",
        ("log_value_per_kg",       "LOW" ): "Value-to-weight ratio is unusually low",
        ("zero_value_flag",        "HIGH"): "Declared value is zero — shipment may be under-declared",
        ("log_declared_weight",    "HIGH"): "Declared weight is unusually large",
        ("log_declared_weight",    "LOW" ): "Declared weight is unusually small",
        ("log_measured_weight",    "HIGH"): "Physically measured weight is abnormally high",
        ("log_measured_weight",    "LOW" ): "Physically measured weight is abnormally low",
        ("importer_risk_rate",      "HIGH"): "This importer has a history of flagged or seized shipments",
        ("importer_risk_rate",      "LOW" ): "This importer has a clean compliance record",
        ("exporter_risk_rate",      "HIGH"): "This exporter has been associated with previous risk incidents",
        ("exporter_risk_rate",      "LOW" ): "This exporter has a low-risk track record",
        ("country_risk_rate",       "HIGH"): "Shipment originates from a country with elevated smuggling risk",
        ("country_risk_rate",       "LOW" ): "Origin country has a low customs risk profile",
        ("port_risk_rate",          "HIGH"): "Destination port has elevated risk and enforcement activity",
        ("port_risk_rate",          "LOW" ): "Destination port has a low risk profile",
        ("shipping_line_risk_rate", "HIGH"): "The shipping line has been linked to previous risk incidents",
        ("shipping_line_risk_rate", "LOW" ): "The shipping line has a clean compliance record",
        ("dest_country_risk_rate",  "HIGH"): "Destination country has elevated risk indicators",
        ("dest_country_risk_rate",  "LOW" ): "Destination country is low-risk",
        ("route_risk_rate",         "HIGH"): "This trade route has a high incidence of flagged shipments",
        ("route_risk_rate",         "LOW" ): "This is a low-risk trade route",
        ("entity_risk_combined",    "HIGH"): "Multiple parties involved (importer, exporter, route) have elevated risk profiles",
        ("entity_risk_combined",    "LOW" ): "All parties involved have low risk profiles",
        ("anomaly_score",  "HIGH"): "Shipment shows highly unusual patterns compared to normal traffic",
        ("anomaly_score",  "LOW" ): "Shipment patterns are consistent with normal traffic",
        ("is_anomaly",     "HIGH"): "Shipment has been flagged as a statistical anomaly",
        ("log_dwell_time", "HIGH"): "Shipment has been at port for an unusually long time",
        ("log_dwell_time", "LOW" ): "Dwell time is shorter than typical",
        ("hour_of_day",    "HIGH"): "Declaration was submitted at an unusual hour",
        ("day_of_week",    "HIGH"): "Declaration was submitted on an unusual day of the week",
        ("month",          "HIGH"): "Seasonal pattern suggests elevated risk for this period",
        ("hs_chapter",  "HIGH"): "Goods are classified under a high-risk HS code category",
        ("hs_chapter",  "LOW" ): "Goods fall under a low-risk HS code category",
        ("is_transit",  "HIGH"): "Shipment is in transit regime, which carries inherently higher risk",
        ("Origin_Country_enc",      "HIGH"): "Origin country is associated with elevated risk",
        ("Destination_Country_enc", "HIGH"): "Destination country is associated with elevated risk",
        ("Destination_Port_enc",    "HIGH"): "Destination port is associated with elevated risk",
        ("Shipping_Line_enc",       "HIGH"): "Shipping line is associated with elevated risk",
        ("Importer_ID_enc",         "HIGH"): "Importer profile indicates elevated risk",
        ("Exporter_ID_enc",         "HIGH"): "Exporter profile indicates elevated risk",
    }

    def _explain(row):
        # Only features with POSITIVE SHAP pushed toward the predicted risk level
        pos_idx = np.where(row > 0)[0]
        if len(pos_idx) == 0:
            pos_idx = np.arange(len(row))  # fallback
        top = pos_idx[np.argsort(row[pos_idx])[::-1][:3]]
        parts = []
        for i in top:
            feat = _PRED_FEATURES[i]
            sentence = _PLAIN_ENGLISH.get(
                (feat, "HIGH"), _FEAT_LABELS.get(feat, feat) + " (elevated)"
            )
            parts.append(sentence)
        return "; ".join(parts)

    df["Explanation_Summary"] = [_explain(shap_pred[i]) for i in range(len(df))]
    return df[["Container_ID", "Risk_Score", "Risk_Level", "Explanation_Summary"]]

# ---------------------------------------------
# SIDEBAR  (filters only)
# ---------------------------------------------
with st.sidebar:
    st.markdown("## 🚢 SmartContainer\n**Risk Engine**")
    st.markdown("---")
    st.markdown("### 🔧 Filters")
    st.markdown("<small style='color:#8b949e'>Filters apply to all charts below.</small>",
                unsafe_allow_html=True)

    # Placeholders — populated after df_all is built below
    _risk_ph    = st.empty()
    _country_ph = st.empty()
    _port_ph    = st.empty()
    _regime_ph  = st.empty()
    _slider_ph  = st.empty()

    score_min, score_max = 0.0, 100.0   # defaults; overwritten by slider below

    st.markdown("---")
    st.markdown("### 📂 Data Files")
    st.markdown("📄 **Historical:** `Historical Data.csv`")
    st.markdown("📊 **Predictions:** `my_test_results.csv`")
    st.markdown("🤖 **Model:** `risk_model.pkl`")
    st.markdown("---")
    st.markdown("<small style='color:#8b949e'>SmartContainer Risk Engine v2.0<br>HackaMined 2026</small>",
                unsafe_allow_html=True)

# ── Build df_all using uploaded predictions if available ──────────────────────
if "prediction_df" in st.session_state:
    df_all = build_df(pred_df=st.session_state["prediction_df"])
else:
    df_all = build_df()

model = load_model()

# ── Fill sidebar filter widgets now that df_all is ready ──────────────────────
with _risk_ph.container():
    risk_levels_available = sorted(df_all["Risk_Level"].dropna().unique().tolist())
    sel_risk = st.multiselect("Risk Level", risk_levels_available,
                              default=risk_levels_available, key="f_risk")

with _country_ph.container():
    countries_available = sorted(df_all["Origin_Country"].dropna().unique().tolist())
    sel_country = st.multiselect("Origin Country", ["All"] + countries_available,
                                 default=["All"], key="f_country")

with _port_ph.container():
    ports_available = sorted(df_all["Destination_Port"].dropna().unique().tolist())
    sel_port = st.multiselect("Destination Port", ["All"] + ports_available,
                              default=["All"], key="f_port")

with _regime_ph.container():
    regimes_available = sorted(df_all["Trade_Regime"].dropna().unique().tolist())
    sel_regime = st.multiselect("Trade Regime", ["All"] + regimes_available,
                                default=["All"], key="f_regime")

with _slider_ph.container():
    score_min, score_max = st.slider("Risk Score Range", 0.0, 100.0,
                                     (0.0, 100.0), step=1.0, key="f_score")

# Apply filters
df = df_all.copy()
if sel_risk:
    df = df[df["Risk_Level"].isin(sel_risk)]
if "All" not in sel_country and sel_country:
    df = df[df["Origin_Country"].isin(sel_country)]
if "All" not in sel_port and sel_port:
    df = df[df["Destination_Port"].isin(sel_port)]
if "All" not in sel_regime and sel_regime:
    df = df[df["Trade_Regime"].isin(sel_regime)]
df = df[(df["Risk_Score"] >= score_min) & (df["Risk_Score"] <= score_max)]

# ---------------------------------------------
# HEADER
# ---------------------------------------------
st.markdown("""
<div style="background: linear-gradient(135deg, #161b22, #1c2128);
            border: 1px solid #30363d; border-radius: 16px;
            padding: 28px 36px; margin-bottom: 28px;">
    <h1 style="color:#e6edf3 !important; font-size:32px; font-weight:700; margin:0;">
        🚢 SmartContainer Risk Engine
    </h1>
    <p style="color:#8b949e; margin:8px 0 0 0; font-size:15px;">
        AI-powered container shipment risk monitoring · Real-time inspection prioritization
    </p>
</div>
""", unsafe_allow_html=True)

# ===============================================================
# UPLOAD & PREDICT  (top of main area)
# ===============================================================
st.markdown("""
<div style="background:linear-gradient(135deg,#0d2137,#0d1117);
            border:1px solid #1f6feb; border-radius:14px;
            padding:22px 28px; margin-bottom:24px;">
    <div style="font-size:18px;font-weight:700;color:#58a6ff;margin-bottom:4px;">
        📤 Upload Shipment Data &amp; Run Predictions
    </div>
    <div style="font-size:13px;color:#8b949e;">
        Upload any shipment CSV — results will refresh all charts automatically.
    </div>
</div>
""", unsafe_allow_html=True)

_up_col1, _up_col2, _up_col3 = st.columns([4, 2, 2])
with _up_col1:
    uploaded_csv = st.file_uploader(
        "Choose a shipment CSV",
        type=["csv"],
        key="upload_csv",
        label_visibility="collapsed",
        help="Any shipment CSV with the standard column names. Filename does not matter."
    )
with _up_col2:
    run_btn = st.button(
        "🔍 Run Predictions",
        disabled=(uploaded_csv is None),
        use_container_width=True,
        type="primary",
    )

# ── Run predictions when button is pressed ────────────────────────────────────
if run_btn and uploaded_csv is not None:
    with st.spinner("Running predictions — feature engineering + SHAP (may take ~240 s)…"):
        try:
            _m   = joblib.load(MODEL_PATH)
            _enc = joblib.load("label_encoders.pkl")
            _iso = joblib.load("isolation_forest.pkl")
            _rt  = joblib.load("risk_rate_tables.pkl")
            _raw = pd.read_csv(uploaded_csv)
            _result = _run_prediction(_raw, _m, _enc, _iso, _rt)
            st.session_state["prediction_df"]  = _result
            st.session_state["prediction_raw"] = _raw
            st.rerun()   # ← triggers a full re-run; df_all rebuilds from session_state
        except Exception as _e:
            st.error(f"❌ Prediction failed: {_e}")

# ── Download button (always visible after predictions are stored) ──────────────
if "prediction_df" in st.session_state:
    _pred_ready = st.session_state["prediction_df"]
    _csv_bytes  = _pred_ready.to_csv(index=False).encode("utf-8")
    _d1, _d2 = st.columns([5, 2])
    with _d1:
        st.info(
            f"**{len(_pred_ready):,}** containers scored — "
            f"🔴 Critical: **{(_pred_ready['Risk_Level']=='Critical').sum():,}**  "
            f"🟡 Low Risk: **{(_pred_ready['Risk_Level']=='Low Risk').sum():,}**  "
            f"🟢 Clear: **{(_pred_ready['Risk_Level']=='Clear').sum():,}**"
        )
    with _d2:
        st.download_button(
            label="⬇️ Download Results CSV",
            data=_csv_bytes,
            file_name="risk_predictions.csv",
            mime="text/csv",
            use_container_width=True,
            key="dl_top",
        )

st.markdown("---")

# ===============================================================
# SECTION 1 - SYSTEM OVERVIEW
# ===============================================================
section_header("📊", "System Overview", "Aggregate KPIs for the current filtered dataset")

total       = len(df)
n_critical  = (df["Risk_Level"] == "Critical").sum()
n_low       = (df["Risk_Level"] == "Low Risk").sum()
n_clear     = (df["Risk_Level"] == "Clear").sum()
avg_score   = df["Risk_Score"].mean() if total > 0 else 0
avg_dwell   = df["Dwell_Time_Hours"].mean() if total > 0 else 0
n_anomaly   = int((df["Weight_Diff_Pct"] > 5).sum())

kpi_html = f"""
<div style="display:grid; grid-template-columns:repeat(6,1fr); gap:14px; margin-bottom:28px;">
    <div class="kpi-card">
        <div class="kpi-label">Total Containers</div>
        <div class="kpi-value kpi-blue">{total:,}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Critical Risk</div>
        <div class="kpi-value kpi-critical">{n_critical:,}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Low Risk</div>
        <div class="kpi-value kpi-lowrisk">{n_low:,}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Clear</div>
        <div class="kpi-value kpi-clear">{n_clear:,}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Avg Risk Score</div>
        <div class="kpi-value kpi-purple">{avg_score:.1f}</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Avg Dwell Time (h)</div>
        <div class="kpi-value kpi-teal">{avg_dwell:.1f}</div>
    </div>
</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)

# Secondary row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Weight Anomalies (>5% mismatch)", f"{n_anomaly:,}",
          delta=f"{n_anomaly/max(total,1)*100:.1f}% of total")
c2.metric("Critical Rate", f"{n_critical/max(total,1)*100:.2f}%")
c3.metric("Max Risk Score", f"{df['Risk_Score'].max():.1f}")
c4.metric("Containers w/ Zero Value", f"{int((df['Declared_Value']==0).sum()):,}")

st.markdown("---")

# ===============================================================
# SECTION 2 - RISK DISTRIBUTION
# ===============================================================
section_header("🎯", "Risk Distribution", "Breakdown of predicted risk levels and score spread")

col1, col2 = st.columns(2)

with col1:
    rl_cnt = df["Risk_Level"].value_counts().reset_index()
    rl_cnt.columns = ["Risk_Level", "Count"]
    fig_pie = px.pie(
        rl_cnt, names="Risk_Level", values="Count",
        color="Risk_Level",
        color_discrete_map=RISK_COLORS,
        hole=0.5,
        title="Risk Level Distribution",
    )
    fig_pie.update_traces(textfont_size=14, pull=[0.04, 0.04, 0.08])
    fig_pie.update_layout(**PLOTLY_DARK, title_font_size=16)
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    fig_hist = px.histogram(
        df, x="Risk_Score", nbins=60, color="Risk_Level",
        color_discrete_map=RISK_COLORS,
        title="Risk Score Distribution",
        labels={"Risk_Score": "Risk Score (0-100)"},
        barmode="overlay", opacity=0.8,
    )
    fig_hist.update_layout(**PLOTLY_DARK, title_font_size=16)
    fig_hist.add_vline(x=55, line_dash="dash", line_color="#f85149",
                       annotation_text="Critical Threshold (55)",
                       annotation_font_color="#f85149")
    fig_hist.add_vline(x=22, line_dash="dash", line_color="#d29922",
                       annotation_text="Low Risk Threshold (22)",
                       annotation_font_color="#d29922")
    st.plotly_chart(fig_hist, use_container_width=True)

# # Risk score violin by level
# fig_violin = px.violin(
#     df, x="Risk_Level", y="Risk_Score", color="Risk_Level",
#     color_discrete_map=RISK_COLORS,
#     box=True, points=False,
#     title="Risk Score Spread by Level",
# )
# fig_violin.update_layout(**PLOTLY_DARK, title_font_size=16)
# st.plotly_chart(fig_violin, use_container_width=True)

st.markdown("---")

# ===============================================================
# SECTION 3 - TRADE FLOW ANALYSIS
# ===============================================================
section_header("🌍", "Trade Flow Analysis", "Origin countries, destination ports, and trade regimes")

col1, col2, col3 = st.columns(3)

with col1:
    top_countries = (
        df.groupby("Origin_Country")["Risk_Score"].count()
        .sort_values(ascending=False).head(15).reset_index()
    )
    top_countries.columns = ["Country", "Shipments"]
    fig_oc = px.bar(
        top_countries, x="Shipments", y="Country", orientation="h",
        title="Top 15 Origin Countries",
        color="Shipments", color_continuous_scale="Blues",
    )
    fig_oc.update_layout(**dark_layout(title_font_size=15,
                         yaxis=dict(autorange="reversed"),
                         coloraxis_showscale=False))
    st.plotly_chart(fig_oc, use_container_width=True)

with col2:
    top_ports = (
        df.groupby("Destination_Port")["Risk_Score"].count()
        .sort_values(ascending=False).head(15).reset_index()
    )
    top_ports.columns = ["Port", "Shipments"]
    fig_dp = px.bar(
        top_ports, x="Shipments", y="Port", orientation="h",
        title="Top 15 Destination Ports",
        color="Shipments", color_continuous_scale="Teal",
    )
    fig_dp.update_layout(**dark_layout(title_font_size=15,
                         yaxis=dict(autorange="reversed"),
                         coloraxis_showscale=False))
    st.plotly_chart(fig_dp, use_container_width=True)

# with col3:
#     regime_cnt = df["Trade_Regime"].value_counts().reset_index()
#     regime_cnt.columns = ["Regime", "Count"]
#     fig_reg = px.bar(
#         regime_cnt, x="Regime", y="Count",
#         title="Trade Regime Distribution",
#         color="Regime",
#         color_discrete_sequence=["#58a6ff", "#3fb950", "#d29922"],
#     )
#     fig_reg.update_layout(**PLOTLY_DARK, title_font_size=15, showlegend=False)
#     st.plotly_chart(fig_reg, use_container_width=True)

st.markdown("---")

# ===============================================================
# SECTION 4 - ANOMALY DETECTION
# ===============================================================
section_header("🔬", "Anomaly Detection", "Physical inconsistencies in weight declarations and dwell patterns")

col1, col2 = st.columns(2)

with col1:
    sample = df.sample(min(3000, len(df)), random_state=42) if len(df) > 3000 else df
    fig_scatter = px.scatter(
        sample,
        x="Declared_Weight", y="Measured_Weight",
        color="Risk_Level", color_discrete_map=RISK_COLORS,
        title="Declared vs Measured Weight",
        labels={"Declared_Weight": "Declared Weight (kg)",
                "Measured_Weight": "Measured Weight (kg)"},
        opacity=0.65, size_max=6,
        hover_data=["Container_ID", "Risk_Score"],
    )
    max_w = max(sample["Declared_Weight"].max(), sample["Measured_Weight"].max())
    fig_scatter.add_trace(go.Scatter(
        x=[0, max_w], y=[0, max_w],
        mode="lines", line=dict(color="#8b949e", dash="dash", width=1),
        name="Perfect Match", showlegend=True,
    ))
    fig_scatter.update_layout(**PLOTLY_DARK, title_font_size=16)
    st.plotly_chart(fig_scatter, use_container_width=True)

with col2:
    fig_dwell = px.histogram(
        df, x="Dwell_Time_Hours", nbins=60,
        color="Risk_Level", color_discrete_map=RISK_COLORS,
        title="Dwell Time Distribution by Risk Level",
        labels={"Dwell_Time_Hours": "Dwell Time (hours)"},
        barmode="overlay", opacity=0.8,
    )
    fig_dwell.update_layout(**PLOTLY_DARK, title_font_size=16)
    st.plotly_chart(fig_dwell, use_container_width=True)

col3, col4 = st.columns(2)

# with col3:
#     vwr_clip = df[df["Value_to_Weight_Ratio"] < df["Value_to_Weight_Ratio"].quantile(0.97)]
#     fig_box = px.box(
#         vwr_clip, x="Risk_Level", y="Value_to_Weight_Ratio",
#         color="Risk_Level", color_discrete_map=RISK_COLORS,
#         title="Value-to-Weight Ratio by Risk Level",
#         labels={"Value_to_Weight_Ratio": "Value / Weight (USD/kg)"},
#     )
#     fig_box.update_layout(**PLOTLY_DARK, title_font_size=16, showlegend=False)
#     st.plotly_chart(fig_box, use_container_width=True)

with col3:
    fig_wd = px.histogram(
        df[df["Weight_Diff_Pct"] < 50],
        x="Weight_Diff_Pct", nbins=60,
        color="Risk_Level", color_discrete_map=RISK_COLORS,
        title="Weight Mismatch % Distribution",
        labels={"Weight_Diff_Pct": "Weight Mismatch (%)"},
        barmode="overlay", opacity=0.8,
    )
    fig_wd.update_layout(**PLOTLY_DARK, title_font_size=16)
    fig_wd.add_vline(x=5, line_dash="dash", line_color="#d29922", annotation_text="5% threshold")
    fig_wd.add_vline(x=15, line_dash="dash", line_color="#f85149", annotation_text="15% threshold")
    st.plotly_chart(fig_wd, use_container_width=True)

st.markdown("---")

# ===============================================================
# SECTION 5 - HIGH RISK CONTAINERS TABLE
# ===============================================================
section_header("🚨", "High Risk Containers", "Filterable table of flagged containers for inspection")

show_levels = st.multiselect(
    "Filter by Risk Level",
    options=["Critical", "Low Risk", "Clear"],
    default=["Critical", "Low Risk"],
    key="tbl_risk_level",
)

tbl_df = df[df["Risk_Level"].isin(show_levels)].copy()
tbl_df = tbl_df.sort_values("Risk_Score", ascending=False)

display_cols = [
    "Container_ID", "Risk_Score", "Risk_Level",
    "Declared_Value", "Declared_Weight", "Measured_Weight",
    "Weight_Difference", "Origin_Country", "Destination_Port",
    "Trade_Regime", "Dwell_Time_Hours", "Explanation_Summary",
]
display_cols = [c for c in display_cols if c in tbl_df.columns]

def color_risk(val):
    colors = {"Critical": "color: #f85149; font-weight:700",
              "Low Risk": "color: #d29922; font-weight:700",
              "Clear":    "color: #3fb950; font-weight:700"}
    return colors.get(val, "")

def color_score(val):
    try:
        v = float(val)
        if v >= 55:   return "color: #f85149; font-weight:700"
        elif v >= 22: return "color: #d29922"
        else:         return "color: #3fb950"
    except:
        return ""

styled = (
    tbl_df[display_cols].head(500)
    .style
    .applymap(color_risk,   subset=["Risk_Level"])
    .applymap(color_score,  subset=["Risk_Score"])
    .format({
        "Risk_Score":      "{:.2f}",
        "Declared_Value":  "${:,.0f}",
        "Declared_Weight": "{:,.1f} kg",
        "Measured_Weight": "{:,.1f} kg",
        "Weight_Difference":"{:,.1f} kg",
        "Dwell_Time_Hours":"{:.1f} h",
    })
)
st.info(f"Showing {min(500, len(tbl_df)):,} of {len(tbl_df):,} containers")
st.dataframe(styled, use_container_width=True, height=420)



# ===============================================================
# SECTION 7 - MODEL EVALUATION METRICS
# ===============================================================
section_header("📈", "Model Evaluation Metrics",
               "Comprehensive evaluation against ground-truth Clearance_Status labels")

# ── Load ground truth + predictions ───────────────────────────────────────────
# Priority: uploaded data (session_state) → disk files
_eval_source = "uploaded file"
try:
    if "prediction_df" in st.session_state and "prediction_raw" in st.session_state:
        df_ground_truth = st.session_state["prediction_raw"].copy()
        df_ground_truth.columns = [c.strip() for c in df_ground_truth.columns]
        df_predictions  = st.session_state["prediction_df"]
        _eval_source = "uploaded file"
    else:
        df_ground_truth = pd.read_csv("Test Data.csv")
        df_predictions  = pd.read_csv("my_test_results.csv")
        _eval_source = "disk (Test Data.csv / my_test_results.csv)"

    # Merge on Container_ID
    df_eval = pd.merge(df_ground_truth, df_predictions, on="Container_ID", how="inner")

    # Prepare labels
    labels_ord = ["Clear", "Low Risk", "Critical"]
    LABEL_MAP = {label: i for i, label in enumerate(labels_ord)}

    perf_df = df_eval.dropna(subset=["Clearance_Status", "Risk_Level"])
    y_true  = perf_df["Clearance_Status"].map(LABEL_MAP)
    y_pred  = perf_df["Risk_Level"].map(LABEL_MAP)
    valid   = y_true.notna() & y_pred.notna()
    y_true, y_pred = y_true[valid], y_pred[valid]

    if len(y_true) == 0:
        st.warning("⚠️ No valid ground-truth + prediction pairs found. Evaluation metrics cannot be shown.")
        st.stop()

    _acc         = accuracy_score(y_true, y_pred)
    _macro_f1    = f1_score(y_true, y_pred, average="macro")
    _weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    _critical_f1 = f1_score(y_true, y_pred, average=None, labels=[LABEL_MAP["Critical"]])[0]
    _critical_rc = recall_score(y_true, y_pred, average=None, labels=[LABEL_MAP["Critical"]])[0]
    _cm          = confusion_matrix(y_true, y_pred).tolist()

    report      = classification_report(y_true, y_pred,
                                        target_names=labels_ord, output_dict=True)
    _per_class  = {c: {"precision": report[c]["precision"],
                       "recall":    report[c]["recall"],
                       "f1":        report[c]["f1-score"],
                       "support":   int(report[c]["support"])} for c in labels_ord}
    _macro_avg  = {"precision": report["macro avg"]["precision"],
                   "recall":    report["macro avg"]["recall"],
                   "f1":        report["macro avg"]["f1-score"]}
    _weighted_avg = {"precision": report["weighted avg"]["precision"],
                     "recall":    report["weighted avg"]["recall"],
                     "f1":        report["weighted avg"]["f1-score"]}

except FileNotFoundError:
    st.warning("⚠️ Evaluation files not found on disk. Upload a CSV and click 'Run Predictions' to see metrics.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error computing metrics: {e}")
    st.stop()

st.caption(f"✅ Live metrics from **{_eval_source}** · {int(valid.sum()):,} samples")

# ── 1. KPI METRICS ────────────────────────────────────────────────────────────
st.markdown("#### 📊 Key Performance Indicators")
col_kpi1, col_kpi2, col_kpi3, col_kpi4, col_kpi5 = st.columns(5)
col_kpi1.metric("✅ Accuracy",           f"{_acc:.4f}",         f"{_acc*100:.2f}%  correct")
col_kpi2.metric("🏆 Macro F1 (Primary)",  f"{_macro_f1:.4f}",    f"{_macro_f1*100:.2f}%")
col_kpi3.metric("⚖️ Weighted F1",         f"{_weighted_f1:.4f}", f"{_weighted_f1*100:.2f}%")
col_kpi4.metric("🎯 F1 — Critical",       f"{_critical_f1:.4f}", f"{_critical_f1*100:.2f}%")
col_kpi5.metric("🔴 Recall — Critical",   f"{_critical_rc:.4f}", f"{_critical_rc*100:.2f}%")

st.markdown("---")

# ── 2. PER-CLASS METRICS TABLE & BAR CHART ────────────────────────────────────
section_header("📋", "Per-Class Classification Report",
               "Precision · Recall · F1-Score · Support for each risk category")

col_tbl, col_bar = st.columns([1, 1])

with col_tbl:
    rows = []
    for cls in labels_ord:
        m = _per_class[cls]
        rows.append({
            "Class": cls,
            "Precision": m["precision"],
            "Recall":    m["recall"],
            "F1-Score":  m["f1"],
            "Support":   m["support"],
        })
    # Append macro/weighted avg rows
    rows.append({"Class": "── macro avg",    **{k: _macro_avg[v]    for k, v in [("Precision","precision"),("Recall","recall"),("F1-Score","f1")]}, "Support": sum(r["Support"] for r in rows[:3])})
    rows.append({"Class": "── weighted avg", **{k: _weighted_avg[v] for k, v in [("Precision","precision"),("Recall","recall"),("F1-Score","f1")]}, "Support": sum(r["Support"] for r in rows[:3])})

    rpt_df = pd.DataFrame(rows)

    def _style_val(v):
        try:
            f = float(v)
            if f >= 0.999: return "color:#3fb950; font-weight:700"
            if f >= 0.97:  return "color:#58a6ff; font-weight:600"
            if f >= 0.95:  return "color:#d29922; font-weight:600"
            return "color:#f85149; font-weight:600"
        except: return ""

    def _style_class(v):
        cmap = {"Clear": "color:#3fb950;font-weight:700",
                "Low Risk": "color:#d29922;font-weight:700",
                "Critical": "color:#f85149;font-weight:700"}
        return cmap.get(v, "color:#8b949e;font-style:italic")

    styled_rpt = (rpt_df.style
        .applymap(_style_class, subset=["Class"])
        .applymap(_style_val,   subset=["Precision", "Recall", "F1-Score"])
        .format({"Precision": "{:.4f}", "Recall": "{:.4f}",
                 "F1-Score":  "{:.4f}", "Support": "{:,.0f}"})
    )
    st.dataframe(styled_rpt, use_container_width=True, hide_index=True)

# with col_bar:
#     bar_data = []
#     for cls in labels_ord:
#         m = _per_class[cls]
#         for metric, val in [("Precision", m["precision"]),
#                             ("Recall",    m["recall"]),
#                             ("F1-Score",  m["f1"])]:
#             bar_data.append({"Class": cls, "Metric": metric, "Score": val})
#     bar_df = pd.DataFrame(bar_data)
#     fig_bar = px.bar(
#         bar_df, x="Class", y="Score", color="Metric", barmode="group",
#         title="Per-Class Precision / Recall / F1-Score",
#         color_discrete_sequence=["#58a6ff", "#3fb950", "#f85149"],
#         text_auto=".4f",
#     )
#     fig_bar.update_traces(textposition="outside", textfont_size=10)
#     fig_bar.update_layout(**dark_layout(title_font_size=15,
#                           yaxis=dict(range=[0, 1.06])), height=360)
#     st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ── 3. CONFUSION MATRIX ───────────────────────────────────────────────────────
section_header("🔲", "Confusion Matrix",
               "Actual vs Predicted risk class — diagonal = correctly classified")

cm_arr = np.array(_cm)

col_cm, col_cm_info = st.columns([2, 1])

with col_cm:
    # Annotate with count + percentage
    total_per_row = cm_arr.sum(axis=1, keepdims=True)
    cm_pct = cm_arr / total_per_row.clip(min=1) * 100
    text_matrix = [[f"{cm_arr[r][c]}<br>{cm_pct[r][c]:.1f}%"
                    for c in range(len(labels_ord))] for r in range(len(labels_ord))]
    max_val = cm_arr.max()
    text_colors = [["white" if cm_arr[r][c] > max_val*0.4 else "#0d1117"
                for c in range(len(labels_ord))]
                for r in range(len(labels_ord))]

    fig_cm = go.Figure(go.Heatmap(
        z=cm_arr,
        x=[f"Pred {l}" for l in labels_ord],
        y=[f"Actual {l}" for l in labels_ord],
        text=text_matrix,
        texttemplate="%{text}",
        textfont={"size": 14, "color": "white"},
        colorscale="Blues",
        showscale=True,
        colorbar=dict(tickfont=dict(color="#e6edf3")),
    ))
    for r in range(len(labels_ord)):
        for c in range(len(labels_ord)):
            fig_cm.add_annotation(
                x=f"Pred {labels_ord[c]}",
                y=f"Actual {labels_ord[r]}",
                text=text_matrix[r][c],
                showarrow=False,
            font=dict(
                color=text_colors[r][c],
                size=14
            )
        )
    fig_cm.update_layout(
        **PLOTLY_DARK,
        title=dict(text="Confusion Matrix (Count + Row %)", font_size=15),
        xaxis=dict(side="bottom", **_AXIS_BASE),
        yaxis=dict(autorange="reversed", **_AXIS_BASE),
        height=340,
    )
    st.plotly_chart(fig_cm, use_container_width=True)

with col_cm_info:
    st.markdown("""
    <div style="background:#161b22; border:1px solid #30363d; border-radius:12px;
                padding:20px; margin-top:8px;">
        <div style="font-size:13px; font-weight:600; color:#e6edf3; margin-bottom:14px;">
            📖 How to Read
        </div>
        <div style="font-size:12px; color:#8b949e; line-height:1.7;">
            <b style="color:#3fb950;">Diagonal cells</b> show correct predictions.<br><br>
            <b style="color:#f85149;">Off-diagonal cells</b> show misclassifications.<br><br>
            Each cell shows <b style="color:#58a6ff;">count</b> and
            <b style="color:#d29922;">% of actual class</b>.<br><br>
            A perfect model has zeros everywhere except the diagonal.
        </div>
    </div>
    """, unsafe_allow_html=True)

    tp_critical = int(cm_arr[2][2])
    fn_critical = int(cm_arr[2][0] + cm_arr[2][1])
    fp_critical = int(cm_arr[0][2] + cm_arr[1][2])
    st.markdown("**🔴 Critical Class Breakdown**")
    k1, k2, k3 = st.columns(3)
    k1.metric("True Positives",  str(tp_critical))
    k2.metric("False Negatives", str(fn_critical))
    k3.metric("False Positives", str(fp_critical))

st.markdown("---")

# ── 4. PRIMARY & SECONDARY METRICS SUMMARY CARD ───────────────────────────────
section_header("🥇", "Competition Scoring Metrics",
               "Primary and secondary metrics as defined in the challenge evaluation criteria")

m_col1, m_col2 = st.columns(2)

with m_col1:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1a2332, #0d1a2a);
                border: 2px solid #58a6ff; border-radius: 14px;
                padding: 28px; text-align: center;">
        <div style="font-size: 13px; font-weight: 600; color: #58a6ff;
                    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 8px;">
            🏆 PRIMARY METRIC
        </div>
        <div style="font-size: 14px; color: #8b949e; margin-bottom: 16px;">
            Macro F1-Score (equal weight across all 3 classes)
        </div>
        <div style="font-size: 56px; font-weight: 800; color: #3fb950; line-height: 1;">
            {_macro_f1:.4f}
        </div>
        <div style="background:#21262d; border-radius:6px; height:8px; margin-top:16px; overflow:hidden;">
            <div style="background: linear-gradient(90deg, #3fb950, #58a6ff);
                        width:{_macro_f1*100:.2f}%; height:100%; border-radius:6px;"></div>
        </div>
        <div style="font-size:12px; color:#8b949e; margin-top:8px;">{_macro_f1*100:.2f}% of perfect score</div>
    </div>
    """, unsafe_allow_html=True)

with m_col2:
    sec_items = [
        ("F1 — Critical Class",   "🎯", _critical_f1, "#f85149"),
        ("Recall — Critical Class","🔴", _critical_rc, "#d29922"),
        ("Weighted F1",           "⚖️", _weighted_f1,  "#bc8cff"),
    ]
    tiles = ""
    for name, icon, val, color in sec_items:
        pct = int(val * 100)
        tiles += f"""
        <div style="background:#161b22; border:1px solid #30363d; border-radius:10px;
                    padding:16px; display:flex; align-items:center; gap:14px;">
            <div style="font-size:24px;">{icon}</div>
            <div style="flex:1;">
                <div style="font-size:12px; color:#8b949e; margin-bottom:4px;">{name}</div>
                <div style="background:#21262d; border-radius:4px; height:6px; overflow:hidden;">
                    <div style="background:{color}; width:{pct}%; height:100%; border-radius:4px;"></div>
                </div>
            </div>
            <div style="font-size:22px; font-weight:700; color:{color}; min-width:70px; text-align:right;">
                {val:.4f}
            </div>
        </div>"""
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1c1a2e, #14101e);
                border: 2px solid #bc8cff; border-radius: 14px;
                padding: 22px;">
        <div style="font-size: 13px; font-weight: 600; color: #bc8cff;
                    letter-spacing: 2px; text-transform: uppercase; margin-bottom: 16px;">
            🎯 SECONDARY METRICS
        </div>
        <div style="display:flex; flex-direction:column; gap:10px;">{tiles}</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("---")

# ===============================================================
# SECTION 7 - EXPLAINABILITY
# ===============================================================
section_header("🧠", "Model Explainability", "LightGBM feature importance and SHAP-derived insights")

col1, col2 = st.columns([3, 2])

with col1:
    if model is not None:
        fi_scores = model.feature_importances_
        fi_df = pd.DataFrame({
            "Feature":    [FEATURE_LABELS.get(f, f) for f in FEATURES],
            "Importance": fi_scores,
        }).sort_values("Importance", ascending=True).tail(20)

        fig_fi = px.bar(
            fi_df, x="Importance", y="Feature", orientation="h",
            title="Top 20 Feature Importance (LightGBM Gain)",
            color="Importance", color_continuous_scale="Blues",
        )
        fig_fi.update_layout(**dark_layout(title_font_size=16,
                             coloraxis_showscale=False,
                             height=500))
        st.plotly_chart(fig_fi, use_container_width=True)
    else:
        st.warning("Model not found. Feature importance unavailable.")

with col2:
    st.markdown("#### 🔍 Top Risk Drivers from SHAP Explanations")
    expl_series = df_all["Explanation_Summary"].dropna()

    # Parse top features from all explanation summaries
    from collections import Counter
    feature_mentions = Counter()
    for expl in expl_series:
        for part in expl.split(";"):
            part = part.strip()
            if part:
                # Strip direction tag: "(HIGH)" or "(LOW)"
                fname = part.replace("(HIGH)", "").replace("(LOW)", "").strip()
                feature_mentions[fname] += 1

    top_features = pd.DataFrame(
        feature_mentions.most_common(12),
        columns=["Feature", "Mentions"]
    )
    fig_shap = px.bar(
        top_features.sort_values("Mentions"), x="Mentions", y="Feature",
        orientation="h", title="Most Referenced SHAP Features",
        color="Mentions", color_continuous_scale="Reds",
    )
    fig_shap.update_layout(**dark_layout(title_font_size=15,
                           coloraxis_showscale=False,
                           height=500))
    st.plotly_chart(fig_shap, use_container_width=True)

# Explanation word tags for Critical containers
st.markdown("#### Critical Containers - Top Explanation Themes")
crit_explanations = df_all[df_all["Risk_Level"] == "Critical"]["Explanation_Summary"].dropna()
if len(crit_explanations) > 0:
    crit_mentions = Counter()
    for expl in crit_explanations:
        for part in expl.split(";"):
            part = part.strip()
            fname = part.replace("(HIGH)", "").replace("(LOW)", "").strip()
            dir_  = "(HIGH)" if "(HIGH)" in part else "(LOW)"
            if fname:
                crit_mentions[f"{fname} {dir_}"] += 1

    tag_df = pd.DataFrame(crit_mentions.most_common(15), columns=["Explanation", "Count"])
    fig_tags = px.bar(
        tag_df.sort_values("Count"), x="Count", y="Explanation",
        orientation="h", color="Count", color_continuous_scale="Reds",
        title="Top Explanation Phrases for Critical Containers",
    )
    fig_tags.update_layout(**dark_layout(coloraxis_showscale=False, height=420))
    st.plotly_chart(fig_tags, use_container_width=True)

st.markdown("---")

# ===============================================================
# SECTION 8 - CONTAINER SEARCH PANEL
# ===============================================================
section_header("🔎", "Container Search Panel", "Look up any container by ID for a full risk profile")

search_id = st.text_input("Enter Container ID", placeholder="e.g. 94748548")

if search_id:
    try:
        cid = int(search_id.strip())
        row = df_all[df_all["Container_ID"] == cid]
        if row.empty:
            st.error(f"Container ID {cid} not found in dataset.")
        else:
            row = row.iloc[0]

            rl = row["Risk_Level"]
            color = RISK_COLORS.get(rl, "#8b949e")

            st.markdown(f"""
            <div style="background:#161b22; border:1px solid {color}; border-radius:14px;
                        padding:24px; margin-top:12px;">
                <div style="display:flex; align-items:center; gap:16px; margin-bottom:20px;">
                    <div style="font-size:36px;">🚢</div>
                    <div>
                        <div style="font-size:22px; font-weight:700; color:#e6edf3;">
                            Container #{cid}
                        </div>
                        <div style="font-size:14px; color:#8b949e;">
                            {row.get('Origin_Country','')} -> {row.get('Destination_Country','')} &nbsp;|&nbsp;
                            {row.get('Trade_Regime','')} &nbsp;|&nbsp;
                            {str(row.get('Declaration_Date',''))[:10]}
                        </div>
                    </div>
                    <div style="margin-left:auto; background:{color}22; border:1px solid {color};
                                border-radius:8px; padding:10px 20px; text-align:center;">
                        <div style="color:{color}; font-size:24px; font-weight:700;">
                            {row.get('Risk_Score', 0):.1f}
                        </div>
                        <div style="color:{color}; font-size:12px; font-weight:600;">{rl}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Declared Weight",  f"{row.get('Declared_Weight', 0):,.1f} kg")
            c2.metric("Measured Weight",  f"{row.get('Measured_Weight', 0):,.1f} kg")
            c3.metric("Weight Difference",f"{row.get('Weight_Difference', 0):,.1f} kg")
            c4.metric("Dwell Time",       f"{row.get('Dwell_Time_Hours', 0):.1f} hrs")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Declared Value",   f"${row.get('Declared_Value', 0):,.0f}")
            c6.metric("Value/Weight",     f"${row.get('Value_to_Weight_Ratio', 0):,.1f}/kg")
            c7.metric("HS Code",          str(row.get("HS_Code", "")))
            c8.metric("Shipping Line",    str(row.get("Shipping_Line", "")))

            st.markdown("#### 🧠 SHAP Explanation")
            expl = str(row.get("Explanation_Summary", "N/A"))
            factors = [f.strip() for f in expl.split(";") if f.strip()]
            cols = st.columns(min(len(factors), 3))
            for i, factor in enumerate(factors[:3]):
                tag_color = "#f85149" if "(HIGH)" in factor else "#3fb950"
                cols[i].markdown(f"""
                <div style="background:#1c2128; border-left:4px solid {tag_color};
                            border-radius:8px; padding:14px; font-size:14px; color:#e6edf3;">
                    {factor}
                </div>
                """, unsafe_allow_html=True)
    except ValueError:
        st.error("Please enter a valid numeric Container ID.")

st.markdown("---")

# ===============================================================
# SECTION 9 - OPERATIONAL INSIGHTS
# ===============================================================
section_header("⚡", "Operational Insights", "Hotspots and high-risk patterns across countries, HS codes, and shipping lines")

crit_df = df_all[df_all["Risk_Level"] == "Critical"].copy()

col1, col2, col3 = st.columns(3)

with col1:
    top_hr_countries = (
        crit_df.groupby("Origin_Country").size()
        .sort_values(ascending=False).head(12).reset_index()
    )
    top_hr_countries.columns = ["Country", "Critical_Count"]
    fig_hrc = px.bar(
        top_hr_countries, x="Critical_Count", y="Country",
        orientation="h", title="Top High-Risk Origin Countries",
        color="Critical_Count", color_continuous_scale="Reds",
    )
    fig_hrc.update_layout(**dark_layout(title_font_size=15,
                          yaxis=dict(autorange="reversed"),
                          coloraxis_showscale=False))
    st.plotly_chart(fig_hrc, use_container_width=True)

with col2:
    crit_df["HS_Chapter"] = (crit_df["HS_Code"] // 10000).astype(int)
    top_hs = (
        crit_df.groupby("HS_Chapter").size()
        .sort_values(ascending=False).head(12).reset_index()
    )
    top_hs.columns = ["HS_Chapter", "Critical_Count"]
    top_hs["HS_Chapter"] = "Ch. " + top_hs["HS_Chapter"].astype(str)
    fig_hs = px.bar(
        top_hs, x="Critical_Count", y="HS_Chapter",
        orientation="h", title="Top High-Risk HS Code Chapters",
        color="Critical_Count", color_continuous_scale="Oranges",
    )
    fig_hs.update_layout(**dark_layout(title_font_size=15,
                         yaxis=dict(autorange="reversed"),
                         coloraxis_showscale=False))
    st.plotly_chart(fig_hs, use_container_width=True)

with col3:
    top_sl = (
        crit_df.groupby("Shipping_Line").size()
        .sort_values(ascending=False).head(12).reset_index()
    )
    top_sl.columns = ["Shipping_Line", "Critical_Count"]
    fig_sl = px.bar(
        top_sl, x="Critical_Count", y="Shipping_Line",
        orientation="h", title="Top High-Risk Shipping Lines",
        color="Critical_Count", color_continuous_scale="Purples",
    )
    fig_sl.update_layout(**dark_layout(title_font_size=15,
                         yaxis=dict(autorange="reversed"),
                         coloraxis_showscale=False))
    st.plotly_chart(fig_sl, use_container_width=True)

# Risk rate heatmap: Origin Country x Destination Port
st.markdown("#### Risk Rate Heatmap - Origin Country x Destination Port")
heatmap_data = (
    df_all.groupby(["Origin_Country", "Destination_Port"])
    .apply(lambda x: (x["Risk_Level"] == "Critical").mean() * 100)
    .reset_index()
)
heatmap_data.columns = ["Origin_Country", "Destination_Port", "Critical_Rate_Pct"]

# Top 15 countries + ports by volume
top15_c = df_all["Origin_Country"].value_counts().head(15).index
top15_p = df_all["Destination_Port"].value_counts().head(15).index
hm_filt = heatmap_data[
    heatmap_data["Origin_Country"].isin(top15_c) &
    heatmap_data["Destination_Port"].isin(top15_p)
]
hm_pivot = hm_filt.pivot(index="Origin_Country",
                          columns="Destination_Port",
                          values="Critical_Rate_Pct").fillna(0)
fig_heat = px.imshow(
    hm_pivot,
    title="Critical Rate % (Top 15 Origins x Top 15 Destinations)",
    color_continuous_scale="Reds",
    labels={"color": "Critical Rate (%)"},
    aspect="auto",
)
fig_heat.update_layout(**PLOTLY_DARK, title_font_size=16, height=480)
st.plotly_chart(fig_heat, use_container_width=True)

st.markdown("---")

# ---------------------------------------------
# FOOTER
# ---------------------------------------------
st.markdown("""
<div style="text-align:center; padding:24px; color:#8b949e; font-size:13px;
            border-top:1px solid #21262d; margin-top:20px;">
    SmartContainer Risk Engine &nbsp;|&nbsp; HackaMined 2026
    &nbsp;|&nbsp; LightGBM + SMOTE + SHAP
    &nbsp;|&nbsp; Built with Streamlit + Plotly
</div>
""", unsafe_allow_html=True)
