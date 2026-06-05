import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

st.set_page_config(
    page_title="Breast Cancer Survival Predictor",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* ═══════════════════════════════════════════
   GOOGLE FONTS
═══════════════════════════════════════════ */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Sora:wght@400;600;700;800&display=swap');

/* ═══════════════════════════════════════════
   GLOBAL RESET & BASE
═══════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 13px;
}

/* Kill default Streamlit padding */
.main .block-container {
    padding: 0.6rem 1.2rem 0.5rem 1.2rem !important;
    max-width: 100% !important;
    margin-top: 70px !important;   /* push content below fixed header */
}

/* Hide native Streamlit top bar */
header[data-testid="stHeader"] {
    height: 0 !important;
    display: none !important;
}

.stApp {
    background: #f0f2f5 !important;
}

/* ═══════════════════════════════════════════
   FIXED FULL-VIEWPORT HEADER OVERLAY
   (sits above sidebar AND main content)
═══════════════════════════════════════════ */
.fixed-hero-wrapper {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 9999;
    padding: 8px 14px 0px 14px;
    pointer-events: none;   /* let clicks pass through to sidebar toggle */
}

.fixed-hero-wrapper .hero-header {
    pointer-events: all;
}

/* ═══════════════════════════════════════════
   SIDEBAR — Dark Charcoal Premium (STATIC)
═══════════════════════════════════════════ */
div[data-testid="stSidebar"] {
    background: #111827 !important;
    border-right: 1px solid #1f2937 !important;
    width: 300px !important;
    min-width: 300px !important;
    /* Push sidebar content below the fixed header */
    padding-top: 70px !important;
}

div[data-testid="stSidebar"] > div {
    padding: 0 !important;
}

div[data-testid="stSidebar"] .sidebar-content {
    padding: 0 !important;
}

/* Hide the collapse/toggle arrow button — makes sidebar static */
div[data-testid="stSidebar"] div[data-testid="collapsedControl"],
button[data-testid="baseButton-headerNoPadding"],
div[data-testid="stSidebarCollapseButton"],
section[data-testid="stSidebar"] > div > div:first-child > div > button,
.st-emotion-cache-1cypcdb,
div[class*="collapsedControl"],
button[aria-label="Close sidebar"],
button[aria-label="Collapse sidebar"] {
    display: none !important;
    visibility: hidden !important;
    pointer-events: none !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
}

/* All sidebar text */
div[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Sidebar labels */
div[data-testid="stSidebar"] label,
div[data-testid="stSidebar"] .stSelectbox label,
div[data-testid="stSidebar"] .stNumberInput label {
    color: #9ca3af !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    margin-bottom: 2px !important;
}

/* Sidebar inputs */
div[data-testid="stSidebar"] input,
div[data-testid="stSidebar"] .stSelectbox > div > div,
div[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #1f2937 !important;
    border: 1px solid #374151 !important;
    border-radius: 8px !important;
    color: #f3f4f6 !important;
    font-size: 0.82rem !important;
    padding: 5px 10px !important;
    min-height: 34px !important;
}

div[data-testid="stSidebar"] input:focus,
div[data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within {
    border-color: #16a34a !important;
    box-shadow: 0 0 0 2px rgba(22,163,74,0.2) !important;
}

/* Sidebar number input buttons */
div[data-testid="stSidebar"] button[kind="secondary"] {
    background: #374151 !important;
    border: none !important;
    color: #d1d5db !important;
    border-radius: 6px !important;
}

/* Sidebar markdown headings */
div[data-testid="stSidebar"] h1,
div[data-testid="stSidebar"] h2,
div[data-testid="stSidebar"] h3 {
    color: #f9fafb !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    margin: 0 !important;
}

/* Sidebar divider */
div[data-testid="stSidebar"] hr {
    border-color: #1f2937 !important;
    margin: 8px 0 !important;
}

/* PREDICT BUTTON */
div[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #16a34a 0%, #dc2626 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1rem !important;
    width: 100% !important;
    box-shadow: 0 4px 15px rgba(22,163,74,0.3) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
}

div[data-testid="stSidebar"] .stButton > button:hover {
    box-shadow: 0 6px 22px rgba(22,163,74,0.45) !important;
    transform: translateY(-1px) !important;
    filter: brightness(1.08) !important;
}

div[data-testid="stSidebar"] .stButton > button:active {
    transform: translateY(0px) !important;
}

/* Sidebar element spacing */
div[data-testid="stSidebar"] .stSelectbox,
div[data-testid="stSidebar"] .stNumberInput {
    margin-bottom: 6px !important;
}

/* ═══════════════════════════════════════════
   FULL-WIDTH HEADER — Fixed, overlaps sidebar
═══════════════════════════════════════════ */
.hero-header {
    background: linear-gradient(110deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border-radius: 12px;
    padding: 0 32px;
    margin-bottom: 0;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 28px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.06);
    position: relative;
    overflow: hidden;
    min-height: 62px;
    border: 1px solid #1e293b;
}

/* accent left stripe */
.hero-header::after {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 4px;
    background: linear-gradient(180deg, #16a34a 0%, #dc2626 100%);
    border-radius: 14px 0 0 14px;
}

/* subtle dot pattern overlay */
.hero-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image: radial-gradient(circle, rgba(255,255,255,0.03) 1px, transparent 1px);
    background-size: 22px 22px;
    pointer-events: none;
}

.hero-left {
    display: flex;
    align-items: center;
    gap: 14px;
    position: relative;
    z-index: 1;
}

.hero-ribbon {
    font-size: 1.9rem;
    line-height: 1;
    filter: drop-shadow(0 2px 6px rgba(0,0,0,0.4));
}

.hero-text h1 {
    font-family: 'Sora', sans-serif !important;
    font-size: 1.18rem !important;
    font-weight: 800 !important;
    color: #f8fafc !important;
    margin: 0 0 3px 0 !important;
    letter-spacing: -0.01em;
    line-height: 1.2;
}

.hero-text p {
    font-size: 0.75rem;
    color: rgba(148,163,184,0.9);
    margin: 0;
    font-weight: 400;
    letter-spacing: 0.01em;
}

.hero-right {
    display: flex;
    align-items: center;
    gap: 20px;
    position: relative;
    z-index: 1;
}

.hero-stat {
    text-align: center;
}

.hero-stat .stat-num {
    font-family: 'Sora', sans-serif;
    font-size: 1.05rem;
    font-weight: 800;
    display: block;
    line-height: 1;
    color: #f1f5f9;
}

.hero-stat .stat-lbl {
    font-size: 0.62rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    display: block;
    margin-top: 3px;
    font-weight: 600;
}

.hero-divider {
    width: 1px;
    height: 32px;
    background: rgba(100,116,139,0.4);
}

/* ═══════════════════════════════════════════
   SECTION LABELS
═══════════════════════════════════════════ */
.sec-label {
    font-family: 'Sora', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0 0 6px 0;
    display: flex;
    align-items: center;
    gap: 5px;
}

.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #e5e7eb;
    margin-left: 6px;
}

/* ═══════════════════════════════════════════
   PATIENT SUMMARY CARD
═══════════════════════════════════════════ */
.summary-card {
    background: white;
    border-radius: 12px;
    padding: 12px 16px;
    box-shadow: 0 1px 8px rgba(0,0,0,0.07);
    border: 1px solid #f3f4f6;
}

/* Compact dataframe */
.stDataFrame {
    border-radius: 8px !important;
    overflow: hidden !important;
}

.stDataFrame table {
    font-size: 0.78rem !important;
}

.stDataFrame thead th {
    background: #f9fafb !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    color: #374151 !important;
    padding: 5px 10px !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.stDataFrame tbody td {
    padding: 4px 10px !important;
    font-size: 0.78rem !important;
    color: #4b5563 !important;
    border-bottom: 1px solid #f9fafb !important;
}

/* ═══════════════════════════════════════════
   RESULT CARDS
═══════════════════════════════════════════ */
.result-card {
    border-radius: 12px;
    padding: 14px 18px;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.result-alive {
    background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
    box-shadow: 0 4px 20px rgba(22,163,74,0.3);
}

.result-dead {
    background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
    box-shadow: 0 4px 20px rgba(220,38,38,0.3);
}

.result-card .rc-icon {
    font-size: 1.6rem;
    display: block;
    margin-bottom: 4px;
    line-height: 1;
}

.result-card .rc-verdict {
    font-family: 'Sora', sans-serif;
    font-size: 1.3rem;
    font-weight: 800;
    color: white;
    display: block;
    line-height: 1.1;
    margin-bottom: 4px;
}

.result-card .rc-sub {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.82);
    display: block;
}

.result-card .rc-badge {
    display: inline-block;
    background: rgba(255,255,255,0.2);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.68rem;
    font-weight: 700;
    color: white;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 8px;
}

/* Confidence pill row */
.conf-row {
    background: white;
    border-radius: 10px;
    padding: 10px 14px;
    margin-top: 8px;
    box-shadow: 0 1px 6px rgba(0,0,0,0.07);
    border: 1px solid #f3f4f6;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
}

.conf-label {
    font-size: 0.72rem;
    color: #6b7280;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.conf-value {
    font-family: 'Sora', sans-serif;
    font-size: 1rem;
    font-weight: 800;
    color: #111827;
}

.conf-alive {
    font-size: 0.8rem;
    font-weight: 700;
    color: #16a34a;
}

.conf-dead {
    font-size: 0.8rem;
    font-weight: 700;
    color: #dc2626;
}

/* ═══════════════════════════════════════════
   CHART CONTAINERS
═══════════════════════════════════════════ */
.chart-card {
    background: white;
    border-radius: 12px;
    padding: 10px 14px 8px 14px;
    box-shadow: 0 1px 8px rgba(0,0,0,0.07);
    border: 1px solid #f3f4f6;
}

/* Info box */
.stAlert {
    border-radius: 10px !important;
    border-left: 3px solid #16a34a !important;
    background: #f0fdf4 !important;
    padding: 8px 14px !important;
    font-size: 0.8rem !important;
}

/* Caption */
.stCaption {
    font-size: 0.7rem !important;
    color: #9ca3af !important;
    margin-top: 4px !important;
}

/* Remove extra streamlit spacers */
div.stVerticalBlock > div[style] {
    gap: 0 !important;
}

.element-container {
    margin-bottom: 0 !important;
}

</style>
""", unsafe_allow_html=True)

# ── LOAD PKL ──
PKL_PATH = "breast_cancer_models.pkl"

@st.cache_resource
def load_bundle(path):
    return joblib.load(path)

if not os.path.exists(PKL_PATH):
    st.error(f"❌ `{PKL_PATH}` not found. Place it in the same directory.")
    st.stop()

bundle        = load_bundle(PKL_PATH)
scaler        = bundle["scaler"]
features      = bundle["features"]
label_encoder = bundle["label_encoder"]

MODEL_NAMES   = ["Logistic Regression", "KNN", "Random Forest", "Decision Tree",
                 "SVM", "Gradient Boosting", "Naive Bayes", "XGBoost"]
SCALED_MODELS = {"Logistic Regression", "KNN", "SVM", "Naive Bayes"}

# ══════════════════════════════
# SIDEBAR
# ══════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#16a34a,#dc2626);padding:14px 18px 12px;margin:-1rem -1rem 0;border-bottom:1px solid #1f2937;">
        <div style="font-family:'Sora',sans-serif;font-size:0.95rem;font-weight:800;color:white;letter-spacing:-0.01em;">🎗️ BC Predictor</div>
        <div style="font-size:0.68rem;color:rgba(255,255,255,0.75);margin-top:2px;">Survival Analysis Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("### 🤖 Model")
    selected_model = st.selectbox("Algorithm", MODEL_NAMES, label_visibility="collapsed")

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    st.markdown("### 🧬 Patient Input")

    survival_months    = st.number_input("Survival Months",        min_value=1,  max_value=120, value=40, step=1)
    tumor_size         = st.number_input("Tumor Size (mm)",         min_value=1,  max_value=200, value=25, step=1)
    reginol_node_pos   = st.number_input("Reginol Node Positive",   min_value=0,  max_value=50,  value=2,  step=1)
    regional_node_exam = st.number_input("Regional Node Examined",  min_value=1,  max_value=60,  value=10, step=1)
    estrogen           = st.selectbox("Estrogen Status",            ["Positive", "Negative"])
    progesterone       = st.selectbox("Progesterone Status",        ["Positive", "Negative"])
    a_stage            = st.selectbox("A Stage",                    ["Regional", "Distant"])

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    predict_btn = st.button("⚡  RUN PREDICTION", use_container_width=True)

    st.markdown("""
    <div style="margin-top:12px;padding:10px 14px;background:#1f2937;border-radius:10px;border:1px solid #374151;">
        <div style="font-size:0.68rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.08em;font-weight:700;margin-bottom:6px;">Model Info</div>
        <div style="font-size:0.75rem;color:#9ca3af;">8 algorithms available<br>Multi-class binary output<br>Real-time inference</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════
# HERO HEADER (full-width)
# ══════════════════════════════
st.markdown(f"""
<div class="fixed-hero-wrapper">
<div class="hero-header">
    <div class="hero-left">
        <div class="hero-ribbon">🎗️</div>
        <div class="hero-text">
            <h1>Breast Cancer Survival Predictor</h1>
            <p>ML-powered clinical decision support &nbsp;·&nbsp; 8 Ensemble Models &nbsp;·&nbsp; Real-time Inference</p>
        </div>
    </div>
    <div class="hero-right">
        <div class="hero-stat">
            <span class="stat-num">8</span>
            <span class="stat-lbl">Models</span>
        </div>
        <div class="hero-divider"></div>
        <div class="hero-stat">
            <span class="stat-num">{len(features)}</span>
            <span class="stat-lbl">Features</span>
        </div>
        <div class="hero-divider"></div>
        <div class="hero-stat">
            <span class="stat-num">Binary</span>
            <span class="stat-lbl">Output</span>
        </div>
        <div class="hero-divider"></div>
        <div style="font-size:2.2rem;opacity:0.7;">🏥</div>
    </div>
</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════
# FEATURE BUILDERS
# ══════════════════════════════
def build_input():
    node_ratio = reginol_node_pos / max(regional_node_exam, 1)
    mapping = {
        "Survival Months":        survival_months,
        "Tumor Size":             tumor_size,
        "Reginol Node Positive":  reginol_node_pos,
        "Regional Node Examined": regional_node_exam,
        "Node_Positive_Ratio":    node_ratio,
        "Estrogen Status":        0 if estrogen == "Positive" else 1,
        "Progesterone Status":    0 if progesterone == "Positive" else 1,
        "A Stage":                0 if a_stage == "Regional" else 1,
    }
    row = {f: mapping[f] for f in features}
    return pd.DataFrame([row])[features]

def get_input_for_model(model_name):
    X = build_input()
    return scaler.transform(X) if model_name in SCALED_MODELS else X.values


# ══════════════════════════════
# CHARTS
# ══════════════════════════════
CHART_SIZE = (4.8, 2.2)   # ← single source of truth for both charts

def chart_confidence(alive_p, dead_p, model_name):
    fig, ax = plt.subplots(figsize=CHART_SIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    vals   = [dead_p, alive_p]
    labels = ["Dead", "Alive"]
    colors = ["#dc2626", "#16a34a"]

    bars = ax.barh(labels, vals, color=colors, height=0.42, edgecolor="none")

    for bar, val in zip(bars, vals):
        ax.text(val + 1.2, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=8.5,
                fontweight="bold", color="#1f2937")

    ax.set_xlim(0, 118)
    ax.set_xlabel("Probability (%)", fontsize=7, color="#9ca3af")
    ax.set_title(f"Confidence — {model_name}", fontsize=8, color="#374151",
                 fontweight="bold", pad=6)
    ax.tick_params(labelsize=7.5, colors="#6b7280")

    # Vertical padding so bars sit in the same canvas space as comparison chart
    ax.set_ylim(-1.2, 2.2)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, color="#f3f4f6", linewidth=0.8)
    ax.set_axisbelow(True)

    plt.tight_layout(pad=0.5)
    return fig


def chart_model_comparison():
    results = []
    for name in MODEL_NAMES:
        m  = bundle[name]
        Xi = get_input_for_model(name)
        if hasattr(m, "predict_proba"):
            p = m.predict_proba(Xi)[0]
            results.append({"Model": name, "Alive %": p[0]*100, "Dead %": p[1]*100})
        else:
            pred = m.predict(Xi)[0]
            results.append({"Model": name,
                             "Alive %": 100 if pred == 0 else 0,
                             "Dead %":  100 if pred == 1 else 0})

    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=CHART_SIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(df))
    w = 0.32

    ax.bar(x - w/2, df["Alive %"], width=w, color="#16a34a",
           label="Alive", edgecolor="none", alpha=0.88)
    ax.bar(x + w/2, df["Dead %"],  width=w, color="#dc2626",
           label="Dead",  edgecolor="none", alpha=0.88)

    # Highlight selected
    idx = MODEL_NAMES.index(selected_model)
    ax.axvspan(idx - 0.5, idx + 0.5, color="#f0fdf4", alpha=0.9, zorder=0)

    short_names = [n.replace(" ", "\n") for n in df["Model"]]
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=6.5, color="#6b7280", ha="center")
    ax.set_ylabel("Probability (%)", fontsize=7, color="#9ca3af")
    ax.set_title("All Models Comparison", fontsize=8, color="#374151",
                 fontweight="bold", pad=6)
    ax.set_ylim(0, 118)
    ax.legend(fontsize=7, framealpha=0, loc="upper right")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.yaxis.grid(True, color="#f3f4f6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=7, colors="#9ca3af")

    plt.tight_layout(pad=0.4)
    return fig


# ══════════════════════════════
# MAIN LAYOUT — 2 columns top
# ══════════════════════════════
col_left, col_right = st.columns([1.45, 1], gap="medium")

# ── Patient Summary ──
with col_left:
    st.markdown('<p class="sec-label">📋 Patient Summary</p>', unsafe_allow_html=True)
    node_ratio = reginol_node_pos / max(regional_node_exam, 1)

    summary = pd.DataFrame({
        "Feature": ["Survival Months", "Tumor Size (mm)", "Node Positive",
                    "Node Examined", "Node Ratio",
                    "Estrogen", "Progesterone", "A Stage"],
        "Value": [str(survival_months), str(tumor_size),
                  str(reginol_node_pos), str(regional_node_exam),
                  f"{node_ratio:.4f}", estrogen, progesterone, a_stage]
    })

    st.markdown('<div class="summary-card">', unsafe_allow_html=True)
    st.dataframe(summary, use_container_width=True, hide_index=True, height=232)
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption(f"Active model: **{selected_model}**  ·  Features: {len(features)}")

# ── Prediction Result ──
with col_right:
    st.markdown('<p class="sec-label">🤖 Prediction Result</p>', unsafe_allow_html=True)

    if predict_btn:
        model   = bundle[selected_model]
        X       = build_input()
        X_input = scaler.transform(X) if selected_model in SCALED_MODELS else X.values

        pred_encoded = model.predict(X_input)[0]
        pred_label   = label_encoder.inverse_transform([pred_encoded])[0]

        alive_p, dead_p, confidence = 50.0, 50.0, 50.0
        has_proba = hasattr(model, "predict_proba")
        if has_proba:
            proba      = model.predict_proba(X_input)[0]
            confidence = proba[pred_encoded] * 100
            alive_p    = proba[0] * 100
            dead_p     = proba[1] * 100

        if pred_label == "Alive":
            st.markdown(f"""
            <div class="result-card result-alive">
                <span class="rc-badge">{selected_model}</span>
                <span class="rc-icon">✅</span>
                <span class="rc-verdict">Likely Alive</span>
                <span class="rc-sub">Model predicts patient survival</span>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-dead">
                <span class="rc-badge">{selected_model}</span>
                <span class="rc-icon">⚠️</span>
                <span class="rc-verdict">High Risk</span>
                <span class="rc-sub">Model predicts elevated mortality risk</span>
            </div>""", unsafe_allow_html=True)

        if has_proba:
            st.markdown(f"""
            <div class="conf-row">
                <div>
                    <div class="conf-label">Confidence</div>
                    <div class="conf-value">{confidence:.1f}%</div>
                </div>
                <div style="text-align:center;">
                    <div class="conf-label">Alive</div>
                    <div class="conf-alive">🟢 {alive_p:.1f}%</div>
                </div>
                <div style="text-align:center;">
                    <div class="conf-label">Dead</div>
                    <div class="conf-dead">🔴 {dead_p:.1f}%</div>
                </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("👈 Configure patient details in the sidebar and click **RUN PREDICTION**.")
        alive_p, dead_p = 50.0, 50.0

# ══════════════════════════════
# CHARTS ROW (below summary)
# ══════════════════════════════
if predict_btn:
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="medium")

    with c1:
        st.markdown('<p class="sec-label">📊 Prediction Confidence</p>', unsafe_allow_html=True)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.pyplot(chart_confidence(alive_p, dead_p, selected_model), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<p class="sec-label">📊 All Models Comparison</p>', unsafe_allow_html=True)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.pyplot(chart_model_comparison(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
