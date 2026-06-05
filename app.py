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
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');

/* ── BASE ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 15px !important;
}

.stApp { background: #f0f2f6 !important; }

/* ── kill default streamlit top bar ── */
header[data-testid="stHeader"] { display: none !important; }

/* ── main content padding ── */
.main .block-container {
    padding: 1rem 1.6rem 1rem 1.6rem !important;
    max-width: 100% !important;
}

/* ══════════════════════════════════════
   SIDEBAR — always visible, no collapse
══════════════════════════════════════ */
div[data-testid="stSidebar"] {
    background: #111827 !important;
    border-right: 2px solid #1f2937 !important;
    min-width: 310px !important;
    width: 310px !important;
}

/* Hide every possible collapse / close button variant */
button[data-testid="baseButton-headerNoPadding"],
div[data-testid="collapsedControl"],
div[data-testid="stSidebarCollapseButton"],
button[aria-label="Close sidebar"],
button[aria-label="Collapse sidebar"],
button[title="Close sidebar"],
[class*="collapsedControl"] {
    display: none !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    pointer-events: none !important;
}

div[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

div[data-testid="stSidebar"] label {
    color: #9ca3af !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}

div[data-testid="stSidebar"] input,
div[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #1f2937 !important;
    border: 1px solid #374151 !important;
    border-radius: 8px !important;
    color: #f3f4f6 !important;
    font-size: 0.92rem !important;
    min-height: 38px !important;
}

div[data-testid="stSidebar"] input:focus,
div[data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within {
    border-color: #16a34a !important;
    box-shadow: 0 0 0 2px rgba(22,163,74,0.25) !important;
}

div[data-testid="stSidebar"] h1,
div[data-testid="stSidebar"] h2,
div[data-testid="stSidebar"] h3 {
    color: #f9fafb !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    margin: 4px 0 2px 0 !important;
}

div[data-testid="stSidebar"] hr {
    border-color: #374151 !important;
    margin: 10px 0 !important;
}

div[data-testid="stSidebar"] .stSelectbox,
div[data-testid="stSidebar"] .stNumberInput {
    margin-bottom: 8px !important;
}

/* Predict button */
div[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #16a34a 0%, #dc2626 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 1rem !important;
    width: 100% !important;
    box-shadow: 0 4px 18px rgba(22,163,74,0.35) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    margin-top: 4px !important;
}

div[data-testid="stSidebar"] .stButton > button:hover {
    box-shadow: 0 6px 24px rgba(22,163,74,0.5) !important;
    transform: translateY(-2px) !important;
    filter: brightness(1.08) !important;
}

/* ══════════════════════════════════════
   HERO HEADER — full-width inside main
   Uses a wide negative margin trick to
   escape block-container padding and
   span edge-to-edge
══════════════════════════════════════ */
.hero-wrap {
    margin: -1rem -1.6rem 1rem -1.6rem;
    padding: 0 1.6rem;
    background: linear-gradient(110deg, #0f172a 0%, #1e293b 45%, #0f172a 100%);
    border-bottom: 3px solid transparent;
    border-image: linear-gradient(90deg, #16a34a, #dc2626) 1;
    box-shadow: 0 4px 24px rgba(0,0,0,0.25);
    position: relative;
    overflow: hidden;
}

/* dot-grid texture */
.hero-wrap::before {
    content: '';
    position: absolute;
    inset: 0;
    background-image: radial-gradient(rgba(255,255,255,0.035) 1px, transparent 1px);
    background-size: 20px 20px;
    pointer-events: none;
}

.hero-inner {
    position: relative;
    z-index: 1;
    display: flex;
    align-items: center;
    justify-content: space-between;
    min-height: 72px;
    padding: 10px 0;
    gap: 20px;
}

.hero-brand {
    display: flex;
    align-items: center;
    gap: 16px;
}

.hero-icon {
    font-size: 2.4rem;
    line-height: 1;
    filter: drop-shadow(0 2px 6px rgba(0,0,0,0.45));
}

.hero-title {
    font-family: 'Sora', sans-serif;
    font-size: 1.35rem;
    font-weight: 800;
    color: #f8fafc;
    margin: 0 0 3px 0;
    letter-spacing: -0.02em;
    line-height: 1.15;
}

.hero-sub {
    font-size: 0.82rem;
    color: #94a3b8;
    margin: 0;
    font-weight: 400;
    letter-spacing: 0.01em;
}

.hero-stats {
    display: flex;
    align-items: center;
    gap: 22px;
}

.hstat {
    text-align: center;
}

.hstat-num {
    font-family: 'Sora', sans-serif;
    font-size: 1.25rem;
    font-weight: 800;
    color: #f1f5f9;
    display: block;
    line-height: 1;
}

.hstat-lbl {
    font-size: 0.66rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    display: block;
    margin-top: 3px;
    font-weight: 600;
}

.hstat-divider {
    width: 1px;
    height: 36px;
    background: rgba(100,116,139,0.35);
}

/* ══════════════════════════════════════
   SECTION LABELS
══════════════════════════════════════ */
.sec-label {
    font-family: 'Sora', sans-serif;
    font-size: 0.78rem;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0 0 8px 0;
    display: flex;
    align-items: center;
    gap: 6px;
}

.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #e2e8f0;
    margin-left: 6px;
}

/* ══════════════════════════════════════
   PATIENT SUMMARY CARD
══════════════════════════════════════ */
.summary-card {
    background: white;
    border-radius: 14px;
    padding: 14px 18px;
    box-shadow: 0 1px 10px rgba(0,0,0,0.08);
    border: 1px solid #f1f5f9;
}

/* dataframe text sizes */
[data-testid="stDataFrame"] * {
    font-size: 0.9rem !important;
}

[data-testid="stDataFrame"] th {
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    background: #f8fafc !important;
    color: #334155 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

[data-testid="stDataFrame"] td {
    color: #334155 !important;
}

/* ══════════════════════════════════════
   RESULT CARDS
══════════════════════════════════════ */
.result-card {
    border-radius: 14px;
    padding: 20px 22px;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.result-alive {
    background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
    box-shadow: 0 6px 24px rgba(22,163,74,0.35);
}

.result-dead {
    background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
    box-shadow: 0 6px 24px rgba(220,38,38,0.35);
}

.rc-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-weight: 700;
    color: white;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 10px;
}

.rc-icon {
    font-size: 2rem;
    display: block;
    margin-bottom: 6px;
    line-height: 1;
}

.rc-verdict {
    font-family: 'Sora', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: white;
    display: block;
    line-height: 1.1;
    margin-bottom: 5px;
}

.rc-sub {
    font-size: 0.82rem;
    color: rgba(255,255,255,0.82);
    display: block;
}

/* confidence row */
.conf-row {
    background: white;
    border-radius: 12px;
    padding: 14px 18px;
    margin-top: 10px;
    box-shadow: 0 1px 8px rgba(0,0,0,0.07);
    border: 1px solid #f1f5f9;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.conf-label {
    font-size: 0.76rem;
    color: #94a3b8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 2px;
}

.conf-value {
    font-family: 'Sora', sans-serif;
    font-size: 1.2rem;
    font-weight: 800;
    color: #0f172a;
}

.conf-alive {
    font-size: 0.95rem;
    font-weight: 700;
    color: #16a34a;
}

.conf-dead {
    font-size: 0.95rem;
    font-weight: 700;
    color: #dc2626;
}

/* ══════════════════════════════════════
   CHART CARDS
══════════════════════════════════════ */
.chart-card {
    background: white;
    border-radius: 14px;
    padding: 14px 16px 10px 16px;
    box-shadow: 0 1px 10px rgba(0,0,0,0.08);
    border: 1px solid #f1f5f9;
}

/* info box */
.stAlert {
    border-radius: 10px !important;
    border-left: 4px solid #16a34a !important;
    background: #f0fdf4 !important;
    font-size: 0.9rem !important;
}

/* caption */
.stCaption, [data-testid="stCaptionContainer"] {
    font-size: 0.8rem !important;
    color: #94a3b8 !important;
    margin-top: 6px !important;
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
    # Branded sidebar header
    st.markdown("""
    <div style="
        background: linear-gradient(135deg,#16a34a,#dc2626);
        padding: 16px 20px 14px;
        margin: -1px -1px 0;
        border-bottom: 1px solid #374151;
    ">
        <div style="font-family:'Sora',sans-serif;font-size:1.05rem;font-weight:800;
                    color:white;letter-spacing:-0.01em;margin-bottom:3px;">
            🎗️ BC Predictor
        </div>
        <div style="font-size:0.75rem;color:rgba(255,255,255,0.72);">
            Survival Analysis Dashboard
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    st.markdown("### 🤖 Model Selection")
    selected_model = st.selectbox("Algorithm", MODEL_NAMES, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### 🧬 Patient Input")

    survival_months    = st.number_input("Survival Months",       min_value=1,  max_value=120, value=40, step=1)
    tumor_size         = st.number_input("Tumor Size (mm)",        min_value=1,  max_value=200, value=25, step=1)
    reginol_node_pos   = st.number_input("Reginol Node Positive",  min_value=0,  max_value=50,  value=2,  step=1)
    regional_node_exam = st.number_input("Regional Node Examined", min_value=1,  max_value=60,  value=10, step=1)
    estrogen           = st.selectbox("Estrogen Status",           ["Positive", "Negative"])
    progesterone       = st.selectbox("Progesterone Status",       ["Positive", "Negative"])
    a_stage            = st.selectbox("A Stage",                   ["Regional", "Distant"])

    st.markdown("---")
    predict_btn = st.button("⚡  RUN PREDICTION", use_container_width=True)

    st.markdown("""
    <div style="margin-top:14px;padding:12px 16px;background:#1f2937;
                border-radius:10px;border:1px solid #374151;">
        <div style="font-size:0.72rem;color:#6b7280;text-transform:uppercase;
                    letter-spacing:0.09em;font-weight:700;margin-bottom:6px;">
            System Info
        </div>
        <div style="font-size:0.82rem;color:#9ca3af;line-height:1.7;">
            8 algorithms available<br>
            Binary classification<br>
            Real-time inference
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════
# HERO HEADER  (full-width via negative-margin escape)
# ══════════════════════════════
st.markdown(f"""
<div class="hero-wrap">
  <div class="hero-inner">
    <div class="hero-brand">
      <div class="hero-icon">🎗️</div>
      <div>
        <div class="hero-title">Breast Cancer Survival Predictor</div>
        <div class="hero-sub">
          ML-powered clinical decision support &nbsp;·&nbsp;
          8 Ensemble Models &nbsp;·&nbsp; Real-time Inference
        </div>
      </div>
    </div>
    <div class="hero-stats">
      <div class="hstat">
        <span class="hstat-num">8</span>
        <span class="hstat-lbl">Models</span>
      </div>
      <div class="hstat-divider"></div>
      <div class="hstat">
        <span class="hstat-num">{len(features)}</span>
        <span class="hstat-lbl">Features</span>
      </div>
      <div class="hstat-divider"></div>
      <div class="hstat">
        <span class="hstat-num">Binary</span>
        <span class="hstat-lbl">Output</span>
      </div>
      <div class="hstat-divider"></div>
      <div style="font-size:2.5rem;opacity:0.65;">🏥</div>
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
CHART_SIZE = (5.4, 2.6)

def chart_confidence(alive_p, dead_p, model_name):
    fig, ax = plt.subplots(figsize=CHART_SIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    vals   = [dead_p, alive_p]
    labels = ["Dead", "Alive"]
    colors = ["#dc2626", "#16a34a"]

    bars = ax.barh(labels, vals, color=colors, height=0.44, edgecolor="none")

    for bar, val in zip(bars, vals):
        ax.text(val + 1.2, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10,
                fontweight="bold", color="#1e293b")

    ax.set_xlim(0, 122)
    ax.set_xlabel("Probability (%)", fontsize=9, color="#94a3b8")
    ax.set_title(f"Prediction Confidence — {model_name}",
                 fontsize=10, color="#334155", fontweight="bold", pad=8)
    ax.tick_params(labelsize=9, colors="#64748b")
    ax.set_ylim(-1.0, 2.0)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, color="#f1f5f9", linewidth=0.9)
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

    df  = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=CHART_SIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(df))
    w = 0.33

    ax.bar(x - w/2, df["Alive %"], width=w, color="#16a34a",
           label="Alive", edgecolor="none", alpha=0.9)
    ax.bar(x + w/2, df["Dead %"],  width=w, color="#dc2626",
           label="Dead",  edgecolor="none", alpha=0.9)

    idx = MODEL_NAMES.index(selected_model)
    ax.axvspan(idx - 0.5, idx + 0.5, color="#f0fdf4", alpha=0.85, zorder=0)
    ax.annotate("▲ selected", xy=(idx, 104), ha="center",
                fontsize=7.5, color="#16a34a", fontweight="700")

    short = [n.replace(" ", "\n") for n in df["Model"]]
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=8, color="#64748b", ha="center")
    ax.set_ylabel("Probability (%)", fontsize=9, color="#94a3b8")
    ax.set_title("All Models Comparison",
                 fontsize=10, color="#334155", fontweight="bold", pad=8)
    ax.set_ylim(0, 120)
    ax.legend(fontsize=9, framealpha=0, loc="upper right")

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.yaxis.grid(True, color="#f1f5f9", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=9, colors="#94a3b8")
    plt.tight_layout(pad=0.5)
    return fig


# ══════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════
col_left, col_right = st.columns([1.5, 1], gap="large")

# ── Patient Summary ──
with col_left:
    st.markdown('<p class="sec-label">📋 Patient Summary</p>', unsafe_allow_html=True)
    node_ratio = reginol_node_pos / max(regional_node_exam, 1)

    summary = pd.DataFrame({
        "Feature": ["Survival Months", "Tumor Size (mm)", "Node Positive",
                    "Node Examined",   "Node Ratio",
                    "Estrogen",        "Progesterone",    "A Stage"],
        "Value":   [str(survival_months), str(tumor_size),
                    str(reginol_node_pos), str(regional_node_exam),
                    f"{node_ratio:.4f}", estrogen, progesterone, a_stage]
    })

    st.markdown('<div class="summary-card">', unsafe_allow_html=True)
    st.dataframe(summary, use_container_width=True, hide_index=True, height=260)
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption(f"Active model: **{selected_model}**  ·  Features used: {len(features)}")

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
# CHARTS ROW
# ══════════════════════════════
if predict_btn:
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")

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
