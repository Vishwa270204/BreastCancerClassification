import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec

st.set_page_config(
    page_title="Breast Cancer Survival Predictor",
    page_icon="🎗️",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Outfit:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

#MainMenu, footer, header { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

.stApp { background: #f5eded; }
.block-container {
    padding-top: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    max-width: 100% !important;
}

/* ── Header ── */
.header-bar {
    background: linear-gradient(90deg, #6b0f1a 0%, #b91c1c 55%, #dc2626 100%);
    padding: 16px 36px;
    display: flex;
    align-items: center;
    gap: 18px;
    box-shadow: 0 4px 24px rgba(107,15,26,0.3);
}
.header-bar h1 {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem; color: white; margin: 0;
}
.header-bar p { font-size: 0.77rem; color: rgba(255,255,255,0.72); margin: 0; }

/* ── Input Zone ── */
.input-zone-wrapper {
    background: linear-gradient(135deg, #1a0305 0%, #2d0608 60%, #1a0305 100%);
    border-bottom: 3px solid #7f1d1d;
    padding: 0px 36px 16px 36px;
    box-shadow: 0 6px 24px rgba(107,15,26,0.35);
    position: relative;
}
.input-zone-wrapper::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #ef4444, transparent);
}
.input-section-label {
    font-family: 'Playfair Display', serif;
    color: #1e3a8a;
    font-size: 1rem;
    font-weight: 700;
    text-transform: uppercase;
    padding: 10px 0 6px 0;
    border-bottom: 1px solid rgba(252,165,165,0.15);
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.input-zone-wrapper label {
    color: #fca5a5 !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}
.input-zone-wrapper p,
.input-zone-wrapper span,
.input-zone-wrapper div { color: #f0dada !important; }
.input-zone-wrapper input,
.input-zone-wrapper [data-baseweb="select"] > div {
    background-color: #ffffff !important;
    border: 2px solid #d4d4d4 !important;
    color: #333333 !important;
    border-radius: 8px !important;
}

.input-zone-wrapper input:focus,
.input-zone-wrapper [data-baseweb="select"] > div:focus {
    border-color: #2563eb !important;
}
.input-zone-wrapper [data-baseweb="select"] > div {
    background-color: #3d0a0a !important;
    border: 1px solid #7f1d1d !important;
    color: #fef2f2 !important;
    border-radius: 6px !important;
}
.input-zone-wrapper [data-baseweb="select"] svg { fill: #fca5a5 !important; }
.input-zone-wrapper button[data-testid="stNumberInputStepDown"],
.input-zone-wrapper button[data-testid="stNumberInputStepUp"] {
    background: #4a1010 !important;
    border-color: #7f1d1d !important;
    color: #fca5a5 !important;
}
.input-zone-wrapper .stButton > button {
    background: linear-gradient(135deg, #991b1b, #ef4444) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 700 !important;
    font-size: 0.9rem !important;
    box-shadow: 0 4px 18px rgba(239,68,68,0.5) !important;
    width: 100% !important;
    height: 44px !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    transition: all 0.2s ease !important;
}
.input-zone-wrapper .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 22px rgba(239,68,68,0.6) !important;
}
.row-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(252,165,165,0.2), transparent);
    margin: 4px 0 10px 0;
}

/* ── Main content ── */
.main-zone {
    padding: 20px 36px 36px 36px;
    background: #f5eded;
}

/* ── Section titles ── */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem; color: #6b0f1a;
    border-bottom: 1px solid #dcc5c5;
    padding-bottom: 5px; margin: 0 0 12px 0;
}

/* ── Result cards ── */
.result-alive {
    background: linear-gradient(135deg, #14532d, #15803d);
    border-radius: 12px; padding: 22px 26px;
    color: white; text-align: center;
}
.result-dead {
    background: linear-gradient(135deg, #6b0f1a, #b91c1c);
    border-radius: 12px; padding: 22px 26px;
    color: white; text-align: center;
}
.result-alive h2, .result-dead h2 {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem; margin: 0 0 6px 0; color: white;
}
.result-alive p, .result-dead p { font-size: 0.86rem; opacity: 0.88; margin: 0; }
.model-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.3);
    color: white; border-radius: 20px;
    padding: 2px 12px; font-size: 0.7rem;
    font-weight: 600; margin-bottom: 8px;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.prob-card {
    background: white; border-radius: 10px;
    padding: 13px 18px; margin-top: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    text-align: center; font-size: 0.9rem; color: #333;
}
.chart-wrap {
    background: white; border-radius: 12px;
    padding: 12px 10px 8px 10px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    margin-bottom: 14px;
}
.chart-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-top: 16px;
}
</style>
""", unsafe_allow_html=True)

# ── LOAD PKL ──
PKL_PATH = "breast_cancer_models.pkl"

@st.cache_resource
def load_bundle(path):
    return joblib.load(path)

if not os.path.exists(PKL_PATH):
    st.error(f"❌ `{PKL_PATH}` not found.")
    st.stop()

bundle        = load_bundle(PKL_PATH)
scaler        = bundle["scaler"]
features      = bundle["features"]
label_encoder = bundle["label_encoder"]

MODEL_NAMES   = ["Logistic Regression", "KNN", "Random Forest", "Decision Tree",
                 "SVM", "Gradient Boosting", "Naive Bayes", "XGBoost"]
SCALED_MODELS = {"Logistic Regression", "KNN", "SVM", "Naive Bayes"}

# ── HEADER ──
st.markdown("""
<div class="header-bar">
    <span style="font-size:2rem">🎗️</span>
    <div>
        <h1>Breast Cancer Survival Predictor</h1>
        <p>Clinical ML Dashboard — Multi-Model Analysis</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ── INPUT ZONE (2 rows) ──
st.markdown('<div class="input-zone-wrapper">', unsafe_allow_html=True)

# Row label
st.markdown('<div class="section-title">Patient Input Parameters</div>', unsafe_allow_html=True)

# ROW 1 — Numerical inputs
r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns([1.4, 1.2, 1.2, 1.2, 1.2])
with r1c1: selected_model     = st.selectbox("Model", MODEL_NAMES)
with r1c2: survival_months    = st.number_input("Survival Months", step=1, min_value=0)
with r1c3: tumor_size         = st.number_input("Tumor Size (mm)", step=1, min_value=0)
with r1c4: reginol_node_pos   = st.number_input("Node Positive",   step=1, min_value=0)
with r1c5: regional_node_exam = st.number_input("Node Examined",   step=1, min_value=0)

st.markdown('<div class="row-divider"></div>', unsafe_allow_html=True)

# ROW 2 — Categorical inputs + predict
r2c1, r2c2, r2c3, r2c4 = st.columns([1.2, 1.2, 1.2, 0.8])
with r2c1: estrogen      = st.selectbox("Estrogen Status",    ["Positive", "Negative"])
with r2c2: progesterone  = st.selectbox("Progesterone Status",["Positive", "Negative"])
with r2c3: a_stage       = st.selectbox("A Stage",            ["Regional", "Distant"])
with r2c4:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Run Prediction", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── HELPERS ──
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
    return pd.DataFrame([{f: mapping[f] for f in features}])[features]

def get_input_for_model(name):
    X = build_input()
    return scaler.transform(X) if name in SCALED_MODELS else X.values

CHART_SIZE = (3.8, 2.6)

# ─── Chart 1: Confidence bar (horizontal) ───
def chart_confidence(alive_p, dead_p, model_name):
    fig, ax = plt.subplots(figsize=CHART_SIZE)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bars = ax.barh(["Dead", "Alive"], [dead_p, alive_p],
                   color=["#b91c1c", "#15803d"], height=0.42, edgecolor="none")
    for bar, val in zip(bars, [dead_p, alive_p]):
        ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=9, fontweight="bold", color="#333")
    ax.set_xlim(0, 125)
    ax.set_xlabel("Probability (%)", fontsize=7, color="#aaa")
    ax.set_title(f"Prediction Confidence\n{model_name}", fontsize=8,
                 color="#6b0f1a", fontweight="bold", pad=8)
    ax.tick_params(colors="#777", labelsize=8)
    for sp in ax.spines.values(): sp.set_visible(False)
    plt.tight_layout(pad=0.8)
    return fig

# ─── Chart 2: All Models comparison ───
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
    x, w = np.arange(len(df)), 0.35
    ax.bar(x - w/2, df["Alive %"], width=w, color="#15803d", label="Alive", edgecolor="none")
    ax.bar(x + w/2, df["Dead %"],  width=w, color="#b91c1c", label="Dead",  edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n") for n in df["Model"]], fontsize=5.5, color="#444")
    ax.set_ylabel("Probability (%)", fontsize=7, color="#aaa")
    ax.set_title("All Models — Alive vs Dead", fontsize=8,
                 color="#6b0f1a", fontweight="bold", pad=8)
    ax.set_ylim(0, 120)
    ax.legend(fontsize=7, framealpha=0, loc="upper right")
    ax.tick_params(colors="#777", labelsize=6)
    for sp in ax.spines.values(): sp.set_visible(False)
    idx = MODEL_NAMES.index(selected_model)
    ax.axvspan(idx - 0.5, idx + 0.5, color="#6b0f1a", alpha=0.08, zorder=0)
    plt.tight_layout(pad=0.8)
    return fig


# ── MAIN ZONE ──
st.markdown('<div class="main-zone">', unsafe_allow_html=True)

node_ratio = reginol_node_pos / max(regional_node_exam, 1)
col1, col2 = st.columns([1.2, 1], gap="large")

with col1:
    st.markdown('<div class="section-title">📋 Patient Summary</div>', unsafe_allow_html=True)
    summary = pd.DataFrame({
        "Feature": ["Survival Months", "Tumor Size (mm)", "Node Positive",
                    "Node Examined", "Node Positive Ratio",
                    "Estrogen Status", "Progesterone Status", "A Stage"],
        "Value":   [str(survival_months), str(tumor_size), str(reginol_node_pos),
                    str(regional_node_exam), f"{node_ratio:.4f}",
                    estrogen, progesterone, a_stage]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True, height=318)

with col2:
    st.markdown('<div class="section-title">🤖 Prediction Result</div>', unsafe_allow_html=True)
    alive_p, dead_p = 50.0, 50.0

    if predict_btn:
        model        = bundle[selected_model]
        X_input      = get_input_for_model(selected_model)
        pred_encoded = model.predict(X_input)[0]
        pred_label   = label_encoder.inverse_transform([pred_encoded])[0]

        if pred_label == "Alive":
            st.markdown(f"""<div class="result-alive">
                <div class="model-badge">{selected_model}</div>
                <h2>✅ Alive</h2>
                <p>The model predicts the patient is likely to survive.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="result-dead">
                <div class="model-badge">{selected_model}</div>
                <h2>⚠️ Dead</h2>
                <p>The model predicts a higher risk of mortality.</p>
            </div>""", unsafe_allow_html=True)

        if hasattr(model, "predict_proba"):
            proba      = model.predict_proba(X_input)[0]
            confidence = proba[pred_encoded] * 100
            alive_p    = proba[0] * 100
            dead_p     = proba[1] * 100
            st.markdown(f"""<div class="prob-card">
                <strong>Confidence: {confidence:.1f}%</strong><br>
                <span style="color:#15803d">● Alive: {alive_p:.1f}%</span>
                &nbsp;&nbsp;
                <span style="color:#b91c1c">● Dead: {dead_p:.1f}%</span>
            </div>""", unsafe_allow_html=True)
    else:
        st.info("⬆️ Fill in patient details above and click **Run Prediction**.")

# ── ANALYTICS ──
if predict_btn:
    st.markdown('<div class="section-title" style="margin-top:22px">📊 Analytics Dashboard</div>',
                unsafe_allow_html=True)

    # Row 1
    ch1, ch2, ch3 = st.columns(3, gap="medium")
    with ch1:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.pyplot(chart_confidence(alive_p, dead_p, selected_model), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with ch2:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.pyplot(chart_risk_radar(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with ch3:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.pyplot(chart_node_gauge(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Row 2
    ch4, ch5, ch6 = st.columns(3, gap="medium")
    with ch4:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.pyplot(chart_model_comparison(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with ch5:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.pyplot(chart_survival_timeline(alive_p), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with ch6:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.pyplot(chart_feature_impact(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
