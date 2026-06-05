import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Breast Cancer Survival Predictor",
    page_icon="🎗️",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Outfit:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }

/* Hide all Streamlit chrome */
#MainMenu, footer, header { display: none !important; }
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] { display: none !important; }

/* Hide sidebar completely — we don't use it */
section[data-testid="stSidebar"] { display: none !important; }

/* App background */
.stApp { background: #f5eded; }
.block-container {
    padding-top: 0 !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 100% !important;
}

/* ── Top Header ── */
.header-bar {
    background: linear-gradient(90deg, #6b0f1a 0%, #b91c1c 55%, #dc2626 100%);
    border-radius: 0 0 16px 16px;
    padding: 16px 32px;
    display: flex;
    align-items: center;
    gap: 18px;
    margin-bottom: 22px;
    box-shadow: 0 4px 24px rgba(107,15,26,0.3);
}
.header-bar h1 {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem; color: white; margin: 0;
}
.header-bar p { font-size: 0.77rem; color: rgba(255,255,255,0.72); margin: 0; }
.hbar-divider { width:1px; height:40px; background:rgba(255,255,255,0.22); margin: 0 6px; }

/* ── Input Panel ── */
.input-panel {
    background: #1c0608;
    border-radius: 14px;
    padding: 18px 24px 14px 24px;
    margin-bottom: 22px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.18);
}
.input-panel label,
.input-panel .stSelectbox label,
.input-panel .stNumberInput label {
    color: #f9c8c8 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
}
.input-panel * { color: #f0dada !important; }
.input-panel input {
    background: #2c0c0c !important;
    border: 1px solid #5a2020 !important;
    color: #f0dada !important;
    border-radius: 6px !important;
}

/* Predict button */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #9b1414, #dc2626) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 10px 22px !important;
    box-shadow: 0 4px 14px rgba(185,28,28,0.4) !important;
    transition: opacity 0.2s;
    width: 100%;
}
div[data-testid="stButton"] > button:hover { opacity: 0.85 !important; }

/* ── Section titles ── */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1rem; color: #6b0f1a;
    border-bottom: 1px solid #dcc5c5;
    padding-bottom: 5px; margin: 0 0 10px 0;
}

/* ── Result cards ── */
.result-alive {
    background: linear-gradient(135deg, #14532d, #15803d);
    border-radius: 12px; padding: 20px 24px;
    color: white; text-align: center;
    box-shadow: 0 6px 20px rgba(21,128,61,0.25);
}
.result-dead {
    background: linear-gradient(135deg, #6b0f1a, #b91c1c);
    border-radius: 12px; padding: 20px 24px;
    color: white; text-align: center;
    box-shadow: 0 6px 20px rgba(185,28,28,0.25);
}
.result-alive h2, .result-dead h2 {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem; margin: 0 0 6px 0; color: white;
}
.result-alive p, .result-dead p { font-size: 0.85rem; opacity: 0.88; margin: 0; }
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
    padding: 12px 18px; margin-top: 10px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    text-align: center; font-size: 0.88rem; color: #333;
}
.info-box {
    background: #fff5f5; border-left: 3px solid #b91c1c;
    border-radius: 6px; padding: 9px 13px; margin-top: 10px;
    font-size: 0.76rem; color: #666;
}
.chart-wrap {
    background: white; border-radius: 12px;
    padding: 16px 10px 8px 10px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
}
</style>
""", unsafe_allow_html=True)

# ── LOAD PKL ──
PKL_PATH = "breast_cancer_models.pkl"

@st.cache_resource
def load_bundle(path):
    return joblib.load(path)

if not os.path.exists(PKL_PATH):
    st.error(f"❌ `{PKL_PATH}` not found. Place it alongside this script.")
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
        <p>ML-powered risk assessment &nbsp;·&nbsp; For educational use only</p>
    </div>
    <div class="hbar-divider"></div>
    <p style="color:rgba(255,255,255,0.55);font-size:0.72rem;margin:0;line-height:1.7">
        8 Models &nbsp;·&nbsp; LR · KNN · RF · DT · SVM · GB · NB · XGB
    </p>
</div>
""", unsafe_allow_html=True)

# ── INPUT PANEL (replaces sidebar) ──
st.markdown('<div class="input-panel">', unsafe_allow_html=True)

r1c1, r1c2, r1c3, r1c4, r1c5, r1c6, r1c7, r1c8, r1c9 = st.columns([1.4,1,1,1,1,1,1,1,1])

with r1c1:
    selected_model = st.selectbox("🤖 Model", MODEL_NAMES)
with r1c2:
    survival_months = st.number_input("Survival Months", min_value=1, max_value=120, value=40, step=1)
with r1c3:
    tumor_size = st.number_input("Tumor Size (mm)", min_value=1, max_value=200, value=25, step=1)
with r1c4:
    reginol_node_pos = st.number_input("Node Positive", min_value=0, max_value=50, value=2, step=1)
with r1c5:
    regional_node_exam = st.number_input("Node Examined", min_value=1, max_value=60, value=10, step=1)
with r1c6:
    estrogen = st.selectbox("Estrogen", ["Positive", "Negative"])
with r1c7:
    progesterone = st.selectbox("Progesterone", ["Positive", "Negative"])
with r1c8:
    a_stage = st.selectbox("A Stage", ["Regional", "Distant"])
with r1c9:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Predict", use_container_width=True)

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

CHART_H = 3.2
CHART_W = 5.0

def chart_confidence(alive_p, dead_p, model_name):
    fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    bars = ax.barh(["Deceased", "Alive"], [dead_p, alive_p],
                   color=["#b91c1c", "#15803d"], height=0.42, edgecolor="none")
    for bar, val in zip(bars, [dead_p, alive_p]):
        ax.text(val + 1.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=11, fontweight="bold", color="#333")
    ax.set_xlim(0, 125)
    ax.set_xlabel("Probability (%)", fontsize=8, color="#aaa")
    ax.set_title(f"Prediction Confidence — {model_name}", fontsize=9,
                 color="#6b0f1a", fontweight="bold", pad=10)
    ax.tick_params(colors="#666", labelsize=9)
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.tight_layout(pad=1.0)
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
    fig, ax = plt.subplots(figsize=(CHART_W, CHART_H))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    x, w = np.arange(len(df)), 0.35
    ax.bar(x - w/2, df["Alive %"], width=w, color="#15803d", label="Alive", edgecolor="none")
    ax.bar(x + w/2, df["Dead %"],  width=w, color="#b91c1c", label="Dead",  edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(" ", "\n") for n in df["Model"]], fontsize=6.5, color="#444")
    ax.set_ylabel("Probability (%)", fontsize=8, color="#aaa")
    ax.set_title("All Models — Alive vs Deceased", fontsize=9,
                 color="#6b0f1a", fontweight="bold", pad=10)
    ax.set_ylim(0, 120)
    ax.legend(fontsize=8, framealpha=0, loc="upper right")
    ax.tick_params(colors="#666", labelsize=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    idx = MODEL_NAMES.index(selected_model)
    ax.axvspan(idx - 0.5, idx + 0.5, color="#6b0f1a", alpha=0.08, zorder=0)
    plt.tight_layout(pad=1.0)
    return fig

# ── ROW 1: Patient Summary | Prediction Result ──
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
    st.caption(f"Model features: {', '.join(features)}")

with col2:
    st.markdown('<div class="section-title">🤖 Prediction Result</div>', unsafe_allow_html=True)

    if predict_btn:
        model        = bundle[selected_model]
        X_input      = get_input_for_model(selected_model)
        pred_encoded = model.predict(X_input)[0]
        pred_label   = label_encoder.inverse_transform([pred_encoded])[0]
        alive_p, dead_p = 50.0, 50.0

        if pred_label == "Alive":
            st.markdown(f"""<div class="result-alive">
                <div class="model-badge">{selected_model}</div>
                <h2>✅ Alive</h2>
                <p>The model predicts the patient is likely to survive.</p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="result-dead">
                <div class="model-badge">{selected_model}</div>
                <h2>⚠️ Deceased</h2>
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
                <span style="color:#b91c1c">● Deceased: {dead_p:.1f}%</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("""<div class="info-box">
            ⚠️ <strong>Disclaimer:</strong> For educational purposes only.
            Not a substitute for clinical diagnosis.
        </div>""", unsafe_allow_html=True)
    else:
        st.info("⬆️ Fill in patient details above and click **Predict**.")

# ── ROW 2: Both Charts Same Size ──
if predict_btn:
    st.markdown('<div class="section-title" style="margin-top:22px">📊 Analytics</div>',
                unsafe_allow_html=True)
    ch1, ch2 = st.columns(2, gap="large")

    with ch1:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.pyplot(chart_confidence(alive_p, dead_p, selected_model), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with ch2:
        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.pyplot(chart_model_comparison(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
