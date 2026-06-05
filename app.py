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

# ── JavaScript: force sidebar open and block collapse button ──
st.components.v1.html("""
<script>
(function keepSidebarOpen() {
    function forceSidebar() {
        // Remove collapsed attribute from sidebar wrapper
        var sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            sidebar.removeAttribute('aria-hidden');
            sidebar.style.display = 'flex';
            sidebar.style.visibility = 'visible';
            sidebar.style.width = '320px';
            sidebar.style.minWidth = '320px';
            sidebar.style.transform = 'none';
            sidebar.style.position = 'relative';
        }

        // Hide all collapse / close buttons
        var selectors = [
            '[data-testid="collapsedControl"]',
            '[data-testid="stSidebarCollapseButton"]',
            'button[aria-label="Close sidebar"]',
            'button[aria-label="Collapse sidebar"]',
            'button[title="Close sidebar"]',
            '[data-testid="baseButton-headerNoPadding"]'
        ];
        selectors.forEach(function(sel) {
            window.parent.document.querySelectorAll(sel).forEach(function(el) {
                el.style.display = 'none';
                el.style.visibility = 'hidden';
                el.style.pointerEvents = 'none';
            });
        });
    }

    // Run immediately and on a short interval to catch re-renders
    forceSidebar();
    setInterval(forceSidebar, 300);
})();
</script>
""", height=0)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Sora:wght@600;700;800&display=swap');

/* ── BASE ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 15px !important;
}

.stApp { background: #f0f2f6 !important; }

header[data-testid="stHeader"] { display: none !important; }

.main .block-container {
    padding: 1rem 1.8rem 1rem 1.8rem !important;
    max-width: 100% !important;
}

/* ══════════════════════════════════════════
   SIDEBAR — force always open
══════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    display: flex !important;
    visibility: visible !important;
    width: 320px !important;
    min-width: 320px !important;
    max-width: 320px !important;
    transform: none !important;
    position: relative !important;
    background: #111827 !important;
    border-right: 2px solid #374151 !important;
    flex-shrink: 0 !important;
}

section[data-testid="stSidebar"] > div:first-child {
    width: 320px !important;
    min-width: 320px !important;
}

/* Kill ALL collapse buttons */
button[data-testid="baseButton-headerNoPadding"],
div[data-testid="collapsedControl"],
section[data-testid="stSidebar"] button[kind="header"],
[data-testid="stSidebarCollapseButton"],
button[aria-label="Close sidebar"],
button[aria-label="Collapse sidebar"],
button[title="Close sidebar"],
[class*="collapsedControl"],
[class*="sSidebarCollapse"] {
    display: none !important;
    visibility: hidden !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    pointer-events: none !important;
    position: absolute !important;
    opacity: 0 !important;
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Sidebar labels */
section[data-testid="stSidebar"] label {
    color: #9ca3af !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.07em !important;
}

/* Sidebar inputs */
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #1f2937 !important;
    border: 1px solid #374151 !important;
    border-radius: 8px !important;
    color: #f3f4f6 !important;
    font-size: 0.95rem !important;
    min-height: 40px !important;
}

section[data-testid="stSidebar"] input:focus,
section[data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within {
    border-color: #16a34a !important;
    box-shadow: 0 0 0 3px rgba(22,163,74,0.2) !important;
}

/* Sidebar headings */
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #f9fafb !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    margin: 6px 0 4px 0 !important;
}

section[data-testid="stSidebar"] hr {
    border-color: #374151 !important;
    margin: 12px 0 !important;
}

section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] .stNumberInput {
    margin-bottom: 10px !important;
}

/* Predict button */
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #16a34a 0%, #dc2626 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    padding: 0.7rem 1rem !important;
    width: 100% !important;
    box-shadow: 0 4px 18px rgba(22,163,74,0.35) !important;
    transition: all 0.2s ease !important;
    cursor: pointer !important;
    margin-top: 6px !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    box-shadow: 0 6px 26px rgba(22,163,74,0.5) !important;
    transform: translateY(-2px) !important;
    filter: brightness(1.1) !important;
}

/* ══════════════════════════════════════════
   HERO HEADER — full-width via negative margins
══════════════════════════════════════════ */
.hero-wrap {
    margin: -1rem -1.8rem 1.2rem -1.8rem;
    padding: 0 2rem;
    background: linear-gradient(110deg, #0f172a 0%, #1e293b 45%, #0f172a 100%);
    border-bottom: 3px solid transparent;
    border-image: linear-gradient(90deg, #16a34a 0%, #dc2626 100%) 1;
    box-shadow: 0 6px 28px rgba(0,0,0,0.3);
    position: relative;
    overflow: hidden;
}

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
    min-height: 76px;
    padding: 12px 0;
    gap: 20px;
}

.hero-brand {
    display: flex;
    align-items: center;
    gap: 18px;
}

.hero-icon {
    font-size: 2.6rem;
    line-height: 1;
    filter: drop-shadow(0 2px 8px rgba(0,0,0,0.5));
}

.hero-title {
    font-family: 'Sora', sans-serif;
    font-size: 1.45rem;
    font-weight: 800;
    color: #f8fafc;
    margin: 0 0 4px 0;
    letter-spacing: -0.02em;
    line-height: 1.15;
}

.hero-sub {
    font-size: 0.85rem;
    color: #94a3b8;
    margin: 0;
    font-weight: 400;
    letter-spacing: 0.01em;
}

.hero-stats {
    display: flex;
    align-items: center;
    gap: 24px;
}

.hstat { text-align: center; }

.hstat-num {
    font-family: 'Sora', sans-serif;
    font-size: 1.35rem;
    font-weight: 800;
    color: #f1f5f9;
    display: block;
    line-height: 1;
}

.hstat-lbl {
    font-size: 0.68rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    display: block;
    margin-top: 4px;
    font-weight: 600;
}

.hstat-div {
    width: 1px;
    height: 38px;
    background: rgba(100,116,139,0.35);
}

/* ══════════════════════════════════════════
   SECTION LABELS
══════════════════════════════════════════ */
.sec-label {
    font-family: 'Sora', sans-serif;
    font-size: 0.82rem;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0 0 10px 0;
    display: flex;
    align-items: center;
    gap: 6px;
}

.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #e2e8f0;
    margin-left: 8px;
}

/* ══════════════════════════════════════════
   SUMMARY CARD
══════════════════════════════════════════ */
.summary-card {
    background: white;
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    border: 1px solid #e2e8f0;
}

/* Bigger dataframe text */
[data-testid="stDataFrame"] * { font-size: 0.92rem !important; }
[data-testid="stDataFrame"] th {
    font-size: 0.84rem !important;
    font-weight: 700 !important;
    background: #f8fafc !important;
    color: #334155 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    padding: 8px 12px !important;
}
[data-testid="stDataFrame"] td {
    color: #334155 !important;
    padding: 7px 12px !important;
}

/* ══════════════════════════════════════════
   RESULT CARDS
══════════════════════════════════════════ */
.result-card {
    border-radius: 14px;
    padding: 22px 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.result-alive {
    background: linear-gradient(135deg, #16a34a 0%, #15803d 100%);
    box-shadow: 0 6px 28px rgba(22,163,74,0.38);
}

.result-dead {
    background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
    box-shadow: 0 6px 28px rgba(220,38,38,0.38);
}

.rc-badge {
    display: inline-block;
    background: rgba(255,255,255,0.18);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    font-weight: 700;
    color: white;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 12px;
}

.rc-icon {
    font-size: 2.2rem;
    display: block;
    margin-bottom: 8px;
    line-height: 1;
}

.rc-verdict {
    font-family: 'Sora', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: white;
    display: block;
    line-height: 1.1;
    margin-bottom: 6px;
}

.rc-sub {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.82);
    display: block;
}

.conf-row {
    background: white;
    border-radius: 12px;
    padding: 14px 20px;
    margin-top: 12px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    border: 1px solid #e2e8f0;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.conf-label {
    font-size: 0.78rem;
    color: #94a3b8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 3px;
}

.conf-value {
    font-family: 'Sora', sans-serif;
    font-size: 1.25rem;
    font-weight: 800;
    color: #0f172a;
}

.conf-alive { font-size: 1rem; font-weight: 700; color: #16a34a; }
.conf-dead  { font-size: 1rem; font-weight: 700; color: #dc2626; }

/* ══════════════════════════════════════════
   CHART CARDS
══════════════════════════════════════════ */
.chart-card {
    background: white;
    border-radius: 14px;
    padding: 16px 18px 12px 18px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    border: 1px solid #e2e8f0;
}

/* Alert */
.stAlert {
    border-radius: 10px !important;
    border-left: 4px solid #16a34a !important;
    background: #f0fdf4 !important;
    font-size: 0.92rem !important;
}

/* Caption */
[data-testid="stCaptionContainer"] p,
.stCaption {
    font-size: 0.82rem !important;
    color: #94a3b8 !important;
    margin-top: 8px !important;
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
    <div style="
        background: linear-gradient(135deg,#16a34a,#dc2626);
        padding: 18px 20px 16px;
        margin: -1px -1px 0;
        border-bottom: 1px solid #374151;
    ">
        <div style="font-family:'Sora',sans-serif;font-size:1.1rem;font-weight:800;
                    color:white;letter-spacing:-0.01em;margin-bottom:4px;">
            🎗️ BC Predictor
        </div>
        <div style="font-size:0.78rem;color:rgba(255,255,255,0.72);">
            Survival Analysis Dashboard
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
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

    

# ══════════════════════════════
# HERO HEADER
# ══════════════════════════════
st.markdown("""
<div class="hero-header">
    <div class="hero-content">
        <h1>🎗 Breast Cancer Survival Predictor</h1>
        <p>AI-Powered Survival Risk Assessment Dashboard</p>
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
CHART_SIZE = (5.6, 2.8)

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
                f"{val:.1f}%", va="center", fontsize=11,
                fontweight="bold", color="#1e293b")

    ax.set_xlim(0, 122)
    ax.set_xlabel("Probability (%)", fontsize=10, color="#94a3b8")
    ax.set_title(f"Prediction Confidence — {model_name}",
                 fontsize=11, color="#334155", fontweight="bold", pad=8)
    ax.tick_params(labelsize=10, colors="#64748b")
    ax.set_ylim(-1.0, 2.0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, color="#f1f5f9", linewidth=0.9)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.6)
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
    ax.annotate("▲ selected", xy=(idx, 106), ha="center",
                fontsize=8, color="#16a34a", fontweight="700")

    short = [n.replace(" ", "\n") for n in df["Model"]]
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=8.5, color="#64748b", ha="center")
    ax.set_ylabel("Probability (%)", fontsize=10, color="#94a3b8")
    ax.set_title("All Models Comparison",
                 fontsize=11, color="#334155", fontweight="bold", pad=8)
    ax.set_ylim(0, 122)
    ax.legend(fontsize=9.5, framealpha=0, loc="upper right")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.yaxis.grid(True, color="#f1f5f9", linewidth=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=10, colors="#94a3b8")
    plt.tight_layout(pad=0.5)
    return fig


# ══════════════════════════════
# MAIN LAYOUT
# ══════════════════════════════
col_left, col_right = st.columns([1.5, 1], gap="large")

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
    st.dataframe(summary, width="stretch", hide_index=True, height=268)
    st.markdown('</div>', unsafe_allow_html=True)
    st.caption(f"Active model: **{selected_model}**  ·  Features used: {len(features)}")

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


if predict_btn:
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown('<p class="sec-label">📊 Prediction Confidence</p>', unsafe_allow_html=True)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.pyplot(chart_confidence(alive_p, dead_p, selected_model), width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<p class="sec-label">📊 All Models Comparison</p>', unsafe_allow_html=True)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.pyplot(chart_model_comparison(), width="stretch")
        st.markdown('</div>', unsafe_allow_html=True)
