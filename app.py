import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InsureIQ · Premium Predictor",
    page_icon="🛡️",
    layout="centered",
)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ── Helper: BMI category ───────────────────────────────────────────────────────
def get_bmi_category(bmi: float) -> str:
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 24.9:
        return "Normal"
    elif bmi < 29.9:
        return "Overweight"
    else:
        return "Obese"

# ── Helper: build feature vector ───────────────────────────────────────────────
FEATURE_COLS = [
    "age", "bmi", "children",
    "sex_male",
    "smoker_yes",
    "region_northwest", "region_southeast", "region_southwest",
    "bmi_age_interaction",
    "bmi_category_Obese", "bmi_category_Overweight", "bmi_category_Underweight",
]

def build_features(age, bmi, children, sex, smoker, region) -> pd.DataFrame:
    cat = get_bmi_category(bmi)
    row = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": int(sex == "Male"),
        "smoker_yes": int(smoker == "Yes"),
        "region_northwest": int(region == "Northwest"),
        "region_southeast": int(region == "Southeast"),
        "region_southwest": int(region == "Southwest"),
        "bmi_age_interaction": bmi * age,
        "bmi_category_Obese": int(cat == "Obese"),
        "bmi_category_Overweight": int(cat == "Overweight"),
        "bmi_category_Underweight": int(cat == "Underweight"),
    }
    return pd.DataFrame([row], columns=FEATURE_COLS)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CSS — injected once
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&display=swap');

/* ── Root variables ── */
:root {
  --ink:      #0d0d14;
  --paper:    #f5f3ee;
  --accent:   #e8420a;
  --accent2:  #1a3fff;
  --muted:    #7a7870;
  --card-bg:  #ffffff;
  --border:   #e2dfd8;
  --radius:   14px;
  --shadow:   0 4px 32px rgba(13,13,20,.08);
}

/* ── Reset Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 1.5rem 4rem !important; max-width: 760px !important; }
.stApp { background: var(--paper); font-family: 'Syne', sans-serif; color: var(--ink); }

/* ── Animated hero banner ── */
@keyframes fadeSlideDown {
  from { opacity: 0; transform: translateY(-28px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeSlideUp {
  from { opacity: 0; transform: translateY(20px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse-ring {
  0%   { box-shadow: 0 0 0 0 rgba(232,66,10,.35); }
  70%  { box-shadow: 0 0 0 18px rgba(232,66,10,0); }
  100% { box-shadow: 0 0 0 0 rgba(232,66,10,0); }
}
@keyframes ticker {
  0%   { transform: translateX(0); }
  100% { transform: translateX(-50%); }
}
@keyframes shimmer {
  0%   { background-position: -400px 0; }
  100% { background-position: 400px 0; }
}
@keyframes countUp {
  from { opacity: 0; transform: scale(.85); }
  to   { opacity: 1; transform: scale(1); }
}

.hero-banner {
  background: var(--ink);
  border-radius: var(--radius);
  padding: 2.8rem 2.4rem 2.2rem;
  margin-bottom: 2rem;
  position: relative;
  overflow: hidden;
  animation: fadeSlideDown .65s cubic-bezier(.22,.68,0,1.2) both;
}
.hero-banner::before {
  content: '';
  position: absolute;
  inset: 0;
  background: repeating-linear-gradient(
    45deg,
    transparent,
    transparent 30px,
    rgba(255,255,255,.018) 30px,
    rgba(255,255,255,.018) 60px
  );
}
.hero-eyebrow {
  font-family: 'DM Mono', monospace;
  font-size: .72rem;
  letter-spacing: .18em;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: .5rem;
}
.hero-title {
  font-size: clamp(2rem, 6vw, 2.9rem);
  font-weight: 800;
  color: #fff;
  line-height: 1.08;
  margin: 0 0 .6rem;
}
.hero-title span { color: var(--accent); }
.hero-sub {
  font-size: .95rem;
  color: rgba(255,255,255,.5);
  font-weight: 400;
  max-width: 420px;
}
.hero-badge {
  position: absolute;
  top: 1.6rem; right: 1.6rem;
  background: var(--accent);
  color: #fff;
  font-family: 'DM Mono', monospace;
  font-size: .7rem;
  padding: .35rem .75rem;
  border-radius: 999px;
  letter-spacing: .08em;
  animation: pulse-ring 2.4s ease-out infinite;
}

/* ── Ticker tape ── */
.ticker-wrap {
  background: var(--accent2);
  overflow: hidden;
  white-space: nowrap;
  border-radius: 8px;
  margin-bottom: 1.8rem;
  padding: .45rem 0;
  animation: fadeSlideUp .6s .3s both;
}
.ticker-inner {
  display: inline-block;
  animation: ticker 18s linear infinite;
}
.ticker-inner span {
  font-family: 'DM Mono', monospace;
  font-size: .78rem;
  color: rgba(255,255,255,.85);
  padding: 0 2.5rem;
  letter-spacing: .05em;
}
.ticker-inner span::before { content: '▸  '; color: rgba(255,255,255,.45); }

/* ── Section label ── */
.section-label {
  font-family: 'DM Mono', monospace;
  font-size: .68rem;
  letter-spacing: .2em;
  text-transform: uppercase;
  color: var(--muted);
  margin: 2rem 0 .8rem;
  display: flex;
  align-items: center;
  gap: .6rem;
}
.section-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}

/* ── Card wrapper ── */
.card {
  background: var(--card-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.6rem 1.8rem;
  margin-bottom: 1.2rem;
  box-shadow: var(--shadow);
  animation: fadeSlideUp .5s cubic-bezier(.22,.68,0,1.1) both;
}

/* ── Streamlit widget overrides ── */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div > input,
div[data-testid="stSlider"] {
  border-radius: 8px !important;
  font-family: 'Syne', sans-serif !important;
  border-color: var(--border) !important;
  transition: border-color .2s, box-shadow .2s !important;
}
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="input"] > div:focus-within {
  border-color: var(--accent2) !important;
  box-shadow: 0 0 0 3px rgba(26,63,255,.12) !important;
}
label[data-testid="stWidgetLabel"] p {
  font-family: 'Syne', sans-serif !important;
  font-weight: 600 !important;
  font-size: .88rem !important;
  color: var(--ink) !important;
}

/* ── Predict button ── */
div[data-testid="stButton"] > button {
  background: var(--accent) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 10px !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: 1rem !important;
  padding: .85rem 2.4rem !important;
  cursor: pointer !important;
  transition: transform .18s, box-shadow .18s, background .18s !important;
  box-shadow: 0 4px 18px rgba(232,66,10,.3) !important;
  width: 100%;
}
div[data-testid="stButton"] > button:hover {
  background: #c73508 !important;
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 28px rgba(232,66,10,.38) !important;
}
div[data-testid="stButton"] > button:active {
  transform: translateY(0) !important;
}

/* ── Result card ── */
.result-card {
  background: var(--ink);
  color: #fff;
  border-radius: var(--radius);
  padding: 2.2rem 2rem;
  text-align: center;
  margin-top: 1.6rem;
  animation: countUp .55s cubic-bezier(.22,.68,0,1.25) both;
  position: relative;
  overflow: hidden;
}
.result-card::after {
  content: '';
  position: absolute;
  bottom: -40px; right: -40px;
  width: 180px; height: 180px;
  border-radius: 50%;
  background: var(--accent);
  opacity: .12;
}
.result-label {
  font-family: 'DM Mono', monospace;
  font-size: .72rem;
  letter-spacing: .18em;
  text-transform: uppercase;
  color: rgba(255,255,255,.5);
  margin-bottom: .4rem;
}
.result-amount {
  font-size: clamp(2.4rem, 8vw, 3.8rem);
  font-weight: 800;
  color: #fff;
  letter-spacing: -.02em;
  line-height: 1;
}
.result-amount em { color: var(--accent); font-style: normal; }
.result-note {
  font-size: .82rem;
  color: rgba(255,255,255,.45);
  margin-top: .7rem;
}

/* ── Risk meter ── */
.risk-wrap {
  margin-top: 1.4rem;
  padding: 0 .4rem;
}
.risk-bar-bg {
  height: 8px;
  border-radius: 99px;
  background: rgba(255,255,255,.12);
  overflow: hidden;
  margin: .5rem 0 .3rem;
}
.risk-bar-fill {
  height: 100%;
  border-radius: 99px;
  transition: width 1.2s cubic-bezier(.22,.68,0,1.1);
}
.risk-labels {
  display: flex;
  justify-content: space-between;
  font-family: 'DM Mono', monospace;
  font-size: .65rem;
  color: rgba(255,255,255,.35);
  letter-spacing: .08em;
}

/* ── Info grid ── */
.info-grid {
  display: grid;
  grid-template-columns: 1fr 1fr 1fr;
  gap: .8rem;
  margin-top: 1.2rem;
}
.info-cell {
  background: rgba(255,255,255,.06);
  border-radius: 10px;
  padding: .8rem .9rem;
  text-align: center;
}
.info-cell-val {
  font-weight: 700;
  font-size: 1.05rem;
  color: #fff;
}
.info-cell-key {
  font-family: 'DM Mono', monospace;
  font-size: .65rem;
  color: rgba(255,255,255,.4);
  text-transform: uppercase;
  letter-spacing: .08em;
  margin-top: .2rem;
}

/* ── Disclaimer ── */
.disclaimer {
  font-family: 'DM Mono', monospace;
  font-size: .7rem;
  color: var(--muted);
  text-align: center;
  margin-top: 2.4rem;
  line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  HERO BANNER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
  <div class="hero-badge">ML · R² 0.86</div>
  <div class="hero-eyebrow">AI-powered estimation</div>
  <div class="hero-title">InsureIQ<br><span>Premium</span> Predictor</div>
  <div class="hero-sub">Enter your profile details below and get an instant medical insurance charge estimate powered by a tuned Random Forest model.</div>
</div>
""", unsafe_allow_html=True)

# ── Ticker ────────────────────────────────────────────────────────────────────
ticker_items = (
    "Age · BMI · Smoking status · Region · Children · BMI-Age interaction · "
    "Random Forest Regressor · GridSearchCV tuned · R² 0.86 · "
    "Age · BMI · Smoking status · Region · Children · BMI-Age interaction · "
    "Random Forest Regressor · GridSearchCV tuned · R² 0.86 · "
)
st.markdown(f"""
<div class="ticker-wrap">
  <div class="ticker-inner">
    <span>{ticker_items}</span><span>{ticker_items}</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  INPUT FORM
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">Personal profile</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", min_value=18, max_value=80, value=35, step=1)
with col2:
    sex = st.selectbox("Biological Sex", ["Male", "Female"])

col3, col4 = st.columns(2)
with col3:
    bmi = st.number_input("BMI", min_value=10.0, max_value=55.0, value=27.5, step=0.1,
                          help="Body Mass Index  (kg / m²)")
with col4:
    children = st.slider("Dependant Children", min_value=0, max_value=5, value=0)

st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">Lifestyle & location</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)

col5, col6 = st.columns(2)
with col5:
    smoker = st.selectbox("Smoker?", ["No", "Yes"])
with col6:
    region = st.selectbox("US Region", ["Northeast", "Northwest", "Southeast", "Southwest"])

st.markdown('</div>', unsafe_allow_html=True)

# ── Derived display values ─────────────────────────────────────────────────────
bmi_cat = get_bmi_category(bmi)

# ══════════════════════════════════════════════════════════════════════════════
#  PREDICT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
predict_btn = st.button("⚡  Calculate My Premium Estimate")

if predict_btn:
    X = build_features(age, bmi, children, sex, smoker, region)
    log_pred = model.predict(X)[0]
    predicted = np.expm1(log_pred)

    # Risk band
    if predicted < 8_000:
        risk_label, risk_pct, risk_color = "Low Risk", 22, "#22c55e"
    elif predicted < 18_000:
        risk_label, risk_pct, risk_color = "Moderate Risk", 55, "#f59e0b"
    elif predicted < 32_000:
        risk_label, risk_pct, risk_color = "High Risk", 78, "#ef4444"
    else:
        risk_label, risk_pct, risk_color = "Very High Risk", 95, "#dc2626"

    formatted = f"${predicted:,.0f}"
    dollar, cents_part = formatted.split(",", 1) if "," in formatted else (formatted, "")

    st.markdown(f"""
    <div class="result-card">
      <div class="result-label">Estimated annual insurance charge</div>
      <div class="result-amount"><em>{dollar}</em>,{cents_part}</div>
      <div class="result-note">Based on your profile — this is a model estimate, not a quote.</div>

      <div class="risk-wrap">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.3rem;">
          <span style="font-family:'DM Mono',monospace;font-size:.72rem;color:rgba(255,255,255,.55);letter-spacing:.1em;text-transform:uppercase;">Risk level</span>
          <span style="font-family:'Syne',sans-serif;font-size:.82rem;font-weight:700;color:{risk_color};">{risk_label}</span>
        </div>
        <div class="risk-bar-bg">
          <div class="risk-bar-fill" style="width:{risk_pct}%;background:{risk_color};"></div>
        </div>
        <div class="risk-labels"><span>LOW</span><span>MODERATE</span><span>HIGH</span></div>
      </div>

      <div class="info-grid">
        <div class="info-cell">
          <div class="info-cell-val">{age} yrs</div>
          <div class="info-cell-key">Age</div>
        </div>
        <div class="info-cell">
          <div class="info-cell-val">{bmi:.1f}</div>
          <div class="info-cell-key">BMI · {bmi_cat}</div>
        </div>
        <div class="info-cell">
          <div class="info-cell-val">{'🚬 Yes' if smoker == 'Yes' else '✅ No'}</div>
          <div class="info-cell-key">Smoker</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ── Disclaimer ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
  InsureIQ uses a Random Forest Regressor (R² ≈ 0.86) trained on the public insurance dataset.<br>
  Predictions are estimates only — not financial or medical advice.<br>
  © 2025 InsureIQ · Powered by scikit-learn & Streamlit
</div>
""", unsafe_allow_html=True)