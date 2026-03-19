# ================================================================
# STREAMLIT DASHBOARD — Indian Student At-Risk Warning System
# Run: streamlit run app/dashboard.py
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib, shap, os, warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Student At-Risk System",
    page_icon="🎓", layout="wide"
)

st.markdown("""
<style>
:root {
    --primary: #1e293b;   /* soft navy */
    --accent:  #34d399;   /* mint green */
    --text:    #0f172a;
    --muted:   #6b7280;
    --good:    #34d399;
    --warn:    #fbbf24;
    --bad:     #f87171;
}
.big-title  { font-size:28px; font-weight:700; color:#ffffff; }
.subtitle   { font-size:14px; color:var(--muted); margin-bottom:20px; }
.risk-high  { background:#fff7f7; border:2px solid var(--bad); border-radius:12px; padding:20px; text-align:center; color:#b91c1c; }
.risk-low   { background:#f0fdf4; border:2px solid var(--good); border-radius:12px; padding:20px; text-align:center; color:#15803d; }
.footer     { font-size:12px; color:var(--muted); text-align:center; margin-top:32px; border-top:1px solid #e2e8f0; padding-top:12px; }
body { background:#f8fafc; color:var(--text); }
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(240,249,255,0.9)),
                                        url('https://images.unsplash.com/photo-1509062522246-3755977927d7?auto=format&fit=crop&w=1600&q=80');
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    z-index: -1;
    opacity: 0.55;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    for p in ['../outputs/models/random_forest.pkl',
              'outputs/models/random_forest.pkl']:
        if os.path.exists(p):
            return joblib.load(p)
    return None

@st.cache_resource
def get_explainer(_m):
    return shap.TreeExplainer(_m)

model = load_model()

# ── Header ───────────────────────────────────────────────────────
st.markdown('<div class="big-title">🎓 Student At-Risk Early Warning System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Indian CBSE School Context | IP University B.Tech Project | ML + Explainable AI</div>', unsafe_allow_html=True)

if model is None:
    st.error("Model not found! Run notebooks/03_models.py first.")
    st.stop()

# ── Sidebar inputs ───────────────────────────────────────────────
st.sidebar.markdown("## 📋 Student Details")

st.sidebar.markdown("**🏫 School Info**")
board       = st.sidebar.selectbox("Board", ["CBSE", "ICSE", "State Board"])
school_type = st.sidebar.selectbox("School Type", ["Government", "Private"])
area        = st.sidebar.selectbox("Area", ["Urban", "Semi-Urban", "Rural"])
std_class   = st.sidebar.selectbox("Class", [9, 10, 11, 12])

st.sidebar.markdown("**📊 Academic Performance**")
attendance   = st.sidebar.slider("Attendance (%)", 0, 100, 75)
prev_year    = st.sidebar.slider("Previous Year % ", 0, 100, 60)
study_hours  = st.sidebar.slider("Study Hours per Day", 0, 12, 3)

st.sidebar.markdown("**📝 Subject Marks (out of 100)**")
math_m    = st.sidebar.slider("Mathematics", 0, 100, 55)
sci_m     = st.sidebar.slider("Science",     0, 100, 55)
eng_m     = st.sidebar.slider("English",     0, 100, 60)
hindi_m   = st.sidebar.slider("Hindi",       0, 100, 60)
social_m  = st.sidebar.slider("Social Science", 0, 100, 58)

st.sidebar.markdown("**👨‍👩‍👦 Family Background**")
parent_edu  = st.sidebar.select_slider("Parent Education",
    options=[0,1,2,3,4],
    format_func=lambda x: {0:"Illiterate",1:"Primary",2:"Secondary",
                            3:"Graduate",4:"Post-Graduate"}[x], value=2)
fam_income  = st.sidebar.select_slider("Family Income (Annual)",
    options=[1,2,3,4,5],
    format_func=lambda x: {1:"< ₹1L",2:"₹1–3L",3:"₹3–6L",
                            4:"₹6–10L",5:"> ₹10L"}[x], value=3)
siblings    = st.sidebar.slider("Number of Siblings", 0, 5, 1)
single_par  = st.sidebar.checkbox("Single Parent Household")

st.sidebar.markdown("**📚 Support & Resources**")
tuition     = st.sidebar.checkbox("Attending Private Tuition", value=True)
internet    = st.sidebar.checkbox("Internet Access at Home", value=True)

# ── Encode inputs ────────────────────────────────────────────────
area_enc        = {"Rural": 0, "Semi-Urban": 1, "Urban": 2}[area]
school_type_enc = {"Government": 0, "Private": 1}[school_type]

FEATURES = [
    'attendance_pct', 'prev_year_pct', 'study_hours_per_day',
    'math_marks', 'science_marks', 'english_marks',
    'hindi_marks', 'social_marks',
    'parent_education', 'family_income', 'siblings',
    'single_parent', 'tuition', 'internet_access',
    'school_type_enc', 'area_enc'
]
LABELS = [
    'Attendance %', 'Prev Year %', 'Study Hours/Day',
    'Maths', 'Science', 'English', 'Hindi', 'Social Sci.',
    'Parent Education', 'Family Income', 'Siblings',
    'Single Parent', 'Tuition', 'Internet',
    'School Type', 'Area'
]

inp = pd.DataFrame([[
    attendance, prev_year, study_hours,
    math_m, sci_m, eng_m, hindi_m, social_m,
    parent_edu, fam_income, siblings,
    int(single_par), int(tuition), int(internet),
    school_type_enc, area_enc
]], columns=FEATURES)

# ── Prediction ───────────────────────────────────────────────────
pred    = model.predict(inp)[0]
prob    = model.predict_proba(inp)[0][1]
risk_pct = round(prob * 100, 1)

# ── Main results ─────────────────────────────────────────────────
c1, c2, c3 = st.columns([1.2, 1, 1])

with c1:
    st.markdown("### Prediction")
    if pred == 1:
        st.markdown(f'<div class="risk-high"><div style="font-size:38px">⚠️</div><div style="font-size:20px;font-weight:700;color:var(--bad);margin:6px 0">AT-RISK STUDENT</div><div style="font-size:12px;color:var(--bad)">Needs immediate attention</div></div>', unsafe_allow_html=True)
    else:
            st.markdown(f'<div class="risk-low"><div style="font-size:38px">✅</div><div style="font-size:20px;font-weight:700;color:var(--good);margin:6px 0">NOT AT RISK</div><div style="font-size:12px;color:var(--good)">Passing trajectory — keep monitoring</div></div>', unsafe_allow_html=True)

with c2:
    st.markdown("### Risk Score")
    color = "#d62828" if risk_pct > 50 else "#1b998b"
    st.markdown(f'<div style="text-align:center;padding:14px"><div style="font-size:52px;font-weight:700;color:{color}">{risk_pct}%</div><div style="font-size:12px;color:#6b6a65">chance of failing</div></div>', unsafe_allow_html=True)
    st.progress(float(prob))
    if risk_pct < 25:   st.success("🟢 Low Risk")
    elif risk_pct < 50: st.warning("🟡 Moderate Risk")
    elif risk_pct < 75: st.warning("🟠 High Risk")
    else:               st.error("🔴 Critical Risk")

with c3:
    st.markdown("### Subject Marks")
    total = math_m + sci_m + eng_m + hindi_m + social_m
    pct   = total / 500 * 100
    failed_subs = [s for s, m in [("Maths",math_m),("Science",sci_m),
                   ("English",eng_m),("Hindi",hindi_m),("Social",social_m)] if m < 33]
    st.metric("Overall %", f"{pct:.1f}%",
              delta=f"{pct-35:.1f}% from pass" if pct < 35 else "Passing")
    if failed_subs:
        st.error(f"Failing in: {', '.join(failed_subs)}")
    else:
        st.success("Passing all subjects")

# ── SHAP explanation ─────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔍 Why This Prediction? (SHAP Explainability)")

try:
    exp = get_explainer(model)
    raw_sv = exp.shap_values(inp)

    # Normalize SHAP output shape and align with feature set
    if isinstance(raw_sv, list):
        sv_class = raw_sv[1] if len(raw_sv) > 1 else raw_sv[0]
    elif getattr(raw_sv, 'ndim', 0) == 3:  # shape: (n_samples, n_classes, n_features)
        sv_class = raw_sv[:, 1, :] if raw_sv.shape[1] > 1 else raw_sv[:, 0, :]
    else:
        sv_class = raw_sv

    n_features = min(sv_class.shape[1], len(LABELS))
    sv_class = sv_class[:, :n_features]
    feature_names = LABELS[:n_features]

    inp_lbl = inp.iloc[:, :n_features].copy()
    inp_lbl.columns = feature_names

    shap_df = pd.DataFrame({
        'Factor':  feature_names,
        'Value':   inp_lbl.iloc[0].values,
        'Impact':  sv_class[0]
    }).sort_values('Impact', key=abs, ascending=False)

    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Factors increasing risk 🔴**")
        for _, r in shap_df[shap_df['Impact'] > 0].head(4).iterrows():
            st.markdown(f"**{r['Factor']}** = {r['Value']} &nbsp;→&nbsp; `+{r['Impact']:.4f}`")
    with cb:
        st.markdown("**Factors protecting from risk 🟢**")
        for _, r in shap_df[shap_df['Impact'] < 0].head(4).iterrows():
            st.markdown(f"**{r['Factor']}** = {r['Value']} &nbsp;→&nbsp; `{r['Impact']:.4f}`")

    base_val = exp.expected_value[1] if hasattr(exp.expected_value, '__len__') else exp.expected_value
    explanation = shap.Explanation(
        values=sv_class[0], base_values=base_val,
        data=inp_lbl.iloc[0].values, feature_names=feature_names)
    fig, _ = plt.subplots(figsize=(10, 4))
    shap.waterfall_plot(explanation, show=False, max_display=12)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

except Exception as e:
    st.warning(f"SHAP could not load: {e}")

# ── Teacher recommendations ──────────────────────────────────────
st.markdown("---")
st.markdown("### 📌 Teacher Recommendations (Indian context)")

recs = []
if attendance < 75:
    recs.append("🔴 **Attendance below 75%** — Student may be barred from exams. Contact parents immediately.")
if attendance < 60:
    recs.append("🔴 **Critical attendance (<60%)** — File attendance report to principal.")
if prev_year < 40:
    recs.append("🔴 **Poor previous year performance** — Refer to school counsellor.")
if any(m < 33 for m in [math_m, sci_m, eng_m, hindi_m, social_m]):
    recs.append("🔴 **Failing in one or more subjects** — Arrange remedial classes immediately.")
if study_hours < 2:
    recs.append("🟡 **Very low study time** — Share NCERT study schedule with parents.")
if not tuition and pred == 1:
    recs.append("🟡 **No private tuition** — Suggest free government remedial classes.")
if not internet:
    recs.append("🟡 **No internet access** — Direct student to school library / PM e-Vidya resources.")
if single_par:
    recs.append("🟡 **Single parent household** — May need extra emotional support from school.")
if fam_income <= 2:
    recs.append("🟡 **Low family income** — Check eligibility for scholarships: NSP, State board schemes.")

if not recs:
    st.success("✅ No immediate concerns. Continue regular monitoring.")
else:
    for r in recs:
        st.markdown(r)

# ── Government support & helplines ─────────────────────────────
st.markdown("### ☎️ Government Support & Helplines")
st.markdown("- CBSE Counselling: 1800-11-8004")
st.markdown("- National Mental Health Helpline (Kiran): 1800-599-0019")
st.markdown("- Childline: 1098 (24x7)")
st.markdown("- NSP Scholarship Helpdesk: helpdesk@nsp.gov.in | 0120-6619540")

# ── Useful links ─────────────────────────────────────────────────
with st.expander("📎 Useful Indian Education Resources"):
    st.markdown("""
    - **PM e-Vidya**: diksha.gov.in — free NCERT content all subjects
    - **CBSE Remedial**: cbseacademic.nic.in
    - **National Scholarship Portal**: scholarships.gov.in
    - **UDISE+ Data**: udiseplus.gov.in
    - **Mid-Day Meal Scheme**: mdm.nic.in
    """)

with st.expander("ℹ️ About This System"):
    st.markdown("""
    **Model**: Random Forest (100 trees) trained on synthetic Indian CBSE student data

    **Dataset**: 1000 students, 16 features — Indian school context (CBSE/ICSE/State Board)

    **At-Risk Rule**: CBSE standard — fail if any subject < 33 OR overall < 33%

    **Explainability**: SHAP (SHapley Additive exPlanations)

    **Project**: B.Tech 4th Semester | IP University | ML + AI Subject
    """)

st.markdown('<div class="footer">Student At-Risk Early Warning System | IP University B.Tech Project | Indian CBSE Context</div>', unsafe_allow_html=True)
