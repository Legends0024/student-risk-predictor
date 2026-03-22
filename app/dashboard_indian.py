import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib, shap, os, warnings
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Student At-Risk Warning System | India",
    page_icon="🎓", layout="wide"
)

st.markdown("""
<style>
.big-title  { font-size:28px; font-weight:700; }
.subtitle   { font-size:14px; color:#6b6a65; margin-bottom:20px; }
.risk-high  { background:#FFEBEE; border:2px solid #E53935; border-radius:12px; padding:20px; text-align:center; }
.risk-low   { background:#E8F5E9; border:2px solid #43A047; border-radius:12px; padding:20px; text-align:center; }
.footer     { font-size:12px; color:#9a9890; text-align:center; margin-top:32px; border-top:1px solid #e8e6e0; padding-top:12px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    for p in ['outputs/models/random_forest.pkl',
              '../outputs/models/random_forest.pkl']:
        if os.path.exists(p):
            return joblib.load(p)
    return None

@st.cache_resource
def get_explainer(_m):
    return shap.TreeExplainer(_m)

model = load_model()

st.markdown('<div class="big-title">🎓 Student At-Risk Early Warning System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Indian School Context | CBSE/ICSE/State Board | IP University B.Tech Project</div>', unsafe_allow_html=True)

if model is None:
    st.error("Model not found! Make sure outputs/models/random_forest.pkl exists.")
    st.stop()

# ── Sidebar ──────────────────────────────────────────────────────
st.sidebar.markdown("## 📋 Student Details")

st.sidebar.markdown("**🏫 School Info**")
board       = st.sidebar.selectbox("Board", ["CBSE", "ICSE", "State Board"], key="board_select")
school_type = st.sidebar.selectbox("School Type", ["Government", "Private"], key="school_type_select")
area        = st.sidebar.selectbox("Area", ["Urban", "Semi-Urban", "Rural"], key="area_select")
# Explicit key + int cast prevents session-state weirdness when switching classes.
std_class   = int(st.sidebar.selectbox("Class", [9, 10, 11, 12], key="class_select"))

# ── Stream selection (disabled for Class 9/10 to always show the widget) ─────
st.sidebar.markdown("**🎯 Stream**")
stream_disabled = std_class not in [11, 12]
stream = st.sidebar.selectbox(
    "Select Stream",
    ["Science (PCM)", "Science (PCB)", "Commerce", "Arts/Humanities"],
    key="stream_select",
    disabled=stream_disabled,
)
if stream_disabled:
    stream = None

stream_colors = {
    "Science (PCM)":   ("🔵", "#E3F2FD", "#1565C0"),
    "Science (PCB)":   ("🟢", "#E8F5E9", "#1B5E20"),
    "Commerce":        ("🟡", "#FFFDE7", "#F57F17"),
    "Arts/Humanities": ("🟣", "#F3E5F5", "#6A1B9A"),
}

st.sidebar.markdown("**📊 Academic**")
attendance  = st.sidebar.slider("Attendance (%)", 0, 100, 75)
prev_year   = st.sidebar.slider("Previous Year %", 0, 100, 60)
study_hours = st.sidebar.slider("Study Hours/Day", 0, 12, 3)

# ── Subject marks — change labels based on class/stream ──────────
st.sidebar.markdown("**📝 Subject Marks (out of 100)**")

if std_class in [9, 10]:
    st.sidebar.caption("Standard subjects — Class 9 & 10")
    s1_label, s2_label, s3_label, s4_label, s5_label = \
        "Mathematics", "Science", "English", "Hindi", "Social Science"

elif stream == "Science (PCM)":
    st.sidebar.caption("PCM — Physics, Chemistry, Mathematics")
    s1_label, s2_label, s3_label, s4_label, s5_label = \
        "Mathematics", "Physics", "English", "Chemistry", "Computer Sci/IP"

elif stream == "Science (PCB)":
    st.sidebar.caption("PCB — Physics, Chemistry, Biology")
    s1_label, s2_label, s3_label, s4_label, s5_label = \
        "Biology", "Physics", "English", "Chemistry", "Physical Education"

elif stream == "Commerce":
    st.sidebar.caption("Commerce stream subjects")
    s1_label, s2_label, s3_label, s4_label, s5_label = \
        "Accountancy", "Business Studies", "English", "Economics", "Mathematics/IP"

else:  # Arts/Humanities
    st.sidebar.caption("Arts/Humanities stream subjects")
    s1_label, s2_label, s3_label, s4_label, s5_label = \
        "History", "Political Science", "English", "Geography/Economics", "Psychology/Sociology"

math_m   = st.sidebar.slider(s1_label, 0, 100, 55)
sci_m    = st.sidebar.slider(s2_label, 0, 100, 55)
eng_m    = st.sidebar.slider(s3_label, 0, 100, 60)
hindi_m  = st.sidebar.slider(s4_label, 0, 100, 60)
social_m = st.sidebar.slider(s5_label, 0, 100, 58)

st.sidebar.markdown("**👨‍👩‍👦 Family Background**")
parent_edu = st.sidebar.select_slider("Parent Education",
    options=[0,1,2,3,4],
    format_func=lambda x: {0:"Illiterate",1:"Primary",2:"Secondary",
                            3:"Graduate",4:"Post-Graduate"}[x], value=2)
fam_income = st.sidebar.select_slider("Family Income (Annual)",
    options=[1,2,3,4,5],
    format_func=lambda x: {1:"< ₹1L",2:"₹1–3L",3:"₹3–6L",
                            4:"₹6–10L",5:"> ₹10L"}[x], value=3)
siblings   = st.sidebar.slider("Siblings", 0, 5, 1)
single_par = st.sidebar.checkbox("Single Parent Household")

st.sidebar.markdown("**📚 Resources**")
tuition  = st.sidebar.checkbox("Private Tuition", value=True)
internet = st.sidebar.checkbox("Internet Access",  value=True)

# ── Encode & predict ─────────────────────────────────────────────
area_enc        = {"Rural":0,"Semi-Urban":1,"Urban":2}[area]
school_type_enc = {"Government":0,"Private":1}[school_type]

FEATURES = ['attendance_pct','prev_year_pct','study_hours_per_day',
            'math_marks','science_marks','english_marks',
            'hindi_marks','social_marks',
            'parent_education','family_income','siblings',
            'single_parent','tuition','internet_access',
            'school_type_enc','area_enc']

LABELS = ['Attendance %','Prev Year %','Study Hours/Day',
          s1_label, s2_label, 'English',
          s4_label, s5_label,
          'Parent Education','Family Income','Siblings',
          'Single Parent','Tuition','Internet',
          'School Type','Area']

inp = pd.DataFrame([[
    attendance, prev_year, study_hours,
    math_m, sci_m, eng_m, hindi_m, social_m,
    parent_edu, fam_income, siblings,
    int(single_par), int(tuition), int(internet),
    school_type_enc, area_enc
]], columns=FEATURES)

pred     = model.predict(inp)[0]
prob     = model.predict_proba(inp)[0][1]
risk_pct = round(prob * 100, 1)

# ── Stream/class badge ───────────────────────────────────────────
if std_class in [11,12] and stream:
    icon, bg, color = stream_colors.get(stream, ("⚪","#F5F5F5","#333"))
    st.markdown(f'<span style="background:{bg};color:{color};padding:4px 14px;border-radius:20px;font-size:12px;font-weight:600;">{icon} {stream} | Class {std_class} | {board}</span><br><br>', unsafe_allow_html=True)
else:
    st.markdown(f'<span style="background:#E3F2FD;color:#1565C0;padding:4px 14px;border-radius:20px;font-size:12px;font-weight:600;">📚 Class {std_class} | {board}</span><br><br>', unsafe_allow_html=True)

# ── Main columns ─────────────────────────────────────────────────
c1, c2, c3 = st.columns([1.2,1,1])

with c1:
    st.markdown("### Prediction")
    if pred == 1:
        st.markdown('<div class="risk-high"><div style="font-size:38px">⚠️</div><div style="font-size:20px;font-weight:700;color:#C62828;margin:6px 0">AT-RISK STUDENT</div><div style="font-size:12px;color:#B71C1C">तत्काल ध्यान आवश्यक | Immediate attention needed</div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="risk-low"><div style="font-size:38px">✅</div><div style="font-size:20px;font-weight:700;color:#2E7D32;margin:6px 0">NOT AT RISK</div><div style="font-size:12px;color:#1B5E20">Passing trajectory — keep monitoring</div></div>', unsafe_allow_html=True)

with c2:
    st.markdown("### Risk Score")
    color = "#E53935" if risk_pct > 50 else "#43A047"
    st.markdown(f'<div style="text-align:center;padding:14px"><div style="font-size:52px;font-weight:700;color:{color}">{risk_pct}%</div><div style="font-size:12px;color:#6b6a65">chance of failing</div></div>', unsafe_allow_html=True)
    st.progress(float(prob))
    if risk_pct < 25:   st.success("🟢 Low Risk")
    elif risk_pct < 50: st.warning("🟡 Moderate Risk")
    elif risk_pct < 75: st.warning("🟠 High Risk")
    else:               st.error("🔴 Critical Risk")

with c3:
    st.markdown("### Subject Performance")
    total = math_m + sci_m + eng_m + hindi_m + social_m
    pct   = round(total / 500 * 100, 1)
    st.metric("Overall %", f"{pct}%")
    sub_names = [s1_label, s2_label, s3_label, s4_label, s5_label]
    failed = [n for n,m in zip(sub_names,[math_m,sci_m,eng_m,hindi_m,social_m]) if m < 33]
    if failed:
        st.error(f"Failing: {', '.join(failed)}")
    else:
        st.success("Passing all subjects ✅")

# ── SHAP ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔍 Why This Prediction? (SHAP Explainability)")

try:
    exp = get_explainer(model)
    sv  = exp.shap_values(inp)
    
    # Handle varying returned shapes of shap_values
    if isinstance(sv, list):
        sv_target = sv[1] if len(sv) > 1 else sv[0]
    elif len(np.shape(sv)) == 3:
        sv_target = sv[:, :, 1] if np.shape(sv)[2] > 1 else sv[:, :, 0]
    else:
        sv_target = sv
        
    # Handle varying expected_value formats
    if isinstance(exp.expected_value, (list, np.ndarray)) and len(exp.expected_value) > 1:
        base_val = exp.expected_value[1]
    elif isinstance(exp.expected_value, (list, np.ndarray)):
        base_val = exp.expected_value[0]
    else:
        base_val = exp.expected_value

    sv1 = sv_target[:, :len(FEATURES)] if len(np.shape(sv_target)) == 2 and np.shape(sv_target)[1] > len(FEATURES) else sv_target

    shap_df = pd.DataFrame({
        'Factor': LABELS, 'Value': inp.iloc[0].values, 'Impact': sv1[0]
    }).sort_values('Impact', key=abs, ascending=False)

    ca, cb = st.columns(2)
    with ca:
        st.markdown("**Factors increasing risk 🔴**")
        for _, r in shap_df[shap_df['Impact']>0].head(4).iterrows():
            st.markdown(f"**{r['Factor']}** = {r['Value']} → `+{r['Impact']:.4f}`")
    with cb:
        st.markdown("**Factors protecting from risk 🟢**")
        for _, r in shap_df[shap_df['Impact']<0].head(4).iterrows():
            st.markdown(f"**{r['Factor']}** = {r['Value']} → `{r['Impact']:.4f}`")

    explanation = shap.Explanation(
        values=sv1[0], base_values=base_val,
        data=inp.iloc[0].values, feature_names=LABELS)
    fig, _ = plt.subplots(figsize=(10,4))
    shap.waterfall_plot(explanation, show=False, max_display=12)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

except Exception as e:
    st.warning(f"SHAP: {e}")

# ── Recommendations ───────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📌 Teacher Recommendations")
recs = []
if attendance < 75:  recs.append("🔴 **Attendance < 75%** — May be barred from board exams. Contact parents immediately.")
if attendance < 60:  recs.append("🔴 **Critical attendance (<60%)** — File report to principal.")
if prev_year < 40:   recs.append("🔴 **Poor previous year %** — Refer to school counsellor.")
if any(m < 33 for m in [math_m,sci_m,eng_m,hindi_m,social_m]):
    recs.append("🔴 **Failing in one or more subjects** — Arrange remedial classes immediately.")
if study_hours < 2:  recs.append("🟡 **Very low study time** — Share structured study timetable.")
if not tuition and pred==1: recs.append("🟡 **No private tuition** — Suggest free DIKSHA/NCERT resources.")
if not internet:     recs.append("🟡 **No internet** — Direct to school library or DIKSHA offline app.")
if single_par:       recs.append("🟡 **Single parent** — May need extra emotional support.")
if fam_income <= 2:  recs.append("🟡 **Low income** — Check NSP scholarships: scholarships.gov.in")
if std_class in [11,12] and stream and "Science" in stream:
    if sci_m < 40 or hindi_m < 40:
        recs.append("🟡 **Weak Physics/Chemistry** — Critical for JEE/NEET. Suggest extra practice.")

if not recs:
    st.success("✅ No immediate concerns. Monitor at next assessment.")
else:
    for r in recs: st.markdown(r)

with st.expander("📎 Free Indian Education Resources"):
    st.markdown("""
    - **DIKSHA (NCERT Free)**: diksha.gov.in
    - **CBSE Sample Papers**: cbseacademic.nic.in
    - **NCERT Free PDFs**: ncert.nic.in
    - **National Scholarship**: scholarships.gov.in
    - **PM e-Vidya**: pmvidya.gov.in
    """)

with st.expander("ℹ️ About This System"):
    st.markdown("""
    **Model**: Random Forest (100 trees) | **Dataset**: 1000 Indian students (synthetic, CBSE context)
    **Pass Rule**: Any subject < 33 OR overall < 33% = At-Risk (CBSE standard)
    **Explainability**: SHAP | **Deployed**: https://student-risk-predictor-2.onrender.com
    **Project**: B.Tech 4th Sem | IP University | ML + AI Subject
    """)

st.markdown('<div class="footer">Student At-Risk Early Warning System | IP University B.Tech Project | CBSE/ICSE/State Board | Class 9–12</div>', unsafe_allow_html=True)