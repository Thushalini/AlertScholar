# app.py
# Streamlit app — Modern UI + PDF/CSV exports for teachers
# Run:  streamlit run app.py
# Reqs: pip install streamlit joblib reportlab pandas numpy

import os, io, zipfile, joblib, numpy as np, pandas as pd, streamlit as st
from datetime import datetime
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

st.set_page_config(page_title="🎓 Early-Warning — At-Risk Students", page_icon="🎓", layout="wide")

# =========================
# Styles
# =========================
st.markdown("""
<style>
:root { --brand:#2563eb; --bg:#0b1020; }
[data-testid="stAppViewContainer"] { background: linear-gradient(180deg,#0b1020 0%,#0f172a 100%); }
h1,h2,h3,h4,h5, .stMarkdown p, .stCaption, .st-emotion-cache-16txtl3 { color:#e5e7eb !important; }
.small {font-size:.85rem;color:#9ca3af;}
.kpi {border:1px solid #1f2937;border-radius:16px;padding:18px;background:#0b1329;}
.badge {display:inline-block;padding:.35rem .6rem;border-radius:9999px;font-weight:600}
.badge.green {background:#DCFCE7;color:#166534}
.badge.red {background:#FEE2E2;color:#991B1B}
.card {border:1px solid #1f2937;border-radius:18px;padding:20px;background:#0b1329;}
hr {border-top:1px solid #1f2937;}
footer {visibility:hidden}
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
left, right = st.columns([0.75, 0.25])
with left:
    st.title("🎓 Early-Warning & Intervention")
    st.caption("Identify at-risk students early, recommend targeted support, and export ready-to-send reports.")
with right:
    st.markdown("<div class='card'><b>School Mode</b><br><span class='small'>Designed for class teachers & grade heads</span></div>", unsafe_allow_html=True)

# =========================
# Config & Load model
# =========================
MODEL_PATH = os.getenv("MODEL_PATH", "model/logistic_regression_model.pkl")

@st.cache_resource(show_spinner=True)
def load_model(path: str):
    model = joblib.load(path)
    meta = getattr(model, "meta_", {}) or {}
    return model, meta

try:
    model, meta = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Couldn't load model at `{MODEL_PATH}`.\n\n{e}")
    st.stop()

THRESHOLD = float(meta.get("threshold", 0.5))
POS_LABEL = meta.get("positive_label", 1)
EXPECTED = list(getattr(model, "feature_names_in_", meta.get("feature_names_", [])))

# =========================
# Encoding & Prep
# =========================
def _encode_inputs(df: pd.DataFrame, meta_: dict) -> pd.DataFrame:
    df = df.copy()
    pe_map = meta_.get("parental_education_mapping") or {
        "high school": 0,
        "some college": 1,
        "associate's degree": 2,
        "bachelor's degree": 3,
        "master's degree": 4,
    }
    if "parental_level_of_education" in df.columns:
        df["parental_education_encoded"] = (
            df["parental_level_of_education"].astype(str).str.strip().str.lower().map(pe_map)
        )
        mode_val = pd.Series(list(pe_map.values())).mode().iat[0]
        df["parental_education_encoded"] = df["parental_education_encoded"].fillna(mode_val)
    # gender not in UI -> stable default
    default_gender_code = meta_.get("gender_encoded_default", 0)
    if "gender_encoded" not in df.columns:
        df["gender_encoded"] = default_gender_code
    return df

def _align_to_expected(df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    if not expected:
        raise RuntimeError("Model has no feature names. Store them at training time in meta['feature_names_'].")
    lower_to_exp = {c.lower(): c for c in expected}
    renamed = {}
    for c in df.columns:
        if c.lower() in lower_to_exp:
            renamed[c] = lower_to_exp[c.lower()]
    df = df.rename(columns=renamed)
    for c in expected:
        if c not in df.columns:
            df[c] = np.nan
    return df[expected]

def _recompute_total(row):
    t = row.get("total_score", np.nan)
    try:
        t = float(t)
    except Exception:
        t = np.nan
    if np.isnan(t) or t < 0 or t > 400:
        return float(row.get("math_score", 0)) + float(row.get("reading_score", 0)) + \
               float(row.get("writing_score", 0)) + float(row.get("science_score", 0))
    return t

def _prepare_for_model(df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    df = _encode_inputs(df, meta)
    for c in expected:
        if c not in df.columns:
            df[c] = 0
    df = df[expected]
    return df.fillna(0)

def _safe_predict_proba(df: pd.DataFrame) -> np.ndarray:
    X = _align_to_expected(df, EXPECTED)
    probas = model.predict_proba(X)
    classes_ = getattr(model, "classes_", None)
    if classes_ is not None:
        if POS_LABEL in list(classes_):
            pos_idx = int(np.where(classes_ == POS_LABEL)[0][0])
        else:
            if len(classes_) == 2:
                try:
                    pos_idx = int(np.argmax(classes_.astype(float)))
                except Exception:
                    pos_idx = 1
            else:
                raise RuntimeError(f"Multiclass detected; set meta['positive_label'] among {list(classes_)}.")
    else:
        pos_idx = 1
    return probas[:, pos_idx]

# =========================
# Interventions
# =========================
def interventions(row: dict):
    tips = []
    if float(row.get("reading_score", 100)) < 60:
        tips.append("2× weekly reading labs + peer tutoring.")
    if float(row.get("math_score", 100)) < 60:
        tips.append("Math clinic + daily 30-min practice set A.")
    if str(row.get("test_preparation_course", "0")) in ("0", "No", "no", "false", "False"):
        tips.append("Enroll in test-prep workshop; monitor completion.")
    if float(row.get("total_score", 999)) < 240:
        tips.append("Advisor meeting + parent briefing + weekly check-ins.")
    return tips or ["General study skills session; monitor for 2 weeks."]

# =========================
# PDF Generation
# =========================
def build_parent_letter_pdf(student: dict, proba: float, threshold: float, tips: list) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    title = ParagraphStyle('title', parent=styles['Title'], fontSize=20, textColor=colors.HexColor("#111827"))
    normal = ParagraphStyle('n', parent=styles['BodyText'], fontSize=11, leading=16, textColor=colors.HexColor("#111827"))
    small = ParagraphStyle('s', parent=styles['BodyText'], fontSize=9, leading=14, textColor=colors.HexColor("#6b7280"))

    elements = []
    elements.append(Paragraph("Early-Warning & Intervention Report", title))
    elements.append(Paragraph(datetime.now().strftime("%B %d, %Y"), small))
    elements.append(Spacer(1, 10))

    # Student summary table
    table_data = [
        ["Student Name", student.get("student_name","—")],
        ["Class/Grade", student.get("class_name","—")],
        ["Total Score (out of 400)", f"{student.get('total_score',0):.1f}"],
        ["Risk Probability", f"{proba:.2%}"],
        ["Operating Threshold", f"{threshold:.2f}"],
        ["Risk Status", "AT RISK" if proba >= threshold else "LOW RISK"]
    ]
    tbl = Table(table_data, hAlign="LEFT", colWidths=[170, 340])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(1,0), colors.HexColor("#eef2ff")),
        ("BOX",(0,0),(-1,-1), 0.5, colors.HexColor("#c7d2fe")),
        ("INNERGRID",(0,0),(-1,-1), 0.5, colors.HexColor("#c7d2fe")),
        ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, colors.HexColor("#f8fafc")])
    ]))
    elements += [tbl, Spacer(1, 12)]

    elements.append(Paragraph("<b>Recommended Actions (next 2–4 weeks)</b>", normal))
    for t in tips:
        elements.append(Paragraph(f"• {t}", normal))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(
        "This report supports teachers and parents to plan timely interventions. It should be used "
        "together with classroom observations and school policy.", small))

    doc.build(elements)
    return buf.getvalue()

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    THRESHOLD = st.slider("Operating Threshold", min_value=0.10, max_value=0.90, value=float(THRESHOLD), step=0.01)
    st.markdown("---")
    st.markdown("**About**\n\nThis tool flags students who may need support and generates parent-friendly reports.", unsafe_allow_html=True)
    st.markdown("<span class='small'>Built for teachers in Sri Lanka to reduce admin workload.</span>", unsafe_allow_html=True)

# =========================
# Tabs
# =========================
tab1, tab2, tab3 = st.tabs(["🔎 Single Prediction", "📦 Batch (CSV)", "📄 Model Card"])

# ---------- Single ----------
with tab1:
    st.markdown("#### Quick Check")
    base_left, base_right = st.columns(2)
    with base_left:
        student_name = st.text_input("Student Name (optional)", "")
        class_name = st.text_input("Class / Grade (optional)", "")
        parental = st.selectbox("Parental Level of Education",
                                ["high school","some college","associate's degree","bachelor's degree","master's degree"])
        lunch = st.selectbox("Lunch (1=standard, 0=reduced/free)", [1, 0], index=0)
        prep = st.selectbox("Test Preparation Course (1=completed, 0=none)", [1, 0], index=1)
    with base_right:
        math = st.number_input("Math Score", min_value=0.0, max_value=100.0, value=65.0, step=1.0)
        read = st.number_input("Reading Score", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
        write = st.number_input("Writing Score", min_value=0.0, max_value=100.0, value=68.0, step=1.0)
        sci = st.number_input("Science Score", min_value=0.0, max_value=100.0, value=66.0, step=1.0)

    total = math + read + write + sci
    st.markdown(f"**Total Score (auto):** {total:.1f}")

    run_cols = st.columns([1,1,1])
    go = run_cols[0].button("🔮 Predict", use_container_width=True)

    if go:
        row = {
            "student_name": student_name,
            "class_name": class_name,
            "parental_level_of_education": parental,
            "lunch": lunch,
            "test_preparation_course": prep,
            "math_score": math,
            "reading_score": read,
            "writing_score": write,
            "science_score": sci,
            "total_score": total
        }
        X = pd.DataFrame([row])
        X = _prepare_for_model(X, EXPECTED)
        X["total_score"] = X.apply(lambda r: _recompute_total(r), axis=1)

        try:
            proba = float(_safe_predict_proba(X)[0])
        except Exception as e:
            st.error(f"Prediction failed due to feature mismatch.\n\nExpected: {EXPECTED}\nIncoming: {list(X.columns)}\n\n{e}")
            st.stop()

        at_risk = proba >= THRESHOLD
        k1, k2, k3 = st.columns([1,1,1])
        with k1: st.markdown(f"<div class='kpi'><b>Risk Probability</b><h2>{proba:.2%}</h2></div>", unsafe_allow_html=True)
        with k2: st.markdown(f"<div class='kpi'><b>Threshold</b><h2>{THRESHOLD:.2f}</h2></div>", unsafe_allow_html=True)
        with k3:
            badge = "<span class='badge red'>AT RISK</span>" if at_risk else "<span class='badge green'>LOW RISK</span>"
            st.markdown(f"<div class='kpi'><b>Status</b><h2>{badge}</h2></div>", unsafe_allow_html=True)

        st.markdown("##### Recommended Interventions")
        tips = interventions({**row, "total_score": X.loc[0,"total_score"]})
        for t in tips:
            st.write("• " + t)

        st.divider()
        c1, c2 = st.columns(2)

        # Single CSV
        with c1:
            single_out = pd.DataFrame([{
                "date": datetime.now().strftime("%Y-%m-%d"),
                "student_name": student_name,
                "class_name": class_name,
                "math_score": math,
                "reading_score": read,
                "writing_score": write,
                "science_score": sci,
                "total_score": float(X.loc[0,"total_score"]),
                "risk_probability": round(proba,4),
                "at_risk": int(at_risk),
                "first_intervention": tips[0] if tips else ""
            }])
            csv_buf = single_out.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV (single)", data=csv_buf, file_name="student_risk_single.csv",
                               mime="text/csv", use_container_width=True)

        # Single PDF
        with c2:
            pdf_bytes = build_parent_letter_pdf(
                {"student_name": student_name, "class_name": class_name, "total_score": float(X.loc[0,"total_score"])},
                proba, THRESHOLD, tips
            )
            st.download_button("⬇️ Download Parent Letter (PDF)", data=pdf_bytes,
                               file_name=f"{(student_name or 'student').replace(' ','_')}_parent_letter.pdf",
                               mime="application/pdf", use_container_width=True)

# ---------- Batch ----------
with tab2:
    st.markdown("#### Batch Upload")
    st.caption("Expected columns (case-insensitive; extra columns ignored): "
               "`student_name (optional), class_name (optional), parental_level_of_education, lunch, test_preparation_course, math_score, reading_score, writing_score, science_score, total_score`")

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)

        needed = ["student_name","class_name","parental_level_of_education","lunch","test_preparation_course",
                  "math_score","reading_score","writing_score","science_score","total_score"]
        need_lower = {c.lower() for c in needed}
        present = [c for c in df.columns if c.lower() in need_lower]
        Xraw = df[present].copy()

        if "total_score" in Xraw.columns:
            Xraw["total_score"] = Xraw.apply(lambda r: _recompute_total(r), axis=1)
        else:
            Xraw["total_score"] = Xraw.apply(lambda r: _recompute_total(r), axis=1)

        try:
            X_model = _prepare_for_model(Xraw, EXPECTED)
            probs = _safe_predict_proba(X_model)
        except Exception as e:
            st.error(f"Batch prediction failed.\n\nExpected: {EXPECTED}\nIncoming: {list(Xraw.columns)}\n\n{e}")
            st.stop()

        preds = (probs >= THRESHOLD).astype(int)
        out = df.copy()
        out["risk_probability"] = np.round(probs, 4)
        out["at_risk"] = preds

        def _first_tip(i: int):
            row = {
                "reading_score": out.loc[i, "reading_score"] if "reading_score" in out.columns else None,
                "math_score": out.loc[i, "math_score"] if "math_score" in out.columns else None,
                "test_preparation_course": out.loc[i, "test_preparation_course"] if "test_preparation_course" in out.columns else None,
                "total_score": out.loc[i, "total_score"] if "total_score" in out.columns else None,
            }
            return interventions(row)[0]
        out["first_intervention"] = [_first_tip(i) for i in range(len(out))]

        st.dataframe(out.head(20), use_container_width=True)

        # CSV Export
        buf = BytesIO()
        out.to_csv(buf, index=False)
        st.download_button("⬇️ Download Results CSV", data=buf.getvalue(),
                           file_name="batch_predictions.csv", mime="text/csv", use_container_width=True)

        # ZIP of PDFs (Parent letters)
        make_zip = st.checkbox("Generate a ZIP of parent letters (PDF) for all rows", value=False)
        if make_zip:
            zip_buf = BytesIO()
            with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for i in range(len(out)):
                    sname = str(out.loc[i, "student_name"]) if "student_name" in out.columns else f"student_{i+1}"
                    cname = str(out.loc[i, "class_name"]) if "class_name" in out.columns else ""
                    total_i = float(out.loc[i, "total_score"]) if "total_score" in out.columns else float(
                        (out.loc[i, "math_score"] if "math_score" in out.columns else 0) +
                        (out.loc[i, "reading_score"] if "reading_score" in out.columns else 0) +
                        (out.loc[i, "writing_score"] if "writing_score" in out.columns else 0) +
                        (out.loc[i, "science_score"] if "science_score" in out.columns else 0)
                    )
                    tips_i = interventions({
                        "reading_score": out.loc[i, "reading_score"] if "reading_score" in out.columns else None,
                        "math_score": out.loc[i, "math_score"] if "math_score" in out.columns else None,
                        "test_preparation_course": out.loc[i, "test_preparation_course"] if "test_preparation_course" in out.columns else None,
                        "total_score": total_i
                    })
                    pdf = build_parent_letter_pdf({"student_name": sname, "class_name": cname, "total_score": total_i},
                                                  float(probs[i]), THRESHOLD, tips_i)
                    safe_name = (sname or f"student_{i+1}").replace(" ", "_")
                    zf.writestr(f"{safe_name}_parent_letter.pdf", pdf)
            st.download_button("⬇️ Download ZIP of Parent Letters (PDFs)",
                               data=zip_buf.getvalue(), file_name="parent_letters.zip",
                               mime="application/zip", use_container_width=True)

# ---------- Model Card ----------
# ---------- Model Card ----------
with tab3:
    st.markdown("#### Model Card")
    st.markdown(
        f"**Model:** {meta.get('model_name','Logistic Regression (scikit-learn)')}  \n"
        f"**Target:** {meta.get('label_definition','Identifying at-risk students')}  \n"
        f"**Threshold:** {THRESHOLD:.2f}  \n"
        f"**Numeric features:** {', '.join(meta.get('num_features', [])) or 'math_score, reading_score, writing_score, science_score, total_score'}  \n"
    )

    # categorical features in bullet points
    cats = meta.get('cat_features', []) or [
        "lunch — Indicates whether the student receives a standard paid lunch (1) or is on a reduced/free school meal program (0)",
        "test_preparation_course — Completed (1) vs none (0)",
        "parental_level_of_education — Encoded categories of parent education level"
    ]
    st.markdown("**Categorical features:**")
    for c in cats:
        st.markdown(f"- {c}")

    st.markdown("""
**Intended Use**  
Support teachers by flagging students who may need early interventions.

**Operational Notes**  
• Probabilities are calibrated; threshold chosen to favor recall.  
• Sensitive attributes are not used for predictions.  
• Use alongside teacher judgment and school policy.
""")

