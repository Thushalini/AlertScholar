import os, joblib, numpy as np, pandas as pd, streamlit as st
from io import BytesIO

st.set_page_config(page_title="🎓 Early-Warning — At-Risk Students", page_icon="🎓", layout="wide")

# ---- Minimal styling ----
st.markdown("""
<style>
.small {font-size: 0.85rem; color: #6b7280;}
.badge {display:inline-block;padding:0.35rem 0.6rem;border-radius:9999px;font-weight:600}
.badge.green {background:#DCFCE7;color:#166534}
.badge.red {background:#FEE2E2;color:#991B1B}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.title("🎓 Early-Warning & Intervention")
st.caption("Identify students who may be at risk early and recommend targeted support.")

# ---------- Config ----------
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

# Operating threshold + expected features + positive label
THRESHOLD = float(meta.get("threshold", 0.5))
POS_LABEL = meta.get("positive_label", 1)  # adjust if you saved "at_risk" or similar
EXPECTED = list(getattr(model, "feature_names_in_", meta.get("feature_names_", [])))

# ---------- Encoding + Prep Helpers ----------
def _encode_inputs(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Create the encoded columns expected by the trained model."""
    df = df.copy()

    # parental_education_encoded
    pe_map = meta.get("parental_education_mapping") or {
        "high school": 0,
        "some college": 1,
        "associate's degree": 2,
        "bachelor's degree": 3,
        "master's degree": 4,
    }
    if "parental_level_of_education" in df.columns:
        df["parental_education_encoded"] = (
            df["parental_level_of_education"].str.strip().str.lower().map(pe_map)
        )
    if "parental_education_encoded" in df.columns:
        mode_val = pd.Series(list(pe_map.values())).mode().iat[0]
        df["parental_education_encoded"] = df["parental_education_encoded"].fillna(mode_val)

    # gender_encoded — UI doesn’t ask, so use default
    default_gender_code = meta.get("gender_encoded_default", 0)
    if "gender_encoded" not in df.columns:
        df["gender_encoded"] = default_gender_code

    return df

def _prepare_for_model(df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    """Align to expected features & fill missing values."""
    df = _encode_inputs(df, meta)
    for c in expected:
        if c not in df.columns:
            df[c] = 0
    df = df[expected]
    return df.fillna(0)

# ---------- Helpers ----------
def _recompute_total(row):
    """Return a sane total_score from the four subjects when missing or out of range."""
    t = row.get("total_score", np.nan)
    try:
        t = float(t)
    except Exception:
        t = np.nan
    if np.isnan(t) or t < 0 or t > 400:
        return float(row.get("math_score", 0)) + float(row.get("reading_score", 0)) + \
               float(row.get("writing_score", 0)) + float(row.get("science_score", 0))
    return t

def _align_to_expected(df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    """
    Align columns by:
      1) Case-insensitive mapping to expected names (left-biased)
      2) Add missing expected columns filled with NaN
      3) Reorder to expected order
    """
    if not expected:
        # If model was fit on numpy arrays, you must persist expected names during training.
        raise RuntimeError("Model has no feature_names_in_. Save expected features to model.meta_['feature_names_'].")

    # Map incoming columns (case-insensitive) -> expected exact names
    lower_to_exp = {c.lower(): c for c in expected}
    renamed = {}
    for c in df.columns:
        key = c.lower()
        if key in lower_to_exp:
            renamed[c] = lower_to_exp[key]
    df = df.rename(columns=renamed)

    # Add any missing expected columns
    for c in expected:
        if c not in df.columns:
            df[c] = np.nan

    # Subset & reorder
    return df[expected]

def _safe_predict_proba(df: pd.DataFrame) -> np.ndarray:
    """Predict proba after aligning columns and finding the positive class index safely."""
    X = _align_to_expected(df, EXPECTED)
    probas = model.predict_proba(X)  # shape: (n, n_classes)

    # robustly select the positive class index
    classes_ = getattr(model, "classes_", None)
    if classes_ is not None:
        # try meta-specified positive label first
        if POS_LABEL in list(classes_):
            pos_idx = int(np.where(classes_ == POS_LABEL)[0][0])
        else:
            # common fallbacks
            if len(classes_) == 2:
                # if labels are [0,1] or ["low","at_risk"] etc.. default to the "max" if numeric else last
                try:
                    pos_idx = int(np.argmax(classes_.astype(float)))
                except Exception:
                    pos_idx = 1
            else:
                raise RuntimeError(f"Multiclass model detected; set meta['positive_label'] to a valid class among {list(classes_)}.")
    else:
        # last resort
        pos_idx = 1

    return probas[:, pos_idx]

# --------- Simple rules → interventions ---------
def interventions(row: dict):
    tips = []
    if float(row.get("reading_score", 100)) < 60:
        tips.append("Assign 2× weekly reading labs + peer tutoring.")
    if float(row.get("math_score", 100)) < 60:
        tips.append("Math clinic referral + practice set A (30 mins/day).")
    if str(row.get("test_preparation_course", "0")) in ("0", "No", "no", "false", "False"):
        tips.append("Enroll in test-prep workshop; track completion.")
    if float(row.get("total_score", 999)) < 240:
        tips.append("Advisor meeting; parent briefing; weekly progress check.")
    return tips or ["General study skills session; monitor for 2 weeks."]

# --------- Tabs ---------
tab1, tab2, tab3 = st.tabs(["Single Prediction", "Batch (CSV)", "Model Card"])

with tab1:
    st.subheader("Quick Check")
    col_left, col_right = st.columns(2)

    with col_left:
        parental = st.selectbox(
            "Parental Level of Education",
            ["high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
        )
        lunch = st.selectbox("Lunch (1=standard, 0=reduced/free)", [1, 0])
        prep = st.selectbox("Test Preparation Course (1=completed, 0=none)", [1, 0])

    with col_right:
        math = st.number_input("Math Score", min_value=0.0, max_value=100.0, value=65.0, step=1.0)
        read = st.number_input("Reading Score", min_value=0.0, max_value=100.0, value=70.0, step=1.0)
        write = st.number_input("Writing Score", min_value=0.0, max_value=100.0, value=68.0, step=1.0)
        sci = st.number_input("Science Score", min_value=0.0, max_value=100.0, value=66.0, step=1.0)

    total = math + read + write + sci
    st.write(f"**Total Score (auto):** {total:.1f}")

    if st.button("Predict Risk", use_container_width=True):
        # Build one-row input with ORIGINAL training column names (raw values; pipeline will encode)
        row = {
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

        # Keep total sane
        X["total_score"] = X.apply(lambda r: _recompute_total(r), axis=1)

        try:
            proba = float(_safe_predict_proba(X)[0])
        except Exception as e:
            st.error(f"Prediction failed due to feature mismatch.\n\nExpected: {EXPECTED}\nIncoming: {list(X.columns)}\n\n{e}")
            st.stop()

        at_risk = proba >= THRESHOLD

        cols = st.columns([1, 1])
        with cols[0]:
            st.metric("Risk Probability", f"{proba:.2%}")
        with cols[1]:
            st.metric("Operating Threshold", f"{THRESHOLD:.2f}")

        st.markdown(
            f"<span class='badge {'red' if at_risk else 'green'}'>{'AT RISK' if at_risk else 'LOW RISK'}</span>",
            unsafe_allow_html=True
        )

        st.divider()
        st.subheader("Recommended Interventions")
        for tip in interventions({**row, "total_score": X.loc[0, "total_score"]}):
            st.write("• " + tip)

with tab2:
    st.subheader("Batch Upload (CSV)")
    st.caption("Columns expected (case-insensitive, extra columns ignored): "
               "`parental_level_of_education, lunch, test_preparation_course, math_score, reading_score, writing_score, science_score, total_score`")
    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)

        # Soft-pick needed columns by case-insensitive match (we'll align exactly later)
        needed = ["parental_level_of_education", "lunch", "test_preparation_course",
                  "math_score", "reading_score", "writing_score", "science_score", "total_score"]
        needed_lower = {c.lower() for c in needed}
        present = [c for c in df.columns if c.lower() in needed_lower]
        Xraw = df[present].copy()

        # Recompute total if missing/invalid
        if "total_score" in Xraw.columns:
            Xraw["total_score"] = Xraw.apply(lambda r: _recompute_total(r), axis=1)
        else:
            Xraw["total_score"] = Xraw.apply(lambda r: _recompute_total(r), axis=1)

        # Align and predict
        try:
            X_model = _prepare_for_model(Xraw, EXPECTED)
            probs = model.predict_proba(X_model)[:, 1]
        except Exception as e:
            st.error(f"Batch prediction failed.\n\nExpected: {EXPECTED}\nIncoming: {list(Xraw.columns)}\n\n{e}")
            st.stop()

        preds = (probs >= THRESHOLD).astype(int)

        out = df.copy()
        out["risk_probability"] = np.round(probs, 4)
        out["at_risk"] = preds

        # First intervention suggestion (lightweight)
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

        # Download
        buf = BytesIO()
        out.to_csv(buf, index=False)
        st.download_button(
            "Download results CSV",
            data=buf.getvalue(),
            file_name="batch_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

with tab3:
    st.subheader("Model Card")
    st.write(
        f"**Model:** {meta.get('model_name','N/A')}  \n"
        f"**Target:** {meta.get('label_definition','N/A')}  \n"
        f"**Threshold:** {meta.get('threshold','N/A')}  \n"
        f"**Numeric features:** {', '.join(meta.get('num_features', []))}  \n"
        f"**Categorical features:** {', '.join(meta.get('cat_features', []))}  \n"
    )
    st.markdown("""
**Intended Use**  
Support teachers by flagging students who may need early interventions.

**Important Notes**  
• Probabilities are calibrated; threshold chosen to favor recall.  
• Sensitive attributes are not used for predictions.  
• Use alongside teacher judgment and school policy.
""")
