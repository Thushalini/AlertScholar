# app.py
# Streamlit — Black UI + Red accents, polished KPI cards (No LLM usage)
# Run:  streamlit run app.py
# Reqs: pip install streamlit joblib reportlab pandas numpy python-dotenv

import os, io, zipfile, joblib, numpy as np, pandas as pd, streamlit as st
import glob
from pathlib import Path
from datetime import datetime
from io import BytesIO

from pathlib import Path, PurePath

# Load .env if present
# try:
#   from dotenv import load_dotenv
#   load_dotenv()
# except Exception:
#   pass

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

st.set_page_config(page_title="Early-Warning — At-Risk Students", page_icon="🎓", layout="centered")

# BASE = Path(__file__).resolve().parent
# MODEL_PATH = BASE / "model" / "logistic_regression_model.pkl"

# # debug (one time)
# print("HERE:", BASE)
# print("TREE:", glob.glob(str(BASE / "**/*"), recursive=True))

# # one-time debug
# st.caption(f"Looking for model at: {MODEL_PATH}")
# st.caption(f"Exists? {MODEL_PATH.exists()}")
# if (BASE / "model").exists():
#     st.caption(f"Files in /model: {[p.name for p in (BASE/'model').iterdir()]}")
# else:
#     st.caption("No /model directory found next to app.py")
# assert MODEL_PATH.exists(), f"Missing: {MODEL_PATH}"
# model = joblib.load(str(MODEL_PATH))
# ---- Model path (robust) ----

BASE = Path(__file__).resolve().parent

# 1) Allow override via env var
env_path = os.getenv("MODEL_PATH")

# 2) Common candidate locations
candidates = [
    env_path,
    BASE / "models" / "logistic_regression_model.pkl",
    BASE / "logistic_regression_model.pkl",
    BASE.parent / "models" / "logistic_regression_model.pkl",
    # Streamlit Cloud/Spaces-style absolute path (seen in your logs)
    Path("/mount/src/alertscholar/student-risk-system/models/logistic_regression_model.pkl"),
]

# 3) Last-resort: shallow search under project root (max 2 levels)
if not any(p and Path(p).exists() for p in candidates):
    for p in BASE.rglob("*.pkl"):
        if "models" in str(p.parent).lower() and "logistic_regression_model" in p.name:
            candidates.insert(0, p)
            break

# 4) Pick the first existing path
MODEL_PATH = next((Path(p) for p in candidates if p and Path(p).exists()), None)

with st.expander("Model loading details", expanded=False):
    st.caption(f"Checked candidates: {[str(c) for c in candidates if c]}")
    st.caption(f"Selected: {str(MODEL_PATH) if MODEL_PATH else 'None'}")

if MODEL_PATH is None:
    st.error("Couldn't find logistic_regression_model.pkl. Place it in ./models/ next to app.py "
             "or set MODEL_PATH env var to the absolute file path.")
    st.stop()

# 5) Load the model
try:
    model = joblib.load(str(MODEL_PATH))
except Exception as e:
    st.error(f"Failed to load model at {MODEL_PATH} → {e}")
    st.stop()


# =========================
# Styles (Black + Red) + Alignment helpers
# =========================
st.markdown("""
<style>
:root{
  --bg:#0a0a0a; 
  --panel:#121212; 
  --text:#f3f4f6; 
  --muted:#9ca3af;
  --accent:#ef4444; 
  --accent-strong:#b91c1c; 
  --border:#1f2937;
  --good:#10b981;
  --warn:#f59e0b;
  --bad:#ef4444;
}

/* Center & narrow the whole app */
.main .block-container{ 
  max-width: 980px; 
  padding-top: 1.5rem; 
}

/* App background + text color */
[data-testid="stAppViewContainer"]{ background: var(--bg) !important; }
[data-testid="stHeader"]{ background: transparent !important; }
h1,h2,h3,h4,h5,h6, p, label, span, div, code, kbd, pre { color: var(--text) !important; }

/* ---------- Polished KPI cards ---------- */
.kpi{
  background: radial-gradient(1200px 600px at -20% -10%, rgba(239,68,68,.07), transparent),
              radial-gradient(1200px 600px at 120% 120%, rgba(37,99,235,.07), transparent),
              var(--panel);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 18px 20px;
  box-shadow: 0 10px 24px rgba(0,0,0,.35);
  min-height: 116px; 
  display:flex; 
  flex-direction:column; 
  justify-content:center;
  position: relative;
  overflow: hidden;
}
.kpi b{ color: var(--muted) !important; font-weight: 700; letter-spacing:.02em;}
.kpi .value{
  margin: 2px 0 0 0; 
  font-size: 2.2rem; 
  font-weight: 800; 
  letter-spacing:.5px; 
  line-height: 1.1;
  text-shadow: 0 1px 0 rgba(255,255,255,.03);
}
.kpi .pill{
  position:absolute; 
  right:14px; top:14px; 
  padding:6px 12px; 
  border-radius:9999px; 
  font-weight:800; 
  letter-spacing:.6px;
  border:1px solid rgba(255,255,255,.08);
}

/* Cards */
.card{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 18px 20px;
  box-shadow: 0 10px 24px rgba(0,0,0,.35);
}

/* Badges */
.badge {display:inline-block;padding:.35rem .6rem;border-radius:9999px;font-weight:700}
.badge.green {background:#064e3b;color:#bbf7d0;border:1px solid #065f46}
.badge.red {background:#7f1d1d;color:#fecaca;border:1px solid #991b1b}

/* ===== Inputs (neutral focus — no red) ===== */
.stTextInput > div > div,
.stNumberInput > div > div,
.stSelectbox [data-baseweb="select"] {
  background: #0f0f0f !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}
.stTextInput > div > div:focus-within,
.stNumberInput > div > div:focus-within,
.stSelectbox [data-baseweb="select"]:focus-within {
  border-color: #374151 !important; /* neutral slate */
  box-shadow: 0 0 0 2px rgba(148,163,184,.18) !important; /* soft gray glow */
}
.stTextInput input, .stNumberInput input, .stSelectbox input { 
  background: transparent !important; 
  color: var(--text) !important; 
  border: none !important;
  outline: none !important;
  box-shadow: none !important;
}

/* ===== Kill red borders coming from invalid state or focus-invalid ===== */
input:invalid,
input:user-invalid,
input:focus:invalid,
input:focus:user-invalid,
textarea:invalid,
textarea:user-invalid,
textarea:focus:invalid,
textarea:focus:user-invalid {
  box-shadow: none !important;
  outline: none !important;
  border: 1px solid var(--border) !important;
}
/* Some browsers mark the wrapper as invalid; keep it neutral */
.stTextInput div[aria-invalid="true"],
.stNumberInput div[aria-invalid="true"]{
  border-color: var(--border) !important;
  box-shadow: none !important;
}

/* ===== Remove number spinners & Streamlit +/- step buttons ===== */
.stNumberInput input[type=number]::-webkit-outer-spin-button,
.stNumberInput input[type=number]::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
.stNumberInput input[type=number] { -moz-appearance: textfield; }
.stNumberInput button { display: none !important; }

/* Radios / checkboxes */
input[type="radio"], input[type="checkbox"]{ accent-color: var(--accent) !important; }
.stRadio > div[role="radiogroup"]{ display:flex; gap:.5rem; flex-wrap:wrap; }
.stRadio > div[role="radiogroup"] label{
  background:#0f0f0f; border:1px solid var(--border); border-radius:9999px; padding:8px 14px;
}

/* Buttons (no red focus ring) */
.stButton > button{
  background: var(--accent) !important; 
  color:#fff !important;
  border:1px solid var(--accent-strong) !important; 
  border-radius:12px !important;
  height:44px; padding:0 .9rem !important; font-weight:700 !important;
}
.stButton > button:hover{ filter: brightness(1.05); }
.stButton > button:focus,
.stButton > button:focus-visible{
  outline:none !important;
  box-shadow:none !important;
  border-color: var(--accent-strong) !important; /* keep consistent, not red glow */
}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{ gap:.5rem; border-bottom:1px solid var(--border); }
.stTabs [data-baseweb="tab"]{
  background:#0f0f0f; color:var(--text); border:1px solid var(--border);
  border-bottom:none; border-radius:12px 12px 0 0; padding:.6rem 1rem;
}
.stTabs [aria-selected="true"]{
  background: var(--panel) !important; border-color: var(--accent) !important; 
  box-shadow: 0 -2px 0 var(--accent) inset;
}

/* Tables/DataFrames */
[data-testid="stTable"] table, .stDataFrame{ border-radius:12px; overflow:hidden; border:1px solid var(--border); }
th, td { color: var(--text) !important; }

/* HR + footer */
hr {border-top:1px solid var(--border);}
footer {visibility:hidden}
.small {font-size:.85rem;color:var(--muted);}
</style>
""", unsafe_allow_html=True)

# =========================
# Header
# =========================
left, right = st.columns([0.72, 0.28])
with left:
  st.title("🎓 AlertScholar")
  st.caption("Identify at-risk students early, recommend targeted support, and export ready-to-send reports.")
with right:
  st.markdown("<div class='card'><b>School Mode</b><br><span class='small'>Designed for class teachers & grade heads</span></div>", unsafe_allow_html=True)

# =========================
# Config & Load model
# =========================
# MODEL_PATH = os.getenv("MODEL_PATH", "models/logistic_regression_model.pkl")

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
# Encoding & Prep (gender-neutral prediction)
# =========================
def _encode_inputs(df: pd.DataFrame, meta_: dict) -> pd.DataFrame:
  df = df.copy()
  pe_map = meta_.get("parental_education_mapping") or {
    "high school": 0, "some college": 1, "associate's degree": 2,
    "bachelor's degree": 3, "master's degree": 4,
  }
  if "parental_level_of_education" in df.columns:
    df["parental_education_encoded"] = (
      df["parental_level_of_education"].astype(str).str.strip().str.lower().map(pe_map)
    )
    mode_val = pd.Series(list(pe_map.values())).mode().iat[0]
    df["parental_education_encoded"] = df["parental_education_encoded"].fillna(mode_val)

  # gender neutral for model
  default_gender_code = meta_.get("gender_encoded_default", 0)
  df["gender_encoded"] = default_gender_code
  return df

def _align_to_expected(df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
  if not expected:
    raise RuntimeError("Model has no feature names. Save them in meta['feature_names_'].")
  lower_to_exp = {c.lower(): c for c in expected}
  renamed = {c: lower_to_exp[c.lower()] for c in df.columns if c.lower() in lower_to_exp}
  df = df.rename(columns=renamed)
  for c in expected:
    if c not in df.columns:
      df[c] = np.nan
  return df[expected]

def _recompute_total(row):
  t = row.get("total_score", np.nan)
  try: t = float(t)
  except Exception: t = np.nan
  if np.isnan(t) or t < 0 or t > 400:
    return float(row.get("math_score", 0)) + float(row.get("reading_score", 0)) + \
           float(row.get("writing_score", 0)) + float(row.get("science_score", 0))
  return t

def _prepare_for_model(df: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
  df = _encode_inputs(df, meta)
  for c in expected:
    if c not in df.columns: df[c] = 0
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
        try: pos_idx = int(np.argmax(classes_.astype(float)))
        except Exception: pos_idx = 1
      else:
        raise RuntimeError(f"Multiclass detected; set meta['positive_label'] among {list(classes_)}.")
  else:
    pos_idx = 1
  return probas[:, pos_idx]

# =========================
# Dynamic, rule-based interventions (no LLM) — uses row + proba + threshold
# =========================
def interventions_dynamic(row: dict, proba: float, threshold: float):
  """
  Return a prioritized list of interventions based on:
  - Subject score gaps (Math/Reading/Writing/Science)
  - Total/average score
  - Test preparation completion
  - Lunch (reduced/free vs standard)
  - Parental level of education
  - Risk probability level vs threshold
  """
  def sev(score):
    try:
      s = float(score)
    except:
      return None
    if s < 40:  return ("high",  "Severe gap (<40)")
    if s < 60:  return ("medium","Moderate gap (40–59)")
    if s < 75:  return ("low",   "Mild gap (60–74)")
    return None

  def enrich(score):
    try:
      s = float(score)
    except:
      return False
    return s >= 85

  plan = []

  # Subject-level gaps
  subjects = [
    ("Math",    row.get("math_score")),
    ("Reading", row.get("reading_score")),
    ("Writing", row.get("writing_score")),
    ("Science", row.get("science_score")),
  ]
  for subject, val in subjects:
    level = sev(val)
    if level:
      priority, note = level
      if subject == "Math":
        steps = [
          "Enroll in math support block 3×/week (45 mins).",
          "Daily 30-minute practice with mistake log.",
          "Weekly mini-quiz; reteach missed objectives."
        ]
      elif subject == "Reading":
        steps = [
          "Guided reading 20 mins/day (leveled texts).",
          "2×/week vocabulary + comprehension drills.",
          "Peer reading aloud on alternate days."
        ]
      elif subject == "Writing":
        steps = [
          "Paragraph→essay scaffold 2×/week (intro, evidence, conclusion).",
          "Weekly 1:1 writing conference for targeted feedback.",
          "Use a pre-submit checklist (grammar + clarity)."
        ]
      else:
        steps = [
          "Weekly hands-on lab linking theory to practice.",
          "After each topic: concept map + 10 MCQs on Friday.",
          "One misconception-fix session/week from exit tickets."
        ]
      plan.append({
        "title": f"{subject} remediation",
        "why": f"{subject} score {float(val):.0f}. {note}.",
        "steps": steps,
        "owner": "Subject teacher + student",
        "when": "2–4 weeks (review weekly)",
        "kpis": [f"{subject} checkpoint +10 pts", "≥80% on weekly mini-quiz", "≤2 missing tasks"],
        "priority": priority
      })

  # Enrichment (strengths)
  for subject, val in subjects:
    if enrich(val):
      plan.append({
        "title": f"{subject} enrichment",
        "why": f"{subject} strength (≥85). Maintain momentum while addressing other gaps.",
        "steps": [
          "Add one challenge task per week (past papers/olympiad-style).",
          "Student leads a 5-min micro-teach on a mastered concept.",
          "Portfolio: collect best work; reflect on winning strategies."
        ],
        "owner": "Subject teacher + student",
        "when": "2–4 weeks",
        "kpis": [f"{subject} avg stays ≥85", "1 challenge task/week completed"],
        "priority": "low"
      })

  # Whole-student risk banding
  if proba >= threshold:
    if proba >= max(0.8, threshold + 0.2):
      when_txt = "1–2 weeks (intensive start)"
      prio = "high"
    elif proba >= threshold + 0.1:
      when_txt = "2–3 weeks"
      prio = "high"
    else:
      when_txt = "3–4 weeks"
      prio = "medium"

    try:
      total = float(row.get("total_score"))
    except:
      total = sum(float(x or 0) for _, x in subjects)

    plan.append({
      "title": "Advisor meeting + coordinated study plan",
      "why": f"Risk probability {proba:.0%} ≥ threshold {threshold:.2f}. Total {total:.0f}/400 indicates need for structured plan.",
      "steps": [
        "Set 3 SMART goals for next 2 weeks; share with parent/guardian.",
        "Daily 45-minute study block + task tracker (Mon–Fri).",
        "Friday progress review; adjust goals and materials."
      ],
      "owner": "Advisor + student (+ parent)",
      "when": when_txt,
      "kpis": ["Goals completed ≥80%", "Total score +30 in 4 weeks", "Attendance ≥95%"],
      "priority": prio
    })

  # Access & habits
  tprep_incomplete = str(row.get("test_preparation_course", "0")).strip().lower() in ("0","no","false","none","")
  if tprep_incomplete:
    plan.append({
      "title": "Join test-preparation program",
      "why": "Course improves timing, strategy, recall; currently not completed.",
      "steps": ["Enroll this week; attend all sessions.",
                "Do 2 timed practice sets/week and review errors.",
                "Track mock score improvement."],
      "owner": "Student + test-prep coordinator",
      "when": "Start now; monitor weekly",
      "kpis": ["+10 percentile in mocks", "100% attendance", "Error rate ↓ 30%"],
      "priority": "high" if proba >= threshold else "medium"
    })

  lunch_reduced = str(row.get("lunch","1")).strip().startswith("0")
  if lunch_reduced:
    plan.append({
      "title": "After-school tutoring & mentoring",
      "why": "Reduced/free lunch may signal access barriers; structured support helps consistency.",
      "steps": ["2×/week tutoring (math/writing focus).",
                "Assign mentor for weekly check-ins.",
                "Provide study materials + quiet study slot."],
      "owner": "Counselor + after-school staff",
      "when": "2–6 weeks",
      "kpis": ["Homework completion ≥90%", "Attendance ≥95%", "Subject quiz avg ≥75%"],
      "priority": "medium"
    })

  ple = str(row.get("parental_level_of_education","")).lower()
  if ple in ["high school", "some college"]:
    plan.append({
      "title": "Parent partnership plan",
      "why": f"Parental education: {ple}. Guided home routines can boost consistency.",
      "steps": ["Share a 20-min nightly study template.",
                "Weekly SMS summary: tasks completed + upcoming checks.",
                "Invite to a workshop on ‘supporting study at home’. "],
      "owner": "Class teacher + parent/guardian",
      "when": "4 weeks",
      "kpis": ["Parent acknowledges weekly SMS", "Home routine ≥5 days/week"],
      "priority": "medium"
    })

  # Prioritize and de-dupe
  prio_rank = {"high": 0, "medium": 1, "low": 2}
  best = {}
  for it in plan:
    t = it["title"]
    if (t not in best) or (prio_rank[it["priority"]] < prio_rank[best[t]["priority"]]):
      best[t] = it
  plan = list(best.values())
  plan.sort(key=lambda x: prio_rank.get(x["priority"], 3))
  return plan

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

  table_data = [
    ["Student Name", student.get("student_name","—")],
    ["Class/Grade", student.get("class_name","—")],
    ["Gender", student.get("gender","—")],
    ["Total Score (out of 400)", f"{student.get('total_score',0):.1f}"],
    ["Risk Probability", f"{proba:.2%}"],
    ["Operating Threshold", f"{threshold:.2f}"],
    ["Risk Status", "AT RISK" if proba >= threshold else "LOW RISK"]
  ]
  tbl = Table(table_data, hAlign="LEFT", colWidths=[170, 340])
  tbl.setStyle(TableStyle([
    ("BACKGROUND",(0,0),(1,0), colors.HexColor("#fee2e2")),
    ("BOX",(0,0),(-1,-1), 0.5, colors.HexColor("#fecaca")),
    ("INNERGRID",(0,0),(-1,-1), 0.5, colors.HexColor("#fecaca")),
    ("ROWBACKGROUNDS",(0,1),(-1,-1), [colors.white, colors.HexColor("#f8fafc")])
  ]))
  elements += [tbl, Spacer(1, 12)]

  elements.append(Paragraph("<b>Recommended Actions (next 2–4 weeks)</b>", normal))
  for t in tips:
    elements.append(Paragraph(f"• {t}", normal))
  elements.append(Spacer(1, 10))
  elements.append(Paragraph(
    "This report supports teachers and parents to plan timely interventions. Use with classroom observations and school policy.", small))
  doc.build(elements)
  return buf.getvalue()

# =========================
# Batch sanitation helper
# =========================
def _sanitize_batch(df: pd.DataFrame) -> pd.DataFrame:
  """Trim whitespace/tabs, normalize flags, and coerce numerics for batch CSVs."""
  df = df.copy()

  # Trim spaces/tabs/newlines in every string column
  for col in df.columns:
    if df[col].dtype == object:
      df[col] = df[col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()

  # Normalize lunch / test_prep to 0/1
  def _to01(x):
    s = str(x).strip().lower()
    if s.startswith("1") or s in {"yes","y","true","t","completed","standard"}:
      return 1
    if s.startswith("0") or s in {"no","n","false","f","none","reduced/free","reduced"}:
      return 0
    try:
      return int(float(s))
    except:
      return np.nan

  if "lunch" in df.columns:
    df["lunch"] = df["lunch"].apply(_to01)
  if "test_preparation_course" in df.columns:
    df["test_preparation_course"] = df["test_preparation_course"].apply(_to01)

  # Coerce all score fields (and total) to numeric
  for c in ["math_score","reading_score","writing_score","science_score","total_score"]:
    if c in df.columns:
      df[c] = (
        df[c].astype(str)
             .str.replace(r"[^\d\.\-]", "", regex=True)
             .replace({"": np.nan})
      )
      df[c] = pd.to_numeric(df[c], errors="coerce")

  return df

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
# Tabs (Model Card removed)
# =========================
tab1, tab2 = st.tabs(["🔎 Single Prediction", "📦 Batch (CSV)"])

# ---------- Single ----------
with tab1:
  st.markdown("#### Quick Check")

  c1, c2 = st.columns(2)
  with c1: student_name = st.text_input("Student Name (optional)", "")
  with c2: math = st.number_input("Math Score", min_value=0.0, max_value=100.0, value=65.0, step=1.0)

  c1, c2 = st.columns(2)
  with c1: class_name = st.text_input("Class / Grade (optional)", "")
  with c2: read = st.number_input("Reading Score", min_value=0.0, max_value=100.0, value=70.0, step=1.0)

  c1, c2 = st.columns(2)
  with c1:
    parental = st.selectbox(
      "Parental Level of Education",
      ["high school","some college","associate's degree","bachelor's degree","master's degree"]
    )
  with c2:
    write = st.number_input("Writing Score", min_value=0.0, max_value=100.0, value=68.0, step=1.0)

  c1, c2 = st.columns(2)
  with c1:
    gender = st.radio("Gender", ["Female", "Male"], index=1, horizontal=True)
  with c2:
    sci = st.number_input("Science Score", min_value=0.0, max_value=100.0, value=66.0, step=1.0)

  c1, c2 = st.columns(2)
  with c1:
    lunch_choice = st.radio("Lunch (1=standard, 0=reduced/free)",
                            ["1 (standard)", "0 (reduced/free)"], index=1, horizontal=True)
  with c2:
    prep_choice = st.radio("Test Preparation Course (1=completed, 0=none)",
                           ["0 (none)", "1 (completed)"], index=0, horizontal=True)

  lunch = 1 if lunch_choice.startswith("1") else 0
  prep = 1 if prep_choice.startswith("1") else 0

  total = math + read + write + sci
  st.markdown(f"**Total Score (auto):** {total:.1f}")
  go = st.button("🔮 Predict", use_container_width=True)

  if go:
    row = {
      "student_name": student_name,
      "class_name": class_name,
      "gender": gender,
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
      try:
        cols_sent = list(_align_to_expected(_encode_inputs(pd.DataFrame([row]), meta), EXPECTED).columns)
      except Exception:
        cols_sent = list(X.columns)
      st.error(f"Prediction failed.\n\nExpected: {EXPECTED}\nSent to model: {cols_sent}\n\nDetails: {e}")
      st.stop()

    at_risk = proba >= THRESHOLD

    # KPI row
    k1, k2, k3 = st.columns(3)
    with k1:
      st.markdown(f"""
        <div class='kpi'>
          <b>Risk Probability</b>
          <div class='value'>{proba:.2%}</div>
        </div>
      """, unsafe_allow_html=True)
    with k2:
      st.markdown(f"""
        <div class='kpi'>
          <b>Threshold</b>
          <div class='value'>{THRESHOLD:.2f}</div>
        </div>
      """, unsafe_allow_html=True)
    with k3:
      status_color = "var(--bad)" if at_risk else "var(--good)"
      status_text = "AT RISK" if at_risk else "LOW RISK"
      st.markdown(f"""
        <div class='kpi'>
          <b>Status</b>
          <div class='pill' style="background:{status_color}; color:#fff">{status_text}</div>
          <div class='value' style="font-size:1.4rem;opacity:.0;">&nbsp;</div>
        </div>
      """, unsafe_allow_html=True)

    # Recommended Interventions (dynamic, rule-based; no LLM)
    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
    st.markdown("<h4 style='margin:0 0 10px 0;'>Recommended Interventions</h4>", unsafe_allow_html=True)

    plan = interventions_dynamic({**row, "total_score": float(X.loc[0, "total_score"])}, proba, THRESHOLD)

    if not plan:
      st.info("No specific gaps detected. Suggest general study skills and weekly monitoring.")
    else:
      for i, it in enumerate(plan, 1):
        badge_color = {"high":"#ef4444", "medium":"#f59e0b", "low":"#10b981"}.get(it["priority"], "#9ca3af")
        badge_text  = it["priority"].upper()
        steps_html = "".join([f"<li>{s}</li>" for s in it["steps"]])
        kpis_html  = "".join([f"<li>{k}</li>" for k in it["kpis"]])

        st.markdown(f"""
        <div class='card' style='margin-bottom:10px'>
          <div style="display:flex;align-items:center;justify-content:space-between;">
            <h4 style="margin:0">{i}. {it['title']}</h4>
            <span class='badge' style="background:{badge_color};color:#fff;border:1px solid rgba(0,0,0,.2)">{badge_text}</span>
          </div>
          <p class='small' style="margin:.25rem 0 .6rem 0">{it['why']}</p>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">
            <div>
              <b>Steps (next {it['when']}):</b>
              <ul style="margin:.4rem 0 0 .9rem">{steps_html}</ul>
            </div>
            <div>
              <b>Success checks (KPIs):</b>
              <ul style="margin:.4rem 0 0 .9rem">{kpis_html}</ul>
            </div>
          </div>
          <p class='small' style="margin-top:.6rem"><b>Owner:</b> {it['owner']} &nbsp;•&nbsp; <b>Timeline:</b> {it['when']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    c1, c2 = st.columns(2)

    # CSV (single)
    with c1:
      first_intervention = plan[0]["title"] if plan else ""
      single_out = pd.DataFrame([{
        "date": datetime.now().strftime("%Y-%m-%d"),
        "student_name": student_name,
        "class_name": class_name,
        "gender": gender,
        "math_score": math,
        "reading_score": read,
        "writing_score": write,
        "science_score": sci,
        "total_score": float(X.loc[0,"total_score"]),
        "risk_probability": round(proba,4),
        "at_risk": int(at_risk),
        "first_intervention": first_intervention
      }])
      csv_buf = single_out.to_csv(index=False).encode("utf-8")
      st.download_button("⬇️ Download CSV (single)", data=csv_buf, file_name="student_risk_single.csv",
                         mime="text/csv", use_container_width=True)

    # PDF (single)
    with c2:
      tips_for_pdf = [f"{i+1}. {it['title']} — {it['why']}" for i, it in enumerate(plan)] if plan else []
      pdf_bytes = build_parent_letter_pdf(
        {"student_name": student_name,
         "class_name": class_name,
         "gender": gender,
         "total_score": float(X.loc[0,"total_score"])},
        proba, THRESHOLD, tips_for_pdf
      )
      st.download_button("⬇️ Download Parent Letter (PDF)", data=pdf_bytes,
                         file_name=f"{(student_name or 'student').replace(' ','_')}_parent_letter.pdf",
                         mime="application/pdf", use_container_width=True)

# ---------- Batch ----------
with tab2:
  st.markdown("#### Batch Upload")
  st.caption("Expected columns (case-insensitive; extra columns ignored): "
             "`student_name (optional), class_name (optional), gender (optional), parental_level_of_education, lunch, test_preparation_course, math_score, reading_score, writing_score, science_score, total_score`")

  file = st.file_uploader("Upload CSV", type=["csv"])
  if file is not None:
    # Read & sanitize immediately to fix things like '\t41'
    df = pd.read_csv(file)
    df = _sanitize_batch(df)

    needed = ["student_name","class_name","gender","parental_level_of_education","lunch","test_preparation_course",
              "math_score","reading_score","writing_score","science_score","total_score"]
    need_lower = {c.lower() for c in needed}
    present = [c for c in df.columns if c.lower() in need_lower]
    Xraw = df[present].copy()

    # Always recompute total safely
    Xraw["total_score"] = Xraw.apply(lambda r: _recompute_total(r), axis=1)

    # Prepare & predict
    try:
      X_model = _prepare_for_model(Xraw, EXPECTED)   # adds gender_encoded + parental_education_encoded
      probs = _safe_predict_proba(X_model)
    except Exception as e:
      try:
        cols_sent = list(_align_to_expected(_encode_inputs(Xraw, meta), EXPECTED).columns)
      except Exception:
        cols_sent = list(Xraw.columns)
      st.error(
        "Batch prediction failed.\n\n"
        f"Expected: {EXPECTED}\n"
        f"Sent to model: {cols_sent}\n\n"
        f"Details: {e}"
      )
      st.stop()

    preds = (probs >= THRESHOLD).astype(int)
    out = df.copy()
    out["risk_probability"] = np.round(probs, 4)
    out["at_risk"] = preds

    def _first_tip(i: int):
      row_i = {
        "parental_level_of_education": out.loc[i, "parental_level_of_education"] if "parental_level_of_education" in out.columns else "",
        "lunch": out.loc[i, "lunch"] if "lunch" in out.columns else 1,
        "test_preparation_course": out.loc[i, "test_preparation_course"] if "test_preparation_course" in out.columns else 0,
        "math_score": out.loc[i, "math_score"] if "math_score" in out.columns else None,
        "reading_score": out.loc[i, "reading_score"] if "reading_score" in out.columns else None,
        "writing_score": out.loc[i, "writing_score"] if "writing_score" in out.columns else None,
        "science_score": out.loc[i, "science_score"] if "science_score" in out.columns else None,
        "total_score": out.loc[i, "total_score"] if "total_score" in out.columns else None,
      }
      p = float(out.loc[i, "risk_probability"])
      pl = interventions_dynamic(row_i, p, THRESHOLD)
      return pl[0]["title"] if pl else ""
    out["first_intervention"] = [_first_tip(i) for i in range(len(out))]

    st.dataframe(out.head(20), use_container_width=True)

    # CSV Export
    buf = BytesIO()
    out.to_csv(buf, index=False)
    st.download_button("⬇️ Download Results CSV", data=buf.getvalue(),
                       file_name="batch_predictions.csv", mime="text/csv", use_container_width=True)

    # ZIP of PDFs
    make_zip = st.checkbox("Generate a ZIP of parent letters (PDF) for all rows", value=False)
    if make_zip:
      zip_buf = BytesIO()
      with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for i in range(len(out)):
          sname = str(out.loc[i, "student_name"]) if "student_name" in out.columns else f"student_{i+1}"
          cname = str(out.loc[i, "class_name"]) if "class_name" in out.columns else ""
          gndr  = str(out.loc[i, "gender"]) if "gender" in out.columns else "—"
          total_i = float(out.loc[i, "total_score"]) if "total_score" in out.columns else float(
            (out.loc[i, "math_score"] if "math_score" in out.columns else 0) +
            (out.loc[i, "reading_score"] if "reading_score" in out.columns else 0) +
            (out.loc[i, "writing_score"] if "writing_score" in out.columns else 0) +
            (out.loc[i, "science_score"] if "science_score" in out.columns else 0)
          )
          row_i = {
            "parental_level_of_education": out.loc[i, "parental_level_of_education"] if "parental_level_of_education" in out.columns else "",
            "lunch": out.loc[i, "lunch"] if "lunch" in out.columns else 1,
            "test_preparation_course": out.loc[i, "test_preparation_course"] if "test_preparation_course" in out.columns else 0,
            "math_score": out.loc[i, "math_score"] if "math_score" in out.columns else None,
            "reading_score": out.loc[i, "reading_score"] if "reading_score" in out.columns else None,
            "writing_score": out.loc[i, "writing_score"] if "writing_score" in out.columns else None,
            "science_score": out.loc[i, "science_score"] if "science_score" in out.columns else None,
            "total_score": total_i
          }
          plan_i = interventions_dynamic(row_i, float(out.loc[i, "risk_probability"]), THRESHOLD)
          tips_i = [f"{j+1}. {it['title']} — {it['why']}" for j, it in enumerate(plan_i)] if plan_i else []

          pdf = build_parent_letter_pdf(
            {"student_name": sname, "class_name": cname, "gender": gndr, "total_score": total_i},
            float(out.loc[i, "risk_probability"]), THRESHOLD, tips_i
          )
          safe_name = (sname or f"student_{i+1}").replace(" ", "_")
          zf.writestr(f"{safe_name}_parent_letter.pdf", pdf)
      st.download_button("⬇️ Download ZIP of Parent Letters (PDFs)",
                         data=zip_buf.getvalue(), file_name="parent_letters.zip",
                         mime="application/zip", use_container_width=True)
