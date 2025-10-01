# 🎓 Student Risk — Early-Warning & Intervention (Streamlit)

A professional, teacher-friendly app to **identify at-risk students early**, explain *why*, and suggest **actionable interventions**. Supports **single-student** checks and **batch CSV** predictions, with **exportable PDF parent letters** and **results CSV**.

> Built with **Python + scikit-learn + Streamlit**. Trained using `Student_performance_prediction.ipynb`.  
> Repo structure and instructions below reflect the files in your `student-risk-system`.

---

## ✨ Features

- **At-Risk Detection** (binary): predicts risk probability and flags `AT RISK` vs `LOW RISK`
- **Actionable Interventions**: auto-generated, teacher-oriented next steps
- **Batch Predictions**: upload a CSV; download:
  - `batch_predictions.csv` (probabilities + flags + first intervention)
  - ZIP of **PDF parent letters** (optional)
- **Clean UI**: modern Streamlit app with KPIs, badges, and export buttons
- **Model-agnostic**: loads any scikit-learn model `.pkl` (expects saved `feature_names_`)

---

## 🗂️ Project Structure

student-risk-system/

├─ app.py              # Streamlit app (run this)

├─ requirements.txt    # Minimal deps

├─ model/

│ └─ logistic_regression_model.pkl          # Reference model

├─ sample_data/

│ └─ student_sample.csv                       # Demo CSV for batch testing

├─ cleaned_student_data.csv                   # Preprocessed Data used during training model

├─ Student_performance.csv                    # Raw dataset (optional)

├─ Student_performance_prediction.ipynb       # Training/evaluation notebook

└─ other_models/                             # model of Gradiant Boosting, Randome forest which used for compare and select best model


---

## 🔧 Setup (Local)

```bash
# 1) Create & activate a virtual environment (Windows PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the app
streamlit run app.py

```
---

## To point to a different file, set the env var:
```bash
$env:MODEL_PATH="model/gradient_boosting_model.pkl"   # PowerShell
# or
export MODEL_PATH="model/gradient_boosting_model.pkl"  # macOS/Linux
streamlit run app.py
```

---

## **Using the App**
1) Single-Student Check
- Enter scores and context in the left panel.
- See Risk Probability, Threshold (from the model), and status badge.
- Review Recommended Interventions.
- Export a PDF letter for parents/guardians if needed.

2) Batch Predictions
- Prepare a CSV (see schema below).
- Drag & drop in the “Batch Predict” section.
- Preview results table, download:
- batch_predictions.csv**
- ZIP of PDFs (parent letters) — optional toggle

**📄 CSV Schema (Batch)**

Header row (case-insensitive) — the app will align columns and ignore extras.

**Required/Useful columns:**
- student_name (optional, used for PDF name; default student_#)
- class_name (optional)
- parental_level_of_education
- lunch (“standard” vs “free/reduced” style; numeric/encoded also allowed)
- test_preparation_course (“completed” / “none”; numeric/encoded allowed)
- math_score
- reading_score
- writing_score
- science_score
- total_score

---

## **🧠 Model Training (Notebook)**

Use Student_performance_prediction.ipynb to:

- Clean & preprocess the dataset

- Train models (Logistic Regression / Random Forest / Gradient Boosting)

- Evaluate with cross-validation

- Save a deployable model that includes:

   - feature_names_ (the exact input feature order)

   - meta_ dict (recommended) with:

      - threshold (operating cutoff for “at risk”)

      - num_features, cat_features

- encoders info if needed

---

## **🙌 Acknowledgements**

scikit-learn, pandas, numpy, Streamlit

Teachers in Sri Lanka carrying significant non-teaching workloads — this tool aims to help reduce that burden.

---

## **📢 Contribution & Feedback**

Feel free to make suggestions and try out the system code.

 - Clone the repo, run locally, and explore the app.

 - Open an Issue or Pull Request if you find bugs or want to suggest improvements.

 - Teachers and students are especially encouraged to test the CSV batch mode and share feedback.

---

## ** Deployment **
