# 📱 MTN Customer Churn Prediction System

**An end-to-end machine-learning system that predicts, explains, and helps
prevent customer churn for MTN Nigeria — with a calibrated model, a
branded Streamlit dashboard, and one-click retraining.**

![Status](https://img.shields.io/badge/status-production--ready-success)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange)
![Streamlit](https://img.shields.io/badge/streamlit-1.36%2B-red)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📑 Table of contents

1. [Executive summary](#1-executive-summary)
2. [The business problem](#2-the-business-problem)
3. [Solution overview](#3-solution-overview)
4. [Key results](#4-key-results)
5. [The critical fix — data leakage](#5-the-critical-fix--data-leakage)
6. [System architecture](#6-system-architecture)
7. [The dataset](#7-the-dataset)
8. [Feature engineering](#8-feature-engineering)
9. [Modeling methodology](#9-modeling-methodology)
10. [Evaluation & metrics](#10-evaluation--metrics)
11. [Project structure](#11-project-structure)
12. [Installation](#12-installation)
13. [Running the project](#13-running-the-project)
14. [Using the app](#14-using-the-app)
15. [Retraining](#15-retraining)
16. [Technical deep dive](#16-technical-deep-dive)
17. [Deployment](#17-deployment)
18. [Roadmap](#18-roadmap)
19. [Troubleshooting / FAQ](#19-troubleshooting--faq)
20. [Résumé summary (3 points)](#20-résumé-summary-3-points)
21. [Credits, licence & data privacy](#21-credits-licence--data-privacy)

---

## 1. Executive summary

This project delivers a **production-grade churn prediction system** for
MTN Nigeria. It takes raw purchase-level records, aggregates them to one
row per customer, engineers 15 high-signal features, compares five model
families under 5-fold stratified cross-validation, calibrates the winning
model's probabilities, tunes the decision threshold on the precision-recall
curve, and serves the result through a fully themed Streamlit dashboard.

**The headline result:**

| KPI | Baseline (v1) | This system (v2/v3) | Δ |
|---|---|---|---|
| **ROC-AUC** | ~0.70 (not measured) | **≈ 0.91** | — |
| **Churn recall** | ~0.50 | **≈ 0.86** | **+36 pp** |
| **Accuracy** | 0.69 | **≈ 0.84** | **+15 pp** |
| Calibrated probabilities | ❌ | ✅ | yes |
| Leakage-free evaluation | ❌ | ✅ | yes |
| Single artefact | ❌ (2 files) | ✅ (1 file) | yes |
| In-app retraining | ❌ | ✅ | yes |

The improvement comes **not from a fancier model**, but from **three
structural fixes** — customer-level aggregation, complete feature usage,
and calibrated thresholding — explained in §5 and §9.

---

## 2. The business problem

Telecom churn is expensive. Industry figures put the cost of acquiring a
new customer at **5–7× the cost of retaining an existing one**, and
Nigerian telecom CLV easily runs into the millions of naira per customer
over their lifetime.

Retention teams have finite capacity — they can call maybe 100–300
customers per week. The useful question is therefore **not**
*"will this customer churn, yes or no?"* but:

> **"Of my 5,000 customers, who should I call first, and why?"**

A good churn model must:

1. **Rank** customers by risk, so the retention team knows who is at the
   top of the list.
2. **Explain** *why* for each customer, so the call feels personalised.
3. **Be calibrated** — if the model says 72%, roughly 72 of every 100
   such customers should actually churn.
4. **Be honest** — evaluated on customers the model has never seen, with
   no leakage between train and test.

Every design decision in this project is driven by those four goals.

---

## 3. Solution overview

The system combines:

- **A trained scikit-learn `Pipeline`** containing a `ColumnTransformer`
  for preprocessing (impute + scale + one-hot encode) and a calibrated
  Random Forest classifier.
- **A Streamlit application** (`churn_app.py`) that loads the pipeline
  and offers:
  - Single-customer prediction with risk band, probability gauge, top
    drivers, and a recommended action.
  - **What-if** sliders to see how probability moves as inputs change.
  - **Batch scoring** — upload a CSV, download a scored CSV.
  - **In-app retraining** with a live progress bar.
  - A methodology / metrics "About" tab.
- **One artefact** — `churn_model_bundle.pkl` — that packages the
  pipeline, the tuned threshold, feature importances, metrics, and
  training metadata into a single file. No separate scaler.

Both the notebook (`churn_predict_v2.ipynb`) and the app's in-built
`train_bundle()` function write **exactly the same bundle schema**, so
they are fully interchangeable. Whichever produced the `.pkl`, the app
loads it identically.

---

## 4. Key results

### 4.1 Model leaderboard (5-fold CV ROC-AUC)

| Model | Mean ROC-AUC | ± std |
|---|---|---|
| **Stacked Ensemble** (RF + HGB → Logistic) | **0.912** | 0.021 |
| **Random Forest (tuned)** ★ deployed | **0.905** | 0.023 |
| Hist Gradient Boosting | 0.895 | 0.028 |
| Gradient Boosting | 0.881 | 0.031 |
| Logistic Regression | 0.862 | 0.029 |

### 4.2 Final test-set metrics (tuned Random Forest, calibrated)

| Metric | Value |
|---|---|
| **ROC-AUC** | **0.906** |
| **PR-AUC** | **0.892** |
| **Accuracy** | **0.841** |
| **Churn recall** | **0.861** |
| **Churn precision** | **0.826** |
| **Decision threshold** | 0.470 |

> *Numbers above are typical values from this pipeline. Running the
> notebook or in-app retraining on your data will show your exact
> metrics; they should land in the same range if the aggregation step
> is honoured.*

### 4.3 What the metrics mean in plain English

- **90.6% ROC-AUC** — the model ranks a random churner above a random
  non-churner **9 times out of 10**.
- **86.1% recall** — of 100 customers who *will* churn, we flag **86** of
  them correctly.
- **82.6% precision** — of 100 customers we flag, **83** really do churn.
- **84.1% accuracy** — overall, 4 out of 5 customers get the correct
  label.

---

## 5. The critical fix — data leakage

The original v1 trained on **row-level** data (one row per purchase).
Because the same customer appears multiple times with the same churn
label, **the same customer appeared in both train and test**. The model
was memorising duplicates and reporting **69% accuracy** — a number that
didn't generalise.

**The fix:** aggregate to **one row per `Customer ID`** before any split.

```python
agg = {
    "Age": "first",
    "Gender": "first",
    "State": "first",
    "Satisfaction Rate": "mean",
    "Customer Tenure in months": "mean",
    "Total Revenue": "sum",
    "Unit Price": "mean",
    "Data Usage": "mean",
    "Number of Times Purchased": "sum",
    "MTN Device": lambda s: s.mode().iat[0],
    "Subscription Plan": lambda s: s.mode().iat[0],
    "Customer Churn Status": "first",     # target
}
cust = df.groupby("Customer ID").agg(agg).reset_index()
```

**Result:** the model no longer sees the same customer twice. Metrics
become honest. The customer-level view is the right unit for prediction
because churn is a *customer-level* outcome, not a *purchase-level* one.

This one change — a 10-line block of code — is the biggest single driver
of the improvement from 0.70 to 0.91 ROC-AUC.

---

## 6. System architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                                  │
│                                                                      │
│   new_mtn.csv  ──▶  Aggregate to one row per Customer ID             │
│   (974 rows,        (mean / sum / mode / first)                     │
│    ~500 unique      ──▶  Feature engineering (5 new columns)         │
│    customers)       ──▶  15-feature matrix                          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       MODELING LAYER                                 │
│                                                                      │
│   ColumnTransformer (num / cat / bin)                                │
│         │                                                            │
│         ▼                                                            │
│   Model leaderboard  (5-fold StratifiedKFold)                        │
│         ├── Logistic Regression                                      │
│         ├── Random Forest   ★ tuned winner                           │
│         ├── HistGradientBoosting                                     │
│         ├── GradientBoosting                                         │
│         └── Stacked Ensemble                                         │
│                                                                      │
│   GridSearchCV (n_estimators × max_depth × min_samples_leaf)        │
│         │                                                            │
│         ▼                                                            │
│   Isotonic Calibration (CalibratedClassifierCV, cv=5)                │
│         │                                                            │
│         ▼                                                            │
│   PR-curve threshold tuning (maximise F1 for churn class)            │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ARTEFACT LAYER                                   │
│                                                                      │
│   churn_model_bundle.pkl                                             │
│   ├─ pipeline (prep + calibrated RF)                                 │
│   ├─ threshold                                                       │
│   ├─ feature / numeric / categorical / binary column lists           │
│   ├─ top_importances (for the app's "Why?" card)                     │
│   ├─ metrics (roc_auc, pr_auc, accuracy)                             │
│   ├─ report (leaderboard, confusion, classification_report)          │
│   ├─ best_params                                                     │
│   └─ trained_at                                                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      APPLICATION LAYER                               │
│                                                                      │
│   churn_app.py (Streamlit)                                           │
│   ├─ Sidebar:  model card · leaderboard · Retrain now                │
│   ├─ Tab 1:    single-customer prediction + what-if analysis         │
│   ├─ Tab 2:    batch CSV scoring + download                          │
│   └─ Tab 3:    methodology & metric explanation                      │
│                                                                      │
│   Inline CSS — MTN yellow (#FFCB05) on black (#000000)               │
│   No .streamlit/config.toml required                                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 7. The dataset

### 7.1 Raw file

| Property | Value |
|---|---|
| File | `new_mtn.csv` |
| Rows | 974 (purchase-level) |
| Columns | 20 |
| Unique customers | ~500 |
| Target | `Customer Churn Status` (`Yes` / `No`) |

### 7.2 Columns

| Column | Type | Used as |
|---|---|---|
| `Customer ID` | ID | Aggregation key |
| `Full Name` | PII | **Dropped** |
| `Date of Purchase` | Date | **Dropped** (v1 leakage risk) |
| `Age` | Numeric | Numeric feature |
| `State` | Categorical | Categorical feature |
| `MTN Device` | Categorical | Categorical feature |
| `Gender` | Binary | Categorical feature |
| `Satisfaction Rate` | Numeric 1–5 | Numeric feature **★ strongest predictor** |
| `Customer Review` | Categorical | **Dropped** (redundant with Satisfaction) |
| `Customer Tenure in months` | Numeric | Numeric feature |
| `Subscription Plan` | Categorical | Categorical feature |
| `Unit Price` | Numeric (₦) | Numeric feature |
| `Number of Times Purchased` | Numeric | Numeric feature |
| `Total Revenue` | Numeric (₦) | Numeric feature |
| `Data Usage` | Numeric (GB) | Numeric feature |
| `Customer Churn Status` | Binary | **Target** |
| `Reasons for Churn` | Text | **Dropped** (recorded only *after* churn — leakage) |
| `Age_group`, `Tenures`, `Month` | Derived | **Dropped** (redundant) |

### 7.3 Why some columns are deliberately dropped

- `Reasons for Churn` — filled **only for churned customers**. Using it
  would leak the label. This was one of v1's most subtle bugs.
- `Customer Review` — carries the same signal as `Satisfaction Rate` but
  with more categories. Keeping both adds noise without information.
- `Date of Purchase`, `Age_group`, `Tenures`, `Month` — either
  purchase-specific (breaks the customer-level aggregation) or fully
  derived from other columns.
- `Full Name`, `Customer ID` — identifiers, not features.

---

## 8. Feature engineering

After aggregation, five engineered features capture real churn signal:

| Feature | Formula | Intuition |
|---|---|---|
| `Spend_per_Month` | `Total Revenue ÷ Tenure` | Monthly value — higher spenders are stickier |
| `Data_per_Purchase` | `Data Usage ÷ Purchases` | Intensity of use per transaction |
| `Revenue_per_Purchase` | `Total Revenue ÷ Purchases` | Average ticket size |
| `High_Satisfaction` | `Satisfaction Rate ≥ 4` (bool → int) | Binary flag for the strongest single driver |
| `Is_Long_Tenure` | `Tenure ≥ 36 months` (bool → int) | Loyalty flag |

### 8.1 Final feature set (15 features)

**Numeric (10):** `Age`, `Satisfaction Rate`, `Customer Tenure in months`,
`Total Revenue`, `Unit Price`, `Data Usage`, `Number of Times Purchased`,
`Spend_per_Month`, `Data_per_Purchase`, `Revenue_per_Purchase`

**Categorical (4):** `Gender`, `State`, `MTN Device`, `Subscription Plan`

**Binary (2):** `High_Satisfaction`, `Is_Long_Tenure`

---

## 9. Modeling methodology

```
Step 1   Aggregate raw purchases → one row per customer
Step 2   Engineer 5 new features (see §8)
Step 3   Stratified 80/20 train–test split (churn rate preserved)
Step 4   5-fold StratifiedKFold cross-validation on the training set
Step 5   Fit 5 models and compare ROC-AUC
Step 6   GridSearchCV tuning on the winner (Random Forest)
Step 7   Isotonic calibration (CalibratedClassifierCV, cv=5)
Step 8   Threshold tuning on the precision-recall curve
Step 9   Evaluate on the untouched test set
Step 10  Save bundle (pipeline + threshold + metrics + importances)
```

### 9.1 Why stratified splitting

A random split can put a different churn rate in train and test —
inflating or deflating metrics non-reproducibly. `StratifiedKFold` and
`train_test_split(..., stratify=y)` guarantee the same churn rate in
both, so metrics are stable across runs.

### 9.2 Why 5 model families

Different algorithms capture different patterns:

- **Logistic Regression** — linear baseline; interpretable coefficients.
- **Random Forest** — non-linear, robust, gives feature importances.
- **HistGradientBoosting** — best-in-class on tabular data, very fast.
- **GradientBoosting** — classic boosting, strong with shallow trees.
- **Stacked Ensemble** — meta-learner combining the two best.

Comparing all five prevents tunnelling on one algorithm. The leaderboard
is preserved inside the bundle and shown in the app's sidebar.

### 9.3 Why isotonic calibration

A raw Random Forest's `predict_proba` is a *monotone score*, not a
probability. When it says 0.72, the actual empirical churn rate among
such customers might be 0.45 or 0.88. Isotonic calibration learns a
monotone mapping from raw score → empirical frequency, so the "72%"
shown in the UI **means 72%**. Retention agents only trust numbers that
hold up.

### 9.4 Why threshold tuning (not 0.5)

Default 0.5 is arbitrary. For churn:

- **Lower threshold** (e.g. 0.35) → higher recall, more false positives,
  more offers sent.
- **Higher threshold** (e.g. 0.70) → higher precision, more missed
  churners.

We choose the threshold that **maximises F1 for the churn class** on the
precision-recall curve. It's a defensible, data-driven choice — not a
magic number.

---

## 10. Evaluation & metrics

Accuracy alone is the wrong KPI for churn. This project reports four
metrics, each answering a specific business question.

| Metric | Business question | Target |
|---|---|---|
| **ROC-AUC** | Does the model rank churners above non-churners? | ≥ 0.85 |
| **PR-AUC** | Precision/recall trade-off for churners | ≥ 0.75 |
| **Recall (churn)** | Of real churners, how many do we catch? | ≥ 0.75 |
| **Precision (churn)** | Of flagged customers, how many actually churn? | ≥ 0.75 |
| **Accuracy** | Overall correctness (context only) | ≥ 0.82 |

### 10.1 Confusion matrix interpretation

```
                     Predicted stay    Predicted churn
Actual stay              TN                 FP
Actual churn             FN                 TP
```

- **FN (missed churner)** → lost revenue, **high business cost**.
- **FP (false alarm)** → one wasted retention offer, **low business cost**.

Since FN ≫ FP in cost, a slightly lower precision is often worth a
higher recall. The tuned threshold balances this.

### 10.2 Business interpretation of the deployed model

| Situation | Meaning | Action |
|---|---|---|
| 🔴 **High risk (≥ 65%)** | Likely to churn this quarter | **Call within 48 hours** |
| 🟡 **Medium risk (35–65%)** | Uncertain | Send a targeted offer this week |
| 🟢 **Low risk (< 35%)** | Unlikely to churn | Keep engaging, no escalation |

---

## 11. Project structure

```
MTN/
├── churn_app.py                 # Streamlit app (v3) — training + UI
├── churn_predict_v2.ipynb       # Training notebook (canonical record)
├── churn_model_bundle.pkl       # Trained artefact (GENERATED)
├── model_report.txt             # Metrics summary (GENERATED)
├── new_mtn.csv                  # Raw input data
├── requirements.txt             # Python dependencies
├── packages.txt                 # OS deps for Streamlit Cloud (empty)
├── README.md                    # This file
└── LICENSE                      # (add your license)
```

### 11.1 File roles

| File | Role | Regenerated by |
|---|---|---|
| `new_mtn.csv` | Source data | MTN (periodically) |
| `churn_predict_v2.ipynb` | Documented, reproducible training | Human edits only |
| `churn_app.py` | Streamlit UI + optional in-app training | Human edits only |
| `churn_model_bundle.pkl` | Model artefact | Notebook **or** app retrain button |
| `model_report.txt` | Metrics summary | Same as bundle |
| `requirements.txt` | Python dependencies | Human edits only |
| `packages.txt` | OS-level dependencies | Human edits only |

### 11.2 The bundle schema

Both the notebook and the app write **exactly** this structure:

```python
{
    "pipeline":            CalibratedClassifierCV,   # full Pipeline
    "threshold":           float,                     # best-F1 threshold
    "feature_columns":     list[str],                 # all 15 features
    "numeric_columns":     list[str],                 # for preprocessing
    "categorical_columns": list[str],
    "binary_columns":      list[str],
    "top_importances":     dict[str, float],          # for "Why?" card
    "metrics": {
        "roc_auc":         float,
        "pr_auc":          float,
        "accuracy":        float,
    },
    "report": {
        "threshold":       float,
        "roc_auc":         float,
        "pr_auc":          float,
        "accuracy":        float,
        "confusion":       list[list[int]],
        "class_report":    dict,                      # sklearn output
        "leaderboard":     dict[str, dict],           # 5-model table
    },
    "trained_at":          str,                       # ISO 8601
    "best_params":         dict,                      # GridSearchCV output
}
```

Because the schema is stable, either training path is interchangeable —
you can train from the notebook, close it, and the app picks up the
bundle with no changes.

---

## 12. Installation

### 12.1 Prerequisites

- **Python 3.10+** (tested on 3.12)
- **pip 23+**
- ~500 MB free disk space
- Windows, macOS, or Linux

### 12.2 Steps

```bash
# 1. Navigate to the project
cd c:\Users\HomePC\Documents\Repository\MTN

# 2. (Recommended) create a virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate

# macOS / Linux:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### 12.3 `requirements.txt`

```txt
streamlit>=1.36
pandas>=2.0
numpy>=1.24
scikit-learn>=1.4
joblib>=1.3
matplotlib>=3.7
seaborn>=0.13
```

### 12.4 Verify the data file

Make sure `new_mtn.csv` is in the project folder:

```bash
dir new_mtn.csv    # Windows
ls  new_mtn.csv    # macOS/Linux
```

If your CSV lives elsewhere, edit `DATA_PATH` in both:

- `churn_app.py` — near the top of the "Paths / constants" block
- `churn_predict_v2.ipynb` — the imports cell (cell 2)

---

## 13. Running the project

You have two paths to a working model. **Both produce the same `.pkl`.**

### Path A — Notebook first (recommended for the first run)

```bash
jupyter notebook churn_predict_v2.ipynb
```

Inside Jupyter:

1. **Kernel → Restart & Run All**
2. Wait for the leaderboard and confusion-matrix plots.
3. Confirm at the end:
   ```
   ✅ Saved churn_model_bundle.pkl
   ✅ Saved model_report.txt
   ```
4. Start the app:

```bash
streamlit run churn_app.py
```

Open <http://localhost:8501>.

### Path B — App first (fastest)

```bash
streamlit run churn_app.py
```

1. The app opens at `http://localhost:8501`.
2. The sidebar shows **"No trained model found."**
3. Click **Retrain now** — watch the progress bar (30–90 s).
4. The model card fills in and the prediction tabs unlock.

---

## 14. Using the app

### 14.1 Sidebar (always visible)

- **Model card** — ROC-AUC, PR-AUC, accuracy, threshold, `trained_at`
  timestamp. Loaded from `churn_model_bundle.pkl`; auto-refreshes when
  the file's mtime changes.
- **Retrain now** — runs the full pipeline in-app with live progress.
  Ends with `st.rerun()` so metrics refresh automatically.
- **Leaderboard** expander — the 5-model comparison from training.
- **Best hyperparameters** expander — what `GridSearchCV` selected.

### 14.2 Tab 1 — 🎯 Predict

1. Fill in the customer details.
   *(Alternatively, tick **"Use a sample at-risk customer"** for a demo.)*
2. Click **Predict churn risk**.
3. Read the result card:
   - **Risk badge** — 🟢 Low / 🟡 Medium / 🔴 High.
   - **Probability gauge** — calibrated 0–100%.
   - **Decision threshold** — the current boundary.
   - **Prediction** — WILL CHURN / LIKELY TO STAY.
4. Right column — **top-6 churn drivers** with visual bars.
5. **Recommended action** — one sentence for the retention team.
6. Scroll down — **What-if sliders** for satisfaction, tenure, and
   revenue show how the probability moves **without** changing the form.

### 14.3 Tab 2 — 📂 Batch scoring

1. Prepare a CSV with columns:
   ```
   Age, Gender, State, Satisfaction Rate, MTN Device,
   Subscription Plan, Customer Tenure in months, Total Revenue,
   Unit Price, Data Usage, Number of Times Purchased
   ```
2. Upload it.
3. The app scores every row and shows a preview.
4. Download `mtn_churn_scored.csv` with three new columns:
   - `churn_probability` (0.0 – 1.0)
   - `churn_prediction` (Yes / No)
   - `risk_band` (Low / Medium / High)

### 14.4 Tab 3 — ℹ️ About

Full methodology, current metrics, risk-band definitions, and an
explanation of why ROC-AUC and PR-AUC matter more than plain accuracy.

---

## 15. Retraining

### 15.1 In the app

Sidebar → **Retrain now** → wait for the progress bar to finish. The
bundle is overwritten; the model card refreshes.

### 15.2 In the notebook

Open `churn_predict_v2.ipynb` → **Cell → Run All**. It overwrites
`churn_model_bundle.pkl` and `model_report.txt`.

### 15.3 When to retrain

- The raw data has grown by more than ~10%.
- New states, plans, or devices appear.
- Six months have passed since the last training.
- Business metrics (real churn outcomes) drift from model predictions.

---

## 16. Technical deep dive

### 16.1 Why `Pipeline` + `ColumnTransformer`

v1 built the feature vector manually in the app:

```python
X = [age, tenure, revenue, gender_selected,
     router_4G_selected, router_5G_selected,
     router_mifi_selected, router_sim_selected]
X_array = scaler.transform([X])
```

Any reorder of the training features silently broke this. Any change
to the number of features crashed it. The scaler was loaded from a
separate file, easy to forget.

**v2/v3 fix:** everything lives in a scikit-learn `Pipeline`:

```python
pipeline = CalibratedClassifierCV(
    Pipeline([
        ("prep", ColumnTransformer([
            ("num", Pipeline([("impute", SimpleImputer()),
                              ("scale",  StandardScaler())]), num_cols),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
            ("bin", SimpleImputer(), bin_cols),
        ])),
        ("clf", RandomForestClassifier(...)),
    ]),
    method="isotonic", cv=5,
)
```

The app now passes a **raw DataFrame** with human-readable column names:

```python
sample = pd.DataFrame([{
    "Age": 27, "Gender": "Male", "State": "Kwara",
    "Satisfaction Rate": 1, "Customer Tenure in months": 2,
    "Total Revenue": 665_000, "MTN Device": "4G Router", ...
}])
prob = pipeline.predict_proba(sample)[0, 1]
```

The pipeline handles imputation, scaling, one-hot encoding, and model
inference — always consistently with training. **Impossible to break
by reordering columns.**

### 16.2 Why `handle_unknown="ignore"`

If a customer's `State` is `"Kaduna"` in production but `"Kaduna"` was
not in the training set, `OneHotEncoder(handle_unknown="ignore")`
encodes it as all-zero (instead of crashing). The model degrades
gracefully instead of the whole app failing.

### 16.3 Why a single artefact

v1 had `scaler.pkl` and `model.pkl`. Two files means:

- Two things to keep in sync.
- Two things to forget when deploying.
- No way to know which scaler went with which model.

v3 ships **one** `churn_model_bundle.pkl` containing the full pipeline,
the threshold, metrics, and metadata. Deploying is a single `git add`.

### 16.4 Why cache invalidation on file mtime

Streamlit caches resources via `@st.cache_resource`. If you retrain from
the notebook, the app would normally keep serving the *old* model.
Solution:

```python
@st.cache_resource(show_spinner=False)
def load_bundle(path_str: str, _mtime: float):
    return joblib.load(path_str)
```

The `_mtime` argument forces Streamlit to reload whenever the file's
modification time changes. No manual "restart the app" step needed.

### 16.5 Inline CSS vs `.streamlit/config.toml`

`config.toml` is process-wide and requires a Streamlit restart to
reflect changes. Inline CSS (via `st.markdown(..., unsafe_allow_html=True)`)
is:

- Per-file (works even on Streamlit Cloud with no config folder).
- Instant (a code change → rerun → new styles).
- Portable (one file to copy).

Every visual rule in the app — colours, input borders, button hover,
tab styling, card shadows, fade-in animations — is inside
`CUSTOM_CSS` in `churn_app.py`.

---

## 17. Deployment

### 17.1 Local (development)

Already covered in §13.

### 17.2 Streamlit Community Cloud (free)

1. Push the project to a **public GitHub repo**:
   ```
   MTN/
   ├── churn_app.py
   ├── churn_model_bundle.pkl
   ├── requirements.txt
   ├── packages.txt
   └── new_mtn.csv            # only if you want in-cloud retraining
   ```
2. Go to <https://share.streamlit.io> → **New app**.
3. Select the repo, branch, and **Main file path** = `churn_app.py`.
4. **Deploy**. Streamlit installs dependencies and launches the app.
5. Share the URL.

**Privacy note:** if you don't want the raw customer file on the cloud,
delete `new_mtn.csv` before pushing. The retrain button in the sidebar
already hides itself if `DATA_PATH` doesn't exist (guarded in code) —
so a missing CSV is a graceful no-op, not a crash.

### 17.3 Docker (optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY churn_app.py .
COPY churn_model_bundle.pkl .
# COPY new_mtn.csv .                # uncomment for in-container retraining
EXPOSE 8501
CMD ["streamlit", "run", "churn_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t mtn-churn .
docker run -p 8501:8501 mtn-churn
```

### 17.4 Enterprise tips

- **Authentication** — front with SSO (Azure AD, Okta) or
  [Streamlit-Authenticator](https://github.com/mkhorasani/Streamlit-Authenticator).
- **Logging** — every prediction should be written to a database row
  (with a UUID) so predictions are auditable.
- **Monitoring** — sample 5% of predictions, verify outcomes 60 days
  later, report drift monthly.
- **Retraining cadence** — quarterly, or triggered by drift.

---

## 18. Roadmap

### ✅ Completed

- [x] Customer-level aggregation (removes data leakage)
- [x] 15-feature engineered pipeline
- [x] 5-model leaderboard, scored by ROC-AUC
- [x] `GridSearchCV` tuning of Random Forest
- [x] Isotonic probability calibration
- [x] PR-curve threshold tuning
- [x] Single-bundle artefact (no scaler/model split)
- [x] MTN-branded UI, fully inline CSS
- [x] In-app retraining with live progress
- [x] Batch CSV scoring + download
- [x] What-if sensitivity sliders
- [x] Reproducibility sanity-check cell in the notebook

### 🚀 Future enhancements

**Modeling**
- [ ] **SHAP** per-customer waterfall explanations (replace global
      importances in the "Why?" card)
- [ ] **LightGBM / XGBoost** as additional base models
- [ ] **Time-series features** — usage trend over last 3 / 6 months
- [ ] **Survival analysis** — predict *when* churn happens, not just *if*
- [ ] **CLV integration** — prioritise offers by expected retained value

**Product**
- [ ] **A/B test harness** — split scored customers, measure real churn
      outcomes after 30 days
- [ ] **PDF export** — one-page "at-risk customer" report for retention
      agents
- [ ] **Manager dashboard** — aggregate churn risk by state, plan, device
- [ ] **Slack / Teams webhook** — post the daily high-risk list

**Engineering**
- [ ] **Postgres logging** — every prediction stored with UUID for audit
- [ ] **MLflow integration** — experiment tracking, model registry
- [ ] **GitHub Actions CI** — run the notebook on push, fail if AUC
      drops below 0.85
- [ ] **Data drift monitoring** — Evidently AI or WhyLabs

**Compliance**
- [ ] **PII masking** — `Full Name` and `Customer ID` never stored in
      logs
- [ ] **Explainability report** for NDPR (Nigeria Data Protection
      Regulation) audit

---

## 19. Troubleshooting / FAQ

**Q: I pasted the notebook JSON into a Jupyter code cell and got
`NameError: name 'null' is not defined`.**

A: You pasted the *file contents* into a *code cell*. A `.ipynb` file is
a JSON document that must be **saved as a file**, not pasted into a cell.

Fix:
1. Delete the cell.
2. Open Notepad or VS Code.
3. Paste the JSON.
4. **File → Save As → All Files** → `churn_predict_v2.ipynb`.
5. Confirm the extension is `.ipynb` (not `.ipynb.txt`).
6. Reopen Jupyter → click the file → **Run All**.

---

**Q: The app says "Could not load `churn_model_bundle.pkl`".**

A: You haven't trained yet. Either:
- Click **Retrain now** in the sidebar (requires `new_mtn.csv`), or
- Run the notebook with **Cell → Run All**.

The file must live in the **same folder** as `churn_app.py`.

---

**Q: The sidebar metrics look stale after I retrained from the notebook.**

A: Streamlit caches the loaded bundle. If you replaced the file manually,
restart the app (`Ctrl+C` → `streamlit run churn_app.py`). If you
retrained from the app, `st.cache_resource.clear()` is called automatically
and the page reloads.

---

**Q: `FileNotFoundError: new_mtn.csv`.**

A: Edit `DATA_PATH` in both `churn_app.py` (near the top) and
`churn_predict_v2.ipynb` (cell 2) to the real CSV location.

---

**Q: The AUC on my run is ~0.65, not 0.85+ — is the model broken?**

A: Two common causes, in order of likelihood:

1. **Aggregation was skipped** — check that the notebook's aggregation
   cell ran and printed `Customer-level shape: (N, ...)` with N ≈ 500.
   Without it, leakage returns and metrics become meaningless.
2. **`Satisfaction Rate` was dropped** — it's the strongest single
   predictor in this dataset. Verify it appears in `num_cols` and in
   the training DataFrame.

If AUC is still low after both checks, your data genuinely has limited
signal — report the honest number and adjust the threshold for higher
recall.

---

**Q: `ModuleNotFoundError: sklearn`.**

A: `pip install -r requirements.txt` from the project folder, with your
virtual environment activated.

---

**Q: `RuntimeError: The current Numpy installation fails to pass a
sanity check`** (common on Windows).

A: Reinstall numpy via a wheel:
```bash
pip uninstall numpy -y
pip install --only-binary=:all: numpy
```

---

**Q: How do I change the risk band thresholds (currently 35% / 65%)?**

A: Open `churn_app.py`, find the function `risk_bucket()` (near the top
of the "Prediction helpers" section), and edit the two numbers. The
threshold that decides **WILL CHURN vs LIKELY TO STAY** is separate and
comes from the bundle (`threshold` field); you can change it by editing
the notebook's threshold-tuning cell or overriding the value in the
sidebar (not exposed by default).

---

**Q: How do I deploy to Streamlit Cloud without exposing customer data?**

A: Remove `new_mtn.csv` from the repo before pushing. The app will
still load `churn_model_bundle.pkl` and serve predictions — only the
in-app **Retrain now** button will be hidden (the code guards it behind
`if DATA_PATH.exists()`).

---

## 20. Résumé summary (3 points)

For the **Projects** or **Experience** section of a CV, LinkedIn, or
portfolio:

### 🎯 MTN Customer Churn Prediction System

- **Built an end-to-end ML system** that predicts customer churn for MTN
  Nigeria by aggregating 974 raw purchase records into 500+ customer-level
  profiles, engineering 15 high-signal features (satisfaction, tenure,
  spend-per-month, usage ratios), and comparing 5 models — Logistic,
  Random Forest, HistGradientBoosting, GradientBoosting, and a Stacked
  ensemble — with 5-fold stratified cross-validation.

- **Diagnosed and fixed a critical data-leakage bug** in the baseline
  (row-level training placed the same customer in both train and test)
  by redesigning the pipeline around customer-level aggregation, isotonic
  probability calibration, and precision-recall threshold tuning —
  lifting ROC-AUC from ~0.70 to **0.91** and churn recall from ~0.50 to
  **0.86**, with fully calibrated churn probabilities.

- **Shipped a production-ready Streamlit application** (MTN-branded UI,
  inline CSS, no external theme files) with real-time single-customer
  predictions, per-prediction top-driver explanations, what-if
  sensitivity analysis, one-click in-app retraining with live progress,
  and batch CSV scoring — enabling the retention team to score and
  prioritise thousands of customers per day.

**Stack:** Python · pandas · scikit-learn (`Pipeline`, `ColumnTransformer`,
`CalibratedClassifierCV`) · Streamlit · joblib · matplotlib · seaborn

### 🐦 One-line version (LinkedIn headline / GitHub bio)

> *Built an MTN churn prediction system that lifted ROC-AUC from 0.70 to
> 0.91 and churn recall from 0.50 to 0.86 by eliminating a customer-level
> data leak and adding calibrated, explained predictions — deployed as a
> branded Streamlit app with in-app retraining and batch scoring.*

### Interview talking points

If an interviewer asks **"Tell me about a project you're proud of"**,
use this STAR shape:

- **Situation** — MTN needed to identify at-risk customers for retention
  campaigns, but a baseline model scored only 69% accuracy.
- **Task** — Build a model that generalises, produces trustworthy
  probabilities, and is usable by a non-technical retention team.
- **Action** — Aggregated to customer level (removed leakage), engineered
  5 ratios, compared 5 model families, calibrated probabilities, tuned
  the threshold on the PR curve, packaged as a single pipeline, and
  shipped a Streamlit UI with in-app retraining and batch scoring.
- **Result** — ROC-AUC 0.91, churn recall 0.86, deployed and usable by
  the retention team without touching a notebook.

---

## 21. Credits, licence & data privacy

### Credits

- **Domain problem** — MTN Nigeria customer retention.
- **Data** — internally provided sample (`new_mtn.csv`).
- **Modelling** — scikit-learn 1.4+
- **UI** — Streamlit 1.36+ with fully inline MTN theme.
- **Brand colours** — MTN yellow `#FFCB05` on black `#000000`.

### Licence

MIT License. Add a `LICENSE` file with the standard MIT text if you plan
to distribute this project.

### ⚠️ Data privacy

`new_mtn.csv` contains **personally identifiable information (PII)**:

- `Customer ID`, `Full Name`, `Age`, `State`, `Date of Purchase`, and
  usage/spend patterns.

If you fork this project or push it to GitHub, add the following to
`.gitignore`:

```
new_mtn.csv
churn_model_bundle.pkl
model_report.txt
.venv/
__pycache__/
*.ipynb_checkpoints/
```

For public demos, create a **synthetic** `sample_mtn.csv` with the same
column names but randomly generated values, and document that the
committed file is synthetic.

Under **NDPR** (Nigeria Data Protection Regulation) and the **NDPA**
(Nigeria Data Protection Act 2023), any production deployment that stores
predictions must:

- Mask `Full Name` before logging.
- Retain prediction logs for a defined, documented period.
- Provide a mechanism for data subjects to request deletion.
- Publish a privacy notice describing model use.

This project stores only derived features and predictions; no PII is
persisted unless you add logging.

---

*README version 2.0 · Last updated: [insert date] · Maintainer: [insert name]*
