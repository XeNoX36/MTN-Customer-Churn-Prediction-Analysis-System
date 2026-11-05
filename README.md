# 📱 MTN Customer Churn Predictor

An end-to-end machine-learning project that predicts **whether an MTN
customer is likely to churn**, explains **why** for each individual
customer, and gives the retention team a **recommendation** they can act on
— all wrapped in an MTN-branded Streamlit app.

Built from a raw telecom dataset of ~500 customers and ~1,000 subscription
records. The project fixes the data-leakage and feature-selection problems
present in earlier versions and reaches **ROC-AUC ≈ 0.85–0.92** on a
properly isolated test set — a large improvement over the initial ~0.69
accuracy.

---

## Table of contents

1. [Why this project matters](#1-why-this-project-matters)
2. [What's inside](#2-whats-inside)
3. [The dataset](#3-the-dataset)
4. [How the model works](#4-how-the-model-works)
5. [Project structure](#5-project-structure)
6. [Setup](#6-setup)
7. [Run the notebook](#7-run-the-notebook)
8. [Run the Streamlit app](#8-run-the-streamlit-app)
9. [The metrics that matter](#9-the-metrics-that-matter)
10. [Reading the app's output](#10-reading-the-apps-output)
11. [Batch scoring](#11-batch-scoring)
12. [Retraining](#12-retraining)
13. [Deploy to Streamlit Community Cloud](#13-deploy-to-streamlit-community-cloud)
14. [Design decisions & trade-offs](#14-design-decisions--trade-offs)
15. [Known limitations](#15-known-limitations)
16. [Roadmap](#16-roadmap)
17. [FAQ / troubleshooting](#17-faq--troubleshooting)
18. [Credits & licence](#18-credits--licence)

---

## 1. Why this project matters

In telecoms, **retaining a customer is 5–25× cheaper than acquiring a new
one**. But retention teams can only call a limited number of at-risk
customers each week. The useful question is therefore **not** *"will this
customer churn, yes or no?"* — it is:

> *"Of the 500 customers on my list, who should I call **first**?"*

A good churn model must:

- **Rank** customers by churn risk so the retention team knows who to call.
- **Explain** *why* for each customer, so the call feels personal rather
  than scripted.
- **Be calibrated** — if the model says 80 %, then roughly 80 of every 100
  customers with that score should actually churn.
- **Be honest** — evaluated on customers the model has never seen, with no
  data leakage between train and test.

Every design choice in this project is driven by those four goals.

---

## 2. What's inside

| Capability | Where |
|---|---|
| Reproducible training notebook | `churn_predict_v2.ipynb` |
| Production-ready Streamlit app | `churn_app.py` |
| Trained model (pipeline + threshold + metrics + importances) | `churn_model_bundle.pkl` |
| Plain-text metrics report | `model_report.txt` |
| In-app **retraining** with live progress | Sidebar → *Retrain now* |
| Single-customer prediction with gauge + drivers + recommendation | *Predict* tab |
| **What-if** analysis (drag a slider → see how probability moves) | *Predict* tab |
| **Batch scoring** — upload CSV → download scored CSV | *Batch* tab |
| Methodology / metric explainer | *About* tab |
| MTN-branded UI, all styling inline (no external config) | `churn_app.py` → `CUSTOM_CSS` |

---

## 3. The dataset

`new_mtn.csv` — one row per subscription/purchase, ~1,000 rows, 20 columns.

| Column | Meaning | Used as |
|---|---|---|
| `Customer ID` | Unique customer identifier | Aggregation key |
| `Age` | Customer's age | Numeric feature |
| `Gender` | Male / Female | Categorical feature |
| `State` | Nigerian state | Categorical feature |
| `Satisfaction Rate` | 1 (Poor) – 5 (Excellent) | Numeric feature (strongest single predictor) |
| `Customer Review` | Text bucket (Poor → Very Good) | *Not used* (highly correlated with Satisfaction) |
| `Customer Tenure in months` | How long they've been a customer | Numeric feature + engineered flag |
| `MTN Device` | 4G Router / 5G Router / MiFi / SIM | Categorical feature |
| `Subscription Plan` | Plan name | Categorical feature |
| `Unit Price` | Average price paid (₦) | Numeric feature |
| `Number of Times Purchased` | Purchases on record | Numeric feature |
| `Total Revenue` | Total spend (₦) | Numeric feature |
| `Data Usage` | Average GB used | Numeric feature |
| `Customer Churn Status` | **Yes / No** | **Target** |
| `Reasons for Churn` | Free-text reason | *Not used* (available only **after** churn — leakage) |
| `Date of Purchase`, `Month`, `Age_group`, `Tenures` | Derived / redundant | *Not used* |

### Data-leakage warning (read this)

Rows are **per subscription**, not per customer. The churn label is
**per customer**. If you train on raw rows, the same customer appears in
both train and test, and the model memorises the answer. That's why v1
scored "69 %" and why that number was meaningless.

**This project aggregates to one row per `Customer ID` before any split.**
That single change is the biggest driver of the improvement.

---

## 4. How the model works
