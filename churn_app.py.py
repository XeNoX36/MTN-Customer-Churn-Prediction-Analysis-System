"""
MTN Customer Churn Predictor — Streamlit app (v3, integrated training)
-----------------------------------------------------------------------
Features:
  • Auto-loads churn_model_bundle.pkl if present.
  • Sidebar "Retrain" button runs the full pipeline in-app.
  • Live model card + leaderboard.
  • Rich single-customer prediction (gauge, drivers, recommendation).
  • Batch scoring: upload CSV → download scored CSV.
  • MTN-themed UI, all styling inline (no .streamlit/config.toml).

Run:
    streamlit run churn_app.py
"""
from __future__ import annotations

import io
import json
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────
# Page config (must be the first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MTN Churn Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# MTN palette + inline CSS
# ─────────────────────────────────────────────────────────────
MTN_YELLOW = "#FFCB05"
MTN_BLACK  = "#000000"
MTN_WHITE  = "#FFFFFF"
MTN_DANGER = "#D93025"
MTN_OK     = "#188038"
MTN_WARN   = "#F9AB00"
MTN_GREY   = "#1A1A1A"
MTN_PANEL  = "#0E0E0E"
MTN_BORDER = "#2A2A2A"

CUSTOM_CSS = f"""
<style>
    /* ---------- base ---------- */
    .stApp {{
        background-color: {MTN_BLACK};
        color: {MTN_WHITE};
        font-family: 'Segoe UI', system-ui, sans-serif;
    }}
    h1, h2, h3, h4, h5 {{
        color: {MTN_YELLOW} !important;
        font-weight: 800 !important;
        letter-spacing: -0.3px;
    }}
    h1 {{ border-bottom: 2px solid {MTN_YELLOW}; padding-bottom: 8px; }}
    hr {{ border-color: {MTN_BORDER}; }}

    /* ---------- sidebar ---------- */
    section[data-testid="stSidebar"] {{
        background-color: {MTN_PANEL};
        border-right: 1px solid {MTN_BORDER};
    }}
    section[data-testid="stSidebar"] * {{ color: {MTN_WHITE}; }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{ color: {MTN_YELLOW} !important; }}

    /* ---------- form card ---------- */
    div[data-testid="stForm"] {{
        border: 1px solid {MTN_BORDER};
        border-radius: 16px;
        padding: 22px;
        background-color: {MTN_PANEL};
        box-shadow: 0 4px 24px rgba(255, 203, 5, 0.06);
    }}

    /* ---------- inputs ---------- */
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] > div,
    .stTextArea textarea {{
        background-color: #171717 !important;
        color: {MTN_WHITE} !important;
        border: 1px solid #333 !important;
        border-radius: 10px !important;
        transition: border 0.2s;
    }}
    .stTextInput input:focus,
    .stNumberInput input:focus,
    .stSelectbox div[data-baseweb="select"] > div:focus-within {{
        border: 1px solid {MTN_YELLOW} !important;
        box-shadow: 0 0 0 2px rgba(255, 203, 5, 0.2);
    }}
    label, .stMarkdown p, .stCaption, .stSlider label {{ color: {MTN_WHITE} !important; }}

    /* ---------- buttons ---------- */
    .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {{
        background-color: {MTN_YELLOW};
        color: {MTN_BLACK};
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        transition: all 0.2s;
    }}
    .stButton > button:hover,
    .stFormSubmitButton > button:hover,
    .stDownloadButton > button:hover {{
        background-color: #e6b800;
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(255, 203, 5, 0.3);
    }}
    .stButton > button:disabled {{ background-color: #333; color: #777; }}

    /* ---------- result / metric cards ---------- */
    .card {{
        background: linear-gradient(145deg, #141414, #0b0b0b);
        border: 1px solid {MTN_BORDER};
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 18px;
        box-shadow: 0 6px 24px rgba(0,0,0,0.5);
    }}
    .card-primary {{ border: 2px solid {MTN_YELLOW}; }}
    .kpi {{ font-size: 1.9rem; font-weight: 800; color: {MTN_YELLOW}; }}
    .kpi-label {{ font-size: 0.8rem; letter-spacing: 1px;
                  text-transform: uppercase; color: #999; }}
    .risk-badge {{
        display: inline-block; padding: 8px 18px;
        border-radius: 22px; font-weight: 800; font-size: 1rem;
        letter-spacing: 0.5px;
    }}
    .risk-low  {{ background: {MTN_OK};     color: white; }}
    .risk-mid  {{ background: {MTN_WARN};   color: black; }}
    .risk-high {{ background: {MTN_DANGER}; color: white; }}

    /* ---------- recommendation chip ---------- */
    .rec {{
        background: rgba(255, 203, 5, 0.08);
        border-left: 4px solid {MTN_YELLOW};
        padding: 12px 16px; border-radius: 8px;
        margin: 8px 0; color: {MTN_WHITE};
    }}

    /* ---------- progress bar ---------- */
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, {MTN_YELLOW}, #ffd83a);
    }}

    /* ---------- tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px; border-bottom: 1px solid {MTN_BORDER};
    }}
    .stTabs [data-baseweb="tab"] {{
        background-color: transparent; color: #aaa;
        border-radius: 8px 8px 0 0; padding: 10px 18px;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: rgba(255, 203, 5, 0.1);
        color: {MTN_YELLOW} !important;
        border-bottom: 2px solid {MTN_YELLOW};
    }}

    /* ---------- dataframe ---------- */
    .stDataFrame {{ border: 1px solid {MTN_BORDER}; border-radius: 10px; }}

    /* ---------- animation ---------- */
    @keyframes fadeUp {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}
    .card, .rec {{ animation: fadeUp 0.35s ease-out; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Paths / constants
# ─────────────────────────────────────────────────────────────
BUNDLE_PATH  = Path("churn_model_bundle.pkl")
DATA_PATH    = Path("new_mtn.csv")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────────────────────
# Training (inlined — same logic as train_churn.py)
# ─────────────────────────────────────────────────────────────
def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Reasons for Churn"] = df["Reasons for Churn"].fillna("")
    agg = {
        "Age":                        "first",
        "Gender":                     "first",
        "State":                      "first",
        "Satisfaction Rate":          "mean",
        "Customer Tenure in months":  "mean",
        "Total Revenue":              "sum",
        "Unit Price":                 "mean",
        "Data Usage":                 "mean",
        "Number of Times Purchased":  "sum",
        "MTN Device":                 lambda s: s.mode().iat[0],
        "Subscription Plan":          lambda s: s.mode().iat[0],
        "Customer Churn Status":      "first",
        "Reasons for Churn":          "first",
    }
    cust = df.groupby("Customer ID").agg(agg).reset_index()
    cust["Churn"] = (cust["Customer Churn Status"] == "Yes").astype(int)
    return cust


def _engineer(cust: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    cust = cust.copy()
    cust["Spend_per_Month"]      = cust["Total Revenue"] / cust["Customer Tenure in months"].clip(lower=1)
    cust["Data_per_Purchase"]    = cust["Data Usage"] / cust["Number of Times Purchased"].clip(lower=1)
    cust["Revenue_per_Purchase"] = cust["Total Revenue"] / cust["Number of Times Purchased"].clip(lower=1)
    cust["High_Satisfaction"]    = (cust["Satisfaction Rate"] >= 4).astype(int)
    cust["Is_Long_Tenure"]       = (cust["Customer Tenure in months"] >= 36).astype(int)
    cust = cust.drop(columns=["Customer ID", "Customer Churn Status", "Reasons for Churn"])
    y = cust["Churn"]
    X = cust.drop(columns=["Churn"])
    return X, y


def _build_preprocess(num_cols, cat_cols, bin_cols):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    return ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale",  StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
        ("bin", SimpleImputer(strategy="most_frequent"), bin_cols),
    ])


def train_bundle(data_path: Path, progress_cb=None) -> tuple[dict, dict]:
    """
    Train the full pipeline. `progress_cb(pct, msg)` is called with progress
    updates so the UI can show a live progress bar.
    Returns (bundle, report_dict).
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import (
        GradientBoostingClassifier, HistGradientBoostingClassifier,
        RandomForestClassifier, StackingClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, average_precision_score, classification_report,
        confusion_matrix, precision_recall_curve, roc_auc_score,
    )
    from sklearn.model_selection import (
        GridSearchCV, StratifiedKFold, cross_val_score, train_test_split,
    )
    from sklearn.pipeline import Pipeline

    def step(pct, msg):
        if progress_cb:
            progress_cb(pct, msg)

    # 1. Load + aggregate
    step(0.05, "Loading raw data…")
    raw = pd.read_csv(data_path)
    step(0.10, "Aggregating to one row per customer…")
    cust = _aggregate(raw)
    X, y = _engineer(cust)

    num_cols = [
        "Age", "Satisfaction Rate", "Customer Tenure in months",
        "Total Revenue", "Unit Price", "Data Usage",
        "Number of Times Purchased", "Spend_per_Month",
        "Data_per_Purchase", "Revenue_per_Purchase",
    ]
    cat_cols = ["Gender", "State", "MTN Device", "Subscription Plan"]
    bin_cols = ["High_Satisfaction", "Is_Long_Tenure"]
    preprocess = _build_preprocess(num_cols, cat_cols, bin_cols)

    # 2. Split
    step(0.15, "Stratified train/test split…")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 3. Leaderboard
    step(0.22, "Fitting candidate models (5-fold CV)…")
    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.06, l2_regularization=0.5,
            random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            random_state=RANDOM_STATE),
    }
    leaderboard = {}
    for i, (name, clf) in enumerate(candidates.items()):
        pipe = Pipeline([("prep", preprocess), ("clf", clf)])
        auc_scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                     scoring="roc_auc", n_jobs=-1)
        leaderboard[name] = {
            "cv_roc_auc_mean": float(auc_scores.mean()),
            "cv_roc_auc_std":  float(auc_scores.std()),
        }
        step(0.22 + 0.12 * (i + 1), f"{name} done (AUC={auc_scores.mean():.3f})")

    stack = StackingClassifier(
        estimators=[
            ("rf",  candidates["Random Forest"]),
            ("hgb", candidates["HistGradientBoosting"]),
        ],
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
        n_jobs=-1, cv=5,
    )
    stack_pipe = Pipeline([("prep", preprocess), ("clf", stack)])
    stack_auc = cross_val_score(stack_pipe, X_train, y_train, cv=cv,
                                scoring="roc_auc", n_jobs=-1)
    leaderboard["Stacked Ensemble"] = {
        "cv_roc_auc_mean": float(stack_auc.mean()),
        "cv_roc_auc_std":  float(stack_auc.std()),
    }
    step(0.72, "Leaderboard complete")

    # 4. Tune RF
    step(0.76, "Tuning Random Forest…")
    rf_pipe = Pipeline([
        ("prep", preprocess),
        ("clf",  RandomForestClassifier(
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE)),
    ])
    grid = GridSearchCV(rf_pipe, {
        "clf__n_estimators":     [300, 500],
        "clf__max_depth":        [None, 12],
        "clf__min_samples_leaf": [1, 2],
    }, cv=cv, scoring="roc_auc", n_jobs=-1)
    grid.fit(X_train, y_train)
    best_pipe = grid.best_estimator_
    step(0.85, f"Best RF AUC={grid.best_score_:.3f}")

    # 5. Calibrate
    step(0.90, "Calibrating probabilities…")
    calibrated = CalibratedClassifierCV(best_pipe, method="isotonic", cv=5)
    calibrated.fit(X_train, y_train)

    # 6. Threshold + metrics
    step(0.95, "Computing final metrics…")
    probs = calibrated.predict_proba(X_test)[:, 1]
    prec, rec, thr = precision_recall_curve(y_test, probs)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    best_thr = float(thr[np.argmax(f1)])
    y_pred = (probs >= best_thr).astype(int)

    report = {
        "threshold":    best_thr,
        "roc_auc":      float(roc_auc_score(y_test, probs)),
        "pr_auc":       float(average_precision_score(y_test, probs)),
        "accuracy":     float(accuracy_score(y_test, y_pred)),
        "confusion":    confusion_matrix(y_test, y_pred).tolist(),
        "class_report": classification_report(
            y_test, y_pred, target_names=["No Churn", "Churn"], output_dict=True),
        "leaderboard":  leaderboard,
    }

    # 7. Feature importances
    inner_pipe = calibrated.calibrated_classifiers_[0].estimator
    rf = inner_pipe.named_steps["clf"]
    feat_names = inner_pipe.named_steps["prep"].get_feature_names_out()
    top_imp = (pd.Series(rf.feature_importances_, index=feat_names)
                 .sort_values(ascending=False).head(10))

    bundle = {
        "pipeline": calibrated,
        "threshold": best_thr,
        "feature_columns": X.columns.tolist(),
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "binary_columns": bin_cols,
        "top_importances": {str(k): float(v) for k, v in top_imp.items()},
        "metrics": {
            "roc_auc":  report["roc_auc"],
            "pr_auc":   report["pr_auc"],
            "accuracy": report["accuracy"],
        },
        "report": report,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "best_params": grid.best_params_,
    }
    joblib.dump(bundle, BUNDLE_PATH)
    step(1.0, "✅ Model trained and saved")
    return bundle, report


# ─────────────────────────────────────────────────────────────
# Bundle loading
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_bundle(path_str: str, _mtime: float):
    """_mtime ensures cache invalidation when the file changes."""
    return joblib.load(path_str)


def get_bundle():
    if not BUNDLE_PATH.exists():
        return None
    return load_bundle(str(BUNDLE_PATH), BUNDLE_PATH.stat().st_mtime)


# ─────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────
def row_from_inputs(inp: dict) -> pd.DataFrame:
    row = {
        "Age":                        inp["age"],
        "Satisfaction Rate":          inp["satisfaction"],
        "Customer Tenure in months":  max(inp["tenure"], 1),
        "Total Revenue":              inp["revenue"],
        "Unit Price":                 inp["unit_price"],
        "Data Usage":                 inp["data_usage"],
        "Number of Times Purchased":  max(inp["times_purchased"], 1),
        "Gender":                     inp["gender"],
        "State":                      inp["state"],
        "MTN Device":                 inp["device"],
        "Subscription Plan":          inp["plan"],
    }
    row["Spend_per_Month"]      = inp["revenue"] / max(inp["tenure"], 1)
    row["Data_per_Purchase"]    = inp["data_usage"] / max(inp["times_purchased"], 1)
    row["Revenue_per_Purchase"] = inp["revenue"] / max(inp["times_purchased"], 1)
    row["High_Satisfaction"]    = int(inp["satisfaction"] >= 4)
    row["Is_Long_Tenure"]       = int(inp["tenure"] >= 36)
    return pd.DataFrame([row])


def risk_bucket(p: float) -> tuple[str, str, str]:
    if p < 0.35:
        return "🟢 LOW RISK", "risk-low", "Keep engaging. Offer loyalty perks."
    if p < 0.65:
        return "🟡 MEDIUM RISK", "risk-mid", "Send a targeted retention offer this week."
    return "🔴 HIGH RISK", "risk-high", "Call within 48 hours. Escalate to retention team."


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
st.title("📱 MTN Customer Churn Predictor")
st.markdown(
    "Predict, explain, and act on customer churn — powered by a "
    "calibrated ensemble trained on your customer data."
)
st.divider()

# ─────────────────────────────────────────────────────────────
# SIDEBAR — model management
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Model")

    bundle = get_bundle()

    if bundle is None:
        st.warning("No trained model found.")
    else:
        m = bundle.get("metrics", {})
        trained = bundle.get("trained_at", "—")
        st.success("Model loaded ✅")
        col_a, col_b = st.columns(2)
        col_a.metric("ROC-AUC",  f"{m.get('roc_auc', 0):.3f}")
        col_b.metric("PR-AUC",   f"{m.get('pr_auc', 0):.3f}")
        col_a.metric("Accuracy", f"{m.get('accuracy', 0):.3f}")
        col_b.metric("Threshold", f"{bundle.get('threshold', 0):.2f}")
        st.caption(f"Trained: `{trained}`")

    st.divider()
    st.subheader("🔄 Retrain")
    st.caption(
        "Runs the full pipeline: aggregation → CV leaderboard → "
        "tuning → calibration → save. Takes ~30–90 s."
    )

    if not DATA_PATH.exists():
        st.error(f"Training data not found:\n`{DATA_PATH}`")
    else:
        if st.button("Retrain now", use_container_width=True):
            progress_bar = st.progress(0.0, text="Starting…")
            status = st.empty()

            def cb(pct, msg):
                progress_bar.progress(min(pct, 1.0), text=msg)
                status.caption(msg)

            try:
                with st.spinner("Training…"):
                    new_bundle, new_report = train_bundle(DATA_PATH, progress_cb=cb)
                st.success(f"Done! ROC-AUC={new_report['roc_auc']:.3f}  "
                           f"Accuracy={new_report['accuracy']:.3f}")
                st.cache_resource.clear()
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Training failed: {e}")

    st.divider()
    if bundle:
        with st.expander("📊 Leaderboard"):
            lb = bundle.get("report", {}).get("leaderboard", {})
            if lb:
                st.dataframe(
                    pd.DataFrame(lb).T.rename(columns={
                        "cv_roc_auc_mean": "CV ROC-AUC",
                        "cv_roc_auc_std":  "± std",
                    }).sort_values("CV ROC-AUC", ascending=False),
                    use_container_width=True,
                )
            else:
                st.caption("Retrain to populate the leaderboard.")

        with st.expander("🔑 Best hyperparameters"):
            st.json(bundle.get("best_params", {}))

# ─────────────────────────────────────────────────────────────
# EMPTY STATE — no model yet
# ─────────────────────────────────────────────────────────────
bundle = get_bundle()
if bundle is None:
    st.info(
        "👈 No model found. Click **Retrain now** in the sidebar to train one. "
        "It saves to `churn_model_bundle.pkl`."
    )
    st.stop()

pipeline    = bundle["pipeline"]
threshold   = bundle["threshold"]
importances = bundle["top_importances"]

# ─────────────────────────────────────────────────────────────
# Result card + what-if (drawn from the saved inputs on every rerun)
# ─────────────────────────────────────────────────────────────
def render_prediction(inp: dict, version: int) -> None:
    X_new = row_from_inputs(inp)

    try:
        prob = float(pipeline.predict_proba(X_new)[0, 1])
        pred = int(prob >= threshold)
    except Exception as e:
        # return (not st.stop) so the Batch and About tabs still render
        st.error(f"Prediction failed: {e}")
        return

    badge, css, advice = risk_bucket(prob)

    # ---- main result card ----
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(f"""
        <div class="card card-primary">
            <span class="risk-badge {css}">{badge}</span>
            <div class="kpi-label" style="margin-top:14px;">Churn probability</div>
            <div class="kpi">{prob*100:.1f}%</div>
            <div style="color:#aaa; margin-top:6px;">
                Decision threshold <b style="color:{MTN_YELLOW}">{threshold:.2f}</b>
                &nbsp;·&nbsp;
                Prediction
                <b style="color:{MTN_YELLOW}">
                {'WILL CHURN' if pred else 'LIKELY TO STAY'}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.progress(prob)

        st.markdown(f'<div class="rec"><b>Recommended action:</b> {advice}</div>',
                    unsafe_allow_html=True)

    with c2:
        st.markdown("#### Why this prediction?")
        for feat, score in list(importances.items())[:6]:
            clean = (feat.replace("num__", "").replace("cat__", "")
                         .replace("bin__", "").replace("_", " "))
            bar = "█" * max(1, int(score * 40))
            st.markdown(
                f"**{clean}**  \n"
                f"<span style='color:{MTN_YELLOW}; font-family: monospace;'>"
                f"{bar}</span> &nbsp;<span style='color:#999'>{score:.3f}</span>",
                unsafe_allow_html=True,
            )

    # ---- what-if ----
    st.divider()
    st.markdown("### 🎛️ What-if: how sensitive is this customer?")
    st.caption("Drag a lever below to see how churn probability moves "
               "**without** changing the form above.")

    auc = bundle.get("metrics", {}).get("roc_auc", 1.0)
    if auc < 0.6:
        st.caption(f"⚠️ This model's ROC-AUC is {auc:.2f} (0.5 = guessing), so "
                   "what-if changes mostly reflect noise, not real customer behaviour.")

    # Slider limits are widened to include the submitted value; before, a tenure
    # above 120 (the form allows 130) crashed the slider.
    ten_max = max(120, int(inp["tenure"]))
    rev_max = max(2_000_000, int(inp["revenue"]))

    # `version` is in each key so a NEW prediction resets the sliders to the new
    # customer's values (a fixed key would keep the previous customer's what-if).
    wcol1, wcol2, wcol3 = st.columns(3)
    new_sat = wcol1.slider("Satisfaction (what-if)", 1, 5,
                           int(inp["satisfaction"]), key=f"w_sat_{version}")
    new_ten = wcol2.slider("Tenure in months (what-if)", 0, ten_max,
                           int(inp["tenure"]), key=f"w_ten_{version}")
    new_rev = wcol3.slider("Total Revenue ₦ (what-if)", 0, rev_max,
                           int(inp["revenue"]), step=10_000, key=f"w_rev_{version}")

    inp2 = {**inp, "satisfaction": new_sat, "tenure": new_ten, "revenue": new_rev}
    X_what = row_from_inputs(inp2)
    try:
        prob_what = float(pipeline.predict_proba(X_what)[0, 1])
    except Exception:
        prob_what = prob

    delta = prob_what - prob
    st.metric(
        "New churn probability",
        f"{prob_what*100:.1f}%",
        delta=f"{delta*100:+.1f} pp vs current",
        delta_color="inverse",
    )


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab_predict, tab_batch, tab_about = st.tabs(
    ["🎯 Predict", "📂 Batch scoring", "ℹ️ About"]
)

# ═════════════════════════════════════════════════════════════
# TAB 1 — single prediction
# ═════════════════════════════════════════════════════════════
with tab_predict:
    st.subheader("Customer details")
    st.caption("Leave the fields on defaults or pick values that describe the customer.")

    sample = st.checkbox("Use a sample at-risk customer", value=False)

    with st.form("churn_form", clear_on_submit=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", 10, 100, 27 if sample else 30)
            gender = st.selectbox("Gender", ["Male", "Female"],
                                   index=0 if not sample else 0)
            state = st.selectbox("State", [
                "Abuja (FCT)", "Kwara", "Sokoto", "Osun", "Bayelsa",
                "Kebbi", "Lagos", "Kano", "Rivers", "Oyo", "Other",
            ], index=1 if sample else 0)
            satisfaction = st.slider(
                "Satisfaction Rate (1=Poor, 5=Excellent)",
                1, 5, 1 if sample else 3,
            )

        with col2:
            device = st.selectbox("MTN Device", [
                "4G Router", "5G Broadband Router",
                "Broadband MiFi", "Mobile SIM Card",
            ])
            plan = st.selectbox("Subscription Plan", [
                "1GB+1.5mins Daily Plan", "7GB Monthly Plan",
                "12.5GB Monthly Plan", "25GB Monthly Plan",
                "30GB Monthly Broadband Plan",
                "150GB FUP Monthly Unlimited", "165GB Monthly Plan",
            ])
            tenure = st.number_input("Tenure (months)", 0, 130,
                                     2 if sample else 12)
            times_purchased = st.number_input("Times Purchased", 0, 200,
                                             19 if sample else 5)

        with col3:
            unit_price = st.number_input(
                "Unit Price (₦)", 0.0, 200_000.0, 35_000.0, step=500.0)
            revenue = st.number_input(
                "Total Revenue (₦)", 0.0, 5_000_000.0, 665_000.0, step=1_000.0)
            data_usage = st.number_input(
                "Data Usage (GB)", 0.0, 500.0, 44.48)

        submitted = st.form_submit_button("🔮 Predict churn risk",
                                           use_container_width=True)

    # IMPORTANT: Streamlit re-runs this whole script whenever ANY widget changes
    # (e.g. a what-if slider), and `submitted` is True only on the single run
    # right after the button click. So we save the submitted inputs in
    # session_state and draw the results from there. Otherwise, moving a
    # what-if slider made `submitted` False and the results + sliders vanished.
    if submitted:
        st.session_state["last_inp"] = dict(
            age=age, gender=gender, state=state, satisfaction=satisfaction,
            device=device, plan=plan, tenure=tenure,
            times_purchased=times_purchased, unit_price=unit_price,
            revenue=revenue, data_usage=data_usage,
        )
        # New prediction -> new version number -> what-if sliders start fresh
        st.session_state["inp_version"] = st.session_state.get("inp_version", 0) + 1

    if "last_inp" in st.session_state:
        render_prediction(st.session_state["last_inp"],
                          st.session_state["inp_version"])


# ═════════════════════════════════════════════════════════════
# TAB 2 — batch scoring
# ═════════════════════════════════════════════════════════════
with tab_batch:
    st.subheader("Score many customers at once")
    st.markdown(
        "Upload a CSV with these columns (any missing optional column will be filled with a median/mode):\n"
        "- `Age`, `Gender`, `State`, `Satisfaction Rate`\n"
        "- `MTN Device`, `Subscription Plan`, `Customer Tenure in months`\n"
        "- `Total Revenue`, `Unit Price`, `Data Usage`, `Number of Times Purchased`\n"
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
            st.write(f"Loaded **{len(batch)} rows**", batch.head(3))

            with st.spinner("Scoring…"):
                # Build the 15 features row-by-row via the same helper
                rows = []
                for _, r in batch.iterrows():
                    inp = dict(
                        age=float(r.get("Age", 30)),
                        gender=str(r.get("Gender", "Male")),
                        state=str(r.get("State", "Other")),
                        satisfaction=float(r.get("Satisfaction Rate", 3)),
                        device=str(r.get("MTN Device", "4G Router")),
                        plan=str(r.get("Subscription Plan", "7GB Monthly Plan")),
                        tenure=float(r.get("Customer Tenure in months", 12)),
                        times_purchased=float(r.get("Number of Times Purchased", 5)),
                        unit_price=float(r.get("Unit Price", 9000)),
                        revenue=float(r.get("Total Revenue", 50000)),
                        data_usage=float(r.get("Data Usage", 50)),
                    )
                    rows.append(row_from_inputs(inp))

                X_batch = pd.concat(rows, ignore_index=True)
                probs = pipeline.predict_proba(X_batch)[:, 1]
                preds = (probs >= threshold).astype(int)
                labels = np.where(
                    probs < 0.35, "Low",
                    np.where(probs < 0.65, "Medium", "High"),
                )
                out = batch.copy()
                out["churn_probability"] = np.round(probs, 4)
                out["churn_prediction"]  = np.where(preds == 1, "Yes", "No")
                out["risk_band"]         = labels

            st.success(f"Scored {len(out)} rows.")
            st.dataframe(out.head(20), use_container_width=True)

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download scored CSV",
                data=csv_bytes,
                file_name="mtn_churn_scored.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Could not process file: {e}")

# ═════════════════════════════════════════════════════════════
# TAB 3 — about
# ═════════════════════════════════════════════════════════════
with tab_about:
    st.subheader("How the model works")

    st.markdown(f"""
**Pipeline**

1. **Aggregation** — raw rows (one per purchase) are collapsed to **one row per `Customer ID`**. Churn is a customer property, so this is what makes the numbers honest.
2. **Feature engineering** — `Spend_per_Month`, `Data_per_Purchase`, `Revenue_per_Purchase`, plus `High_Satisfaction` (sat ≥ 4) and `Is_Long_Tenure` (tenure ≥ 36).
3. **Preprocessing** — median imputer + `StandardScaler` for numeric columns; mode imputer + `OneHotEncoder` for categorical columns.
4. **Models compared** — Logistic Regression, Random Forest, HistGradientBoosting, GradientBoosting, and a Stacked ensemble. Scored by 5-fold **ROC-AUC**.
5. **Tuning** — lightweight `GridSearchCV` on the Random Forest.
6. **Calibration** — `CalibratedClassifierCV(method="isotonic")` so the probability shown in the UI is trustworthy.
7. **Threshold** — chosen to maximise F1 for the *churn* class from the precision-recall curve (currently **{threshold:.2f}**).

**Current metrics (from the bundle)**

| Metric | Value |
|---|---|
| ROC-AUC | `{bundle.get('metrics', {}).get('roc_auc', 0):.4f}` |
| PR-AUC | `{bundle.get('metrics', {}).get('pr_auc', 0):.4f}` |
| Accuracy | `{bundle.get('metrics', {}).get('accuracy', 0):.4f}` |
| Threshold | `{threshold:.2f}` |

**Reading the risk bands**

- 🟢 **Low** — churn probability < 35%. Keep engaging.
- 🟡 **Medium** — 35 %–65 %. Send a retention offer.
- 🔴 **High** — ≥ 65 %. Call within 48 hours.
""")

    st.divider()
    st.markdown("### Why these metrics matter more than accuracy")
    st.markdown("""
For churn, **missing a real churner** (false negative) costs far more than
a wasted retention call (false positive). That's why the app reports
**ROC-AUC**, **PR-AUC**, and a **tuned threshold** — not raw accuracy.
If you want to catch more churners, lower the threshold (e.g. **0.45**);
if you want fewer false alarms, raise it (e.g. **0.70**).
""")
