"""
Streamlit demo for spam email classification using five pre-trained models.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
ARTIFACTS = ROOT / "artifacts" / "metrics.json"
TARGET_COL = "spam"

MODEL_OPTIONS = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}


@st.cache_resource
def load_model(filename: str):
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing model file: {path}. Run train_models.py first."
        )
    return joblib.load(path)


@st.cache_data
def load_training_metrics() -> dict | None:
    if ARTIFACTS.exists():
        with open(ARTIFACTS, encoding="utf-8") as f:
            return json.load(f)
    return None


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> tuple[dict, object, str]:
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, 1]
        auc = float(roc_auc_score(y, y_proba))
    else:
        auc = float("nan")

    metrics = {
        "Accuracy": float(accuracy_score(y, y_pred)),
        "AUC": auc,
        "Precision": float(precision_score(y, y_pred, zero_division=0)),
        "Recall": float(recall_score(y, y_pred, zero_division=0)),
        "F1": float(f1_score(y, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, y_pred)),
    }
    cm = confusion_matrix(y, y_pred)
    report = classification_report(
        y,
        y_pred,
        target_names=["not_spam (0)", "spam (1)"],
        zero_division=0,
    )
    return metrics, cm, report


def plot_confusion_matrix(cm, title: str):
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["Actual 0", "Actual 1"],
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    fig.tight_layout()
    return fig


def main():
    st.set_page_config(
        page_title="Spambase Classifier Lab",
        layout="wide",
    )

    st.title("Spambase Email Classifier")
    st.caption(
        "Compare five classical ML models on the UCI Spambase dataset. "
        "Upload held-out test CSV data — models are pre-trained and never retrained here."
    )

    meta = load_training_metrics()
    with st.sidebar:
        st.header("About")
        st.markdown(
            """
            **Dataset:** UCI Spambase  
            **Task:** Binary classification (spam vs not-spam)  
            **Features:** 57 continuous inputs  
            **Models:** Logistic Regression, Decision Tree, KNN, Gaussian NB, SVM
            """
        )
        if meta:
            st.markdown(
                f"Training split used **{meta['n_train']}** rows; "
                f"bundled test set has **{meta['n_test']}** rows."
            )
        st.divider()
        st.markdown("Upload a CSV that includes all feature columns **and** the `spam` target.")

    uploaded = st.file_uploader("Upload test CSV", type=["csv"])
    model_name = st.selectbox("Select model", list(MODEL_OPTIONS.keys()))

    col_a, col_b = st.columns([1, 1])
    with col_a:
        use_bundled = st.checkbox("Use bundled test_data.csv", value=uploaded is None)
    with col_b:
        show_baseline = st.checkbox("Show training-set metrics table", value=True)

    if show_baseline and meta:
        st.subheader("Reference metrics (from training script hold-out)")
        ref = pd.DataFrame(meta["metrics"]).T
        st.dataframe(ref.style.format("{:.4f}"), use_container_width=True)

    if uploaded is not None:
        data = pd.read_csv(uploaded)
        source_label = "uploaded file"
    elif use_bundled and (ROOT / "test_data.csv").exists():
        data = pd.read_csv(ROOT / "test_data.csv")
        source_label = "bundled test_data.csv"
    else:
        st.info("Upload a test CSV (with target column `spam`) to evaluate a model.")
        return

    st.success(f"Loaded **{len(data)}** rows from {source_label}.")

    if TARGET_COL not in data.columns:
        st.error(f"Target column `{TARGET_COL}` is missing from the CSV.")
        return

    feature_cols = [c for c in data.columns if c != TARGET_COL]
    if meta:
        expected = meta["feature_columns"]
        missing = [c for c in expected if c not in data.columns]
        if missing:
            st.error(
                "Uploaded CSV is missing required feature columns: "
                + ", ".join(missing[:10])
                + (" ..." if len(missing) > 10 else "")
            )
            return
        feature_cols = expected

    X = data[feature_cols]
    y = data[TARGET_COL].astype(int)

    try:
        model = load_model(MODEL_OPTIONS[model_name])
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    metrics, cm, report = evaluate(model, X, y)

    st.subheader(f"Evaluation — {model_name}")
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
    m2.metric("AUC", f"{metrics['AUC']:.4f}")
    m3.metric("Precision", f"{metrics['Precision']:.4f}")
    m4.metric("Recall", f"{metrics['Recall']:.4f}")
    m5.metric("F1", f"{metrics['F1']:.4f}")
    m6.metric("MCC", f"{metrics['MCC']:.4f}")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Confusion matrix")
        st.pyplot(plot_confusion_matrix(cm, model_name), clear_figure=True)
    with right:
        st.markdown("#### Classification report")
        st.code(report)

    with st.expander("Preview loaded data"):
        st.dataframe(data.head(20), use_container_width=True)

    st.caption(
        "Predictions are produced by saved `.joblib` models only — "
        "no fitting occurs inside this app."
    )


if __name__ == "__main__":
    main()
