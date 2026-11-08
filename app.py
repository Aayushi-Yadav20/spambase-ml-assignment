"""
BITS AIML Assignment-2 — Streamlit frontend
Author workspace: Aayushi-Yadav20 / spambase-ml-assignment
Evaluates pre-trained spam classifiers on uploaded test CSV (no retraining).
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

BASE_DIR = Path(__file__).resolve().parent
SAVED_MODELS = BASE_DIR / "model"
METRICS_FILE = BASE_DIR / "artifacts" / "metrics.json"
LABEL_COLUMN = "spam"
BUNDLED_TEST = BASE_DIR / "test_data.csv"

CLASSIFIER_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

PAGE_STYLE = """
<style>
    .stApp {
        background: linear-gradient(165deg, #f3f7f4 0%, #e8eef2 45%, #f7f4ef 100%);
    }
    .hero-block {
        padding: 1.1rem 1.35rem;
        border-left: 5px solid #1f6f5b;
        background: rgba(31, 111, 91, 0.08);
        border-radius: 0 12px 12px 0;
        margin-bottom: 1rem;
    }
    .hero-block h1 {
        color: #163a32;
        font-size: 1.85rem;
        margin: 0 0 0.35rem 0;
    }
    .hero-block p {
        color: #3d4f4a;
        margin: 0;
        font-size: 0.98rem;
    }
    .metric-note {
        font-size: 0.85rem;
        color: #5a6b66;
        margin-top: 0.4rem;
    }
</style>
"""


@st.cache_resource
def fetch_saved_classifier(joblib_name: str):
    model_path = SAVED_MODELS / joblib_name
    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find {model_path}. Train models first with train_models.py."
        )
    return joblib.load(model_path)


@st.cache_data
def fetch_holdout_summary() -> dict | None:
    if not METRICS_FILE.exists():
        return None
    with open(METRICS_FILE, encoding="utf-8") as handle:
        return json.load(handle)


def score_holdout(clf, features: pd.DataFrame, labels: pd.Series):
    preds = clf.predict(features)
    if hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(features)[:, 1]
        auc_value = float(roc_auc_score(labels, scores))
    else:
        auc_value = float("nan")

    summary = {
        "Accuracy": float(accuracy_score(labels, preds)),
        "AUC": auc_value,
        "Precision": float(precision_score(labels, preds, zero_division=0)),
        "Recall": float(recall_score(labels, preds, zero_division=0)),
        "F1": float(f1_score(labels, preds, zero_division=0)),
        "MCC": float(matthews_corrcoef(labels, preds)),
    }
    matrix = confusion_matrix(labels, preds)
    text_report = classification_report(
        labels,
        preds,
        target_names=["Ham / not-spam", "Spam"],
        zero_division=0,
    )
    return summary, matrix, text_report


def draw_matrix(matrix, classifier_label: str):
    fig, axis = plt.subplots(figsize=(5.0, 4.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        cbar=True,
        xticklabels=["Predicted Ham", "Predicted Spam"],
        yticklabels=["Actual Ham", "Actual Spam"],
        ax=axis,
        linewidths=0.4,
        linecolor="#d9e2ec",
    )
    axis.set_title(f"Confusion matrix — {classifier_label}")
    axis.set_xlabel("")
    axis.set_ylabel("")
    fig.tight_layout()
    return fig


def inject_header():
    st.markdown(PAGE_STYLE, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-block">
            <h1>Spam Mail Detector — Model Playground</h1>
            <p>
                BITS WILP AIML · Machine Learning Assignment 2 ·
                Compare five saved classifiers on UCI Spambase test rows.
                Upload CSV only — models are never re-fit in this app.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_sidebar(summary: dict | None):
    with st.sidebar:
        st.markdown("### Project card")
        st.write("**Student repo:** Aayushi-Yadav20")
        st.write("**Dataset:** UCI Spambase (57 features)")
        st.write("**Problem type:** Binary — spam vs ham")
        if summary:
            st.write(f"**Train rows:** {summary['n_train']}")
            st.write(f"**Bundled test rows:** {summary['n_test']}")
        st.divider()
        st.markdown(
            "CSV must include all input feature columns plus the target column "
            "`spam` (0 = ham, 1 = spam)."
        )
        st.caption("Saved artefacts live in the `model/` folder as `.joblib` files.")


def resolve_dataframe(uploaded_file, prefer_bundle: bool):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file), "your uploaded CSV"
    if prefer_bundle and BUNDLED_TEST.exists():
        return pd.read_csv(BUNDLED_TEST), "bundled test_data.csv"
    return None, None


def main():
    st.set_page_config(
        page_title="Spam Mail Detector | AIML A2",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_header()
    holdout_meta = fetch_holdout_summary()
    build_sidebar(holdout_meta)

    control_left, control_right = st.columns([1.15, 1])
    with control_left:
        uploaded_file = st.file_uploader(
            "Step 1 — Upload test CSV (test data only)",
            type=["csv"],
            help="Free Streamlit tier: upload the held-out test file, not the full training set.",
        )
    with control_right:
        chosen_model = st.selectbox(
            "Step 2 — Pick a trained classifier",
            options=list(CLASSIFIER_FILES.keys()),
            index=0,
        )
        use_bundle = st.toggle(
            "Fall back to bundled test_data.csv",
            value=uploaded_file is None,
        )
        show_leaderboard = st.toggle(
            "Show training-time leaderboard",
            value=True,
        )

    tabs = st.tabs(
        [
            "Live evaluation",
            "Training leaderboard",
            "How this app works",
        ]
    )

    with tabs[1]:
        if holdout_meta and show_leaderboard:
            st.subheader("Metrics recorded when models were trained")
            board = pd.DataFrame(holdout_meta["metrics"]).T.round(4)
            st.dataframe(board, use_container_width=True)
            st.markdown(
                '<p class="metric-note">Winner on this split: Support Vector Machine '
                "(best Accuracy / F1 / MCC). Logistic Regression leads on AUC.</p>",
                unsafe_allow_html=True,
            )
        else:
            st.info("Leaderboard hidden, or artifacts/metrics.json is missing.")

    with tabs[2]:
        st.markdown(
            """
            1. Models were trained offline in `train_models.py` and saved under `model/`.  
            2. This page only **loads** those files and scores the CSV you provide.  
            3. Required outputs for the assignment: metrics strip, confusion matrix,
               and classification report — all computed on the uploaded / bundled test set.
            """
        )

    with tabs[0]:
        frame, source_name = resolve_dataframe(uploaded_file, use_bundle)
        if frame is None:
            st.warning(
                "Upload a test CSV, or enable the bundled test_data.csv toggle."
            )
            return

        st.success(f"Ready — {len(frame)} rows loaded from **{source_name}**.")

        if LABEL_COLUMN not in frame.columns:
            st.error(f"Missing target column `{LABEL_COLUMN}` in the CSV.")
            return

        input_cols = [col for col in frame.columns if col != LABEL_COLUMN]
        if holdout_meta:
            needed = holdout_meta["feature_columns"]
            absent = [col for col in needed if col not in frame.columns]
            if absent:
                preview = ", ".join(absent[:8])
                extra = " ..." if len(absent) > 8 else ""
                st.error(f"CSV is missing feature columns: {preview}{extra}")
                return
            input_cols = needed

        X_test = frame[input_cols]
        y_test = frame[LABEL_COLUMN].astype(int)

        try:
            classifier = fetch_saved_classifier(CLASSIFIER_FILES[chosen_model])
        except FileNotFoundError as err:
            st.error(str(err))
            return

        scores, matrix, report_text = score_holdout(classifier, X_test, y_test)

        st.markdown(f"### Results for **{chosen_model}**")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Accuracy", f"{scores['Accuracy']:.4f}")
        c2.metric("AUC", f"{scores['AUC']:.4f}")
        c3.metric("Precision", f"{scores['Precision']:.4f}")
        c4.metric("Recall", f"{scores['Recall']:.4f}")
        c5.metric("F1", f"{scores['F1']:.4f}")
        c6.metric("MCC", f"{scores['MCC']:.4f}")

        viz_left, viz_right = st.columns(2)
        with viz_left:
            st.markdown("#### Confusion matrix")
            st.pyplot(draw_matrix(matrix, chosen_model), clear_figure=True)
        with viz_right:
            st.markdown("#### Classification report")
            st.code(report_text, language="text")

        with st.expander("Peek at the first 15 test rows"):
            st.dataframe(frame.head(15), use_container_width=True)

        st.caption(
            "Inference-only mode: uploaded data is scored with saved .joblib models; "
            "no training loop runs inside Streamlit."
        )


if __name__ == "__main__":
    main()
