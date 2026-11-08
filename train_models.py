"""
Train five classification models on the UCI Spambase dataset
and persist models, metrics, and a held-out test CSV.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
ARTIFACTS = ROOT / "artifacts"
TEST_CSV = ROOT / "test_data.csv"
TARGET_COL = "spam"


FEATURE_NAMES = [
    "word_freq_make",
    "word_freq_address",
    "word_freq_all",
    "word_freq_3d",
    "word_freq_our",
    "word_freq_over",
    "word_freq_remove",
    "word_freq_internet",
    "word_freq_order",
    "word_freq_mail",
    "word_freq_receive",
    "word_freq_will",
    "word_freq_people",
    "word_freq_report",
    "word_freq_addresses",
    "word_freq_free",
    "word_freq_business",
    "word_freq_email",
    "word_freq_you",
    "word_freq_credit",
    "word_freq_your",
    "word_freq_font",
    "word_freq_000",
    "word_freq_money",
    "word_freq_hp",
    "word_freq_hpl",
    "word_freq_george",
    "word_freq_650",
    "word_freq_lab",
    "word_freq_labs",
    "word_freq_telnet",
    "word_freq_857",
    "word_freq_data",
    "word_freq_415",
    "word_freq_85",
    "word_freq_technology",
    "word_freq_1999",
    "word_freq_parts",
    "word_freq_pm",
    "word_freq_direct",
    "word_freq_cs",
    "word_freq_meeting",
    "word_freq_original",
    "word_freq_project",
    "word_freq_re",
    "word_freq_edu",
    "word_freq_table",
    "word_freq_conference",
    "char_freq_semicolon",
    "char_freq_parenthesis",
    "char_freq_bracket",
    "char_freq_exclamation",
    "char_freq_dollar",
    "char_freq_hash",
    "capital_run_length_average",
    "capital_run_length_longest",
    "capital_run_length_total",
]


def load_spambase() -> pd.DataFrame:
    """Load UCI Spambase via OpenML (public repository dataset)."""
    data = fetch_openml(name="spambase", version=1, as_frame=True, parser="auto")
    frame = data.frame.copy()
    # OpenML uses class column; normalize to spam (1) / not-spam (0)
    if "class" in frame.columns:
        frame = frame.rename(columns={"class": TARGET_COL})
    frame[TARGET_COL] = frame[TARGET_COL].astype(int)
    # Keep stable feature names for the Streamlit app / README
    feature_cols = [c for c in frame.columns if c != TARGET_COL]
    if len(feature_cols) == len(FEATURE_NAMES):
        rename_map = dict(zip(feature_cols, FEATURE_NAMES))
        frame = frame.rename(columns=rename_map)
    return frame


def build_models() -> dict:
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12,
            min_samples_leaf=5,
            random_state=42,
        ),
        "KNN": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance")),
            ]
        ),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    CalibratedClassifierCV(
                        estimator=SVC(
                            kernel="rbf",
                            C=2.0,
                            gamma="scale",
                            random_state=42,
                        ),
                        method="sigmoid",
                        cv=3,
                    ),
                ),
            ]
        ),
    }


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float(roc_auc_score(y_true, y_proba)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print("Loading UCI Spambase dataset...")
    df = load_spambase()
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols]
    y = df[TARGET_COL]

    print(f"Instances: {len(df)}, Features: {len(feature_cols)}, Classes: {sorted(y.unique())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values
    test_df.to_csv(TEST_CSV, index=False)
    print(f"Saved test data -> {TEST_CSV} ({len(test_df)} rows)")

    metrics_table = {}
    confusion_matrices = {}
    models = build_models()

    for name, model in models.items():
        print(f"Training: {name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            # fallback for models without probability
            decision = model.decision_function(X_test)
            y_proba = 1.0 / (1.0 + np.exp(-decision))

        metrics_table[name] = compute_metrics(y_test, y_pred, y_proba)
        confusion_matrices[name] = confusion_matrix(y_test, y_pred).tolist()

        safe_name = name.lower().replace(" ", "_")
        out_path = MODEL_DIR / f"{safe_name}.joblib"
        joblib.dump(model, out_path)
        print(f"  saved -> {out_path}")
        print(f"  metrics -> {metrics_table[name]}")

    meta = {
        "dataset": "UCI Spambase (OpenML)",
        "source": "https://archive.ics.uci.edu/dataset/94/spambase",
        "target_column": TARGET_COL,
        "feature_columns": feature_cols,
        "class_labels": {0: "not_spam", 1: "spam"},
        "n_features": len(feature_cols),
        "n_instances_total": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "metrics": metrics_table,
        "confusion_matrices": confusion_matrices,
        "model_files": {
            name: f"model/{name.lower().replace(' ', '_')}.joblib"
            for name in models
        },
    }

    with open(ARTIFACTS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n=== Metrics summary ===")
    summary = pd.DataFrame(metrics_table).T.round(4)
    print(summary.to_string())
    print(f"\nArtifacts written to {ARTIFACTS}")


if __name__ == "__main__":
    main()
