# Spambase Email Classification — ML Assignment 2

## a. Problem statement

Email spam remains a practical binary classification problem: given numeric descriptors of an email (word frequencies, character frequencies, and capital-letter run statistics), predict whether the message is **spam (1)** or **not spam (0)**.

This project trains and compares five classical supervised classifiers on the same public dataset, reports Accuracy, AUC, Precision, Recall, F1, and Matthews Correlation Coefficient (MCC), and serves the saved models through an interactive Streamlit application. The web app evaluates uploaded test CSV data with pre-trained models only — it does not retrain.

## b. Dataset description

| Item | Detail |
|---|---|
| Name | **UCI Spambase** |
| Source | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/94/spambase) / OpenML `spambase` |
| Task | Binary classification |
| Instances | **4601** (≥ 500 required) |
| Features | **57** continuous predictors (≥ 12 required) |
| Target | `spam` — `1` = spam, `0` = not spam |
| Class balance | ~39.4% spam / ~60.6% not spam |
| Feature groups | 48 word-frequency attributes, 6 character-frequency attributes, 3 capital-run-length statistics |

Train/test split: **75% / 25%**, stratified, `random_state=42` (3450 train / 1151 test).  
The held-out portion is shipped as `test_data.csv` for Streamlit evaluation.

## c. Github Repository Link

**GitHub Repository:** https://github.com/Aayushi-Yadav20/spambase-ml-assignment

Repository layout:

```text
ML-Final/
├── app.py                 # Streamlit frontend
├── train_models.py        # Training + evaluation pipeline
├── requirements.txt
├── README.md
├── test_data.csv          # Held-out test set (features + spam label)
├── model/                 # Saved .joblib models for all five classifiers
└── artifacts/
    └── metrics.json       # Metrics & metadata from training
```

## d. Models used

All five models were trained on the same Spambase training split. Scaling (`StandardScaler`) is applied inside pipelines for Logistic Regression, KNN, and SVM. Decision Tree and Gaussian Naive Bayes consume raw features.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9279 | 0.9728 | 0.9304 | 0.8833 | 0.9062 | 0.8485 |
| Decision Tree | 0.9062 | 0.9427 | 0.8794 | 0.8833 | 0.8813 | 0.8037 |
| KNN | 0.9157 | 0.9621 | 0.8993 | 0.8855 | 0.8923 | 0.8232 |
| Naive Bayes | 0.8306 | 0.9474 | 0.7133 | 0.9537 | 0.8162 | 0.6893 |
| Support Vector Machine | 0.9288 | 0.9708 | 0.9326 | 0.8833 | 0.9072 | 0.8503 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Excellent baseline. Best AUC (0.9728) with strong, balanced precision/recall. Linear boundary works well after scaling on this dense numeric spam representation. |
| Decision Tree | Solid accuracy (~0.91) but lower AUC/MCC than linear/kernel models. Captures non-linear rules yet is more variance-prone; depth/leaf constraints limited overfitting. |
| KNN | Competitive mid-tier results (Accuracy 0.9157, AUC 0.9621). Distance weighting and scaling help, but 57-dimensional space limits neighbor reliability versus parametric models. |
| Naive Bayes | Weakest accuracy (0.8306) and MCC (0.6893). Very high recall (0.9537) but poor precision (0.7133) — many false positives. Independence assumption fits correlated word frequencies poorly. |
| Support Vector Machine | Best accuracy (0.9288), F1 (0.9072), and MCC (0.8503). RBF kernel with probability calibration stays close to LR on AUC while slightly improving hard-label decisions. |
| **Overall winner for this dataset?** | **Support Vector Machine** — tops Accuracy / F1 / MCC; Logistic Regression is a very close second (best AUC). |

## How to run locally

```bash
pip install -r requirements.txt
python train_models.py          # (re)train & refresh model/ + test_data.csv
streamlit run app.py
```

## Streamlit Cloud deployment

1. Push this repository to GitHub (include `model/`, `test_data.csv`, `artifacts/`).
2. Open [https://streamlit.io/cloud](https://streamlit.io/cloud) and sign in with GitHub.
3. **New App** → select this repo → branch `main` → main file `app.py` → **Deploy**.
4. In the app: upload `test_data.csv` (or use the bundled file), choose a model, and review metrics + confusion matrix.

## Live Streamlit App Link

**Live App:** https://spambase-ml-assignment-2026ak05554.streamlit.app

## Notes for evaluators

- Models are loaded from `model/*.joblib`; evaluation on uploaded CSV does **not** retrain.
- Uploaded CSV must contain all 57 feature columns and the `spam` target column.
- Assignment work should also be demonstrated on **BITS Virtual Lab** with a screenshot included in the submission PDF.
