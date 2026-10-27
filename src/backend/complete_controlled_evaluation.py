import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    cohen_kappa_score, confusion_matrix, log_loss,
    roc_curve, precision_recall_curve
)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/hybrid_dataset.csv"
OUT_DIR = "evaluation"

os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PHISHXPERT COMPLETE CONTROLLED EVALUATION")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

X = df.drop(columns=["label"])
y = df["label"].astype(int).values

# First 23 = engineered features
X_engineered = X.iloc[:, :23].values

# Remaining 768 = DistilBERT
X_bert = X.iloc[:, 23:].values

# All 791 = Hybrid
X_hybrid = X.values


# ============================================================
# EXACT SAME SPLIT FOR EVERY MODEL
# ============================================================

indices = np.arange(len(y))

train_idx, test_idx = train_test_split(
    indices,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)

y_train = y[train_idx]
y_test = y[test_idx]

print("\nControlled split:")
print("Training samples:", len(train_idx))
print("Testing samples :", len(test_idx))

print("\nTraining class distribution:")
print(pd.Series(y_train).value_counts().sort_index())

print("\nTesting class distribution:")
print(pd.Series(y_test).value_counts().sort_index())


# ============================================================
# MODEL DEFINITIONS
# ============================================================

models = {

    "Random Forest - 23 Engineered": (
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        X_engineered
    ),

    "XGBoost - 23 Engineered": (
        XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1
        ),
        X_engineered
    ),

    "CatBoost - 23 Engineered": (
        CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            random_seed=42,
            verbose=False
        ),
        X_engineered
    ),

    "LightGBM - 23 Engineered": (
        LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        ),
        X_engineered
    ),

    "LightGBM - 768 DistilBERT": (
        LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        ),
        X_bert
    ),

    "PhishXpert Hybrid - 791": (
        LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            verbosity=-1
        ),
        X_hybrid
    )
}


# ============================================================
# STORAGE
# ============================================================

metric_rows = []
prediction_rows = []
roc_rows = []
pr_rows = []
threshold_rows = []
cm_rows = []


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, model, X_data):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    X_train = X_data[train_idx]
    X_test = X_data[test_idx]

    print("Training...")
    model.fit(X_train, y_train)

    print("Predicting...")
    y_prob = model.predict_proba(X_test)[:, 1]

    y_pred = (y_prob >= 0.50).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    mcc = matthews_corrcoef(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp)
    sensitivity = recall

    ll = log_loss(y_test, y_prob)

    print(f"Accuracy    : {accuracy * 100:.2f}%")
    print(f"Precision   : {precision * 100:.2f}%")
    print(f"Recall      : {recall * 100:.2f}%")
    print(f"F1          : {f1 * 100:.2f}%")
    print(f"ROC-AUC     : {roc_auc * 100:.2f}%")
    print(f"PR-AUC      : {pr_auc * 100:.2f}%")
    print(f"MCC         : {mcc:.4f}")
    print(f"Kappa       : {kappa:.4f}")
    print(f"Specificity : {specificity * 100:.2f}%")
    print(f"Log Loss    : {ll:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    # --------------------------------------------------------
    # Metrics CSV
    # --------------------------------------------------------

    metric_rows.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "MCC": mcc,
        "Kappa": kappa,
        "Specificity": specificity,
        "Sensitivity": sensitivity,
        "LogLoss": ll,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

    # --------------------------------------------------------
    # Individual predictions
    # --------------------------------------------------------

    for i, idx in enumerate(test_idx):

        prediction_rows.append({
            "Model": name,
            "Dataset_Index": int(idx),
            "True_Label": int(y_test[i]),
            "Predicted_Label": int(y_pred[i]),
            "Probability_Phishing": float(y_prob[i])
        })

    # --------------------------------------------------------
    # ROC
    # --------------------------------------------------------

    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)

    for a, b, c in zip(fpr, tpr, roc_thresholds):

        roc_rows.append({
            "Model": name,
            "FPR": float(a),
            "TPR": float(b),
            "Threshold": float(c)
        })

    # --------------------------------------------------------
    # Precision-Recall
    # --------------------------------------------------------

    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
        y_test,
        y_prob
    )

    for i in range(len(precision_curve)):

        threshold = (
            pr_thresholds[i]
            if i < len(pr_thresholds)
            else np.nan
        )

        pr_rows.append({
            "Model": name,
            "Precision": float(precision_curve[i]),
            "Recall": float(recall_curve[i]),
            "Threshold": float(threshold)
        })

    # --------------------------------------------------------
    # Threshold Analysis
    # --------------------------------------------------------

    thresholds = np.arange(0.10, 0.91, 0.01)

    for threshold in thresholds:

        pred = (y_prob >= threshold).astype(int)

        tn_t, fp_t, fn_t, tp_t = confusion_matrix(
            y_test,
            pred
        ).ravel()

        acc_t = accuracy_score(y_test, pred)
        prec_t = precision_score(
            y_test,
            pred,
            zero_division=0
        )
        rec_t = recall_score(
            y_test,
            pred,
            zero_division=0
        )
        f1_t = f1_score(
            y_test,
            pred,
            zero_division=0
        )

        spec_t = (
            tn_t / (tn_t + fp_t)
            if (tn_t + fp_t) > 0
            else 0
        )

        fpr_t = (
            fp_t / (fp_t + tn_t)
            if (fp_t + tn_t) > 0
            else 0
        )

        threshold_rows.append({
            "Model": name,
            "Threshold": round(float(threshold), 2),
            "Accuracy": acc_t,
            "Precision": prec_t,
            "Recall": rec_t,
            "F1": f1_t,
            "Specificity": spec_t,
            "FPR": fpr_t,
            "TN": tn_t,
            "FP": fp_t,
            "FN": fn_t,
            "TP": tp_t
        })


# ============================================================
# RUN ALL MODELS
# ============================================================

for name, (model, X_data) in models.items():

    evaluate_model(
        name,
        model,
        X_data
    )


# ============================================================
# SAVE RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SAVING EVALUATION RESULTS")
print("=" * 70)

pd.DataFrame(metric_rows).to_csv(
    f"{OUT_DIR}/complete_metrics.csv",
    index=False
)

pd.DataFrame(prediction_rows).to_csv(
    f"{OUT_DIR}/controlled_predictions.csv",
    index=False
)

pd.DataFrame(roc_rows).to_csv(
    f"{OUT_DIR}/roc_curve_data.csv",
    index=False
)

pd.DataFrame(pr_rows).to_csv(
    f"{OUT_DIR}/pr_curve_data.csv",
    index=False
)

pd.DataFrame(threshold_rows).to_csv(
    f"{OUT_DIR}/threshold_analysis_all_models.csv",
    index=False
)

pd.DataFrame(metric_rows)[
    ["Model", "TN", "FP", "FN", "TP"]
].to_csv(
    f"{OUT_DIR}/confusion_matrices.csv",
    index=False
)


# ============================================================
# BEST THRESHOLD BY F1
# ============================================================

threshold_df = pd.DataFrame(threshold_rows)

best_thresholds = (
    threshold_df
    .sort_values(
        ["Model", "F1"],
        ascending=[True, False]
    )
    .groupby("Model")
    .head(1)
)

best_thresholds.to_csv(
    f"{OUT_DIR}/best_threshold_by_f1.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\nEvaluation completed successfully.")

print("\nFiles created:")

for filename in [
    "complete_metrics.csv",
    "controlled_predictions.csv",
    "roc_curve_data.csv",
    "pr_curve_data.csv",
    "threshold_analysis_all_models.csv",
    "confusion_matrices.csv",
    "best_threshold_by_f1.csv"
]:

    print(
        "  ",
        os.path.join(OUT_DIR, filename)
    )

print("\nFinal metric summary:")
print(
    pd.DataFrame(metric_rows)[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC_AUC",
            "PR_AUC"
        ]
    ].to_string(index=False)
)