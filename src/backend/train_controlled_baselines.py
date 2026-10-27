import os
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
    log_loss
)

from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# ============================================================
# OPTIONAL MODELS
# ============================================================

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except Exception:
    LIGHTGBM_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/hybrid_dataset.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20

ENGINEERED_FEATURES = 23
BERT_FEATURES = 768
TOTAL_FEATURES = 791


# ============================================================
# LOAD DATA
# ============================================================

print("\n============================================================")
print(" PHISHXPERT CONTROLLED BASELINE EVALUATION")
print("============================================================\n")

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

X = df.drop("label", axis=1)
y = df["label"]

print(f"Features: {X.shape[1]}")
print(f"Samples : {len(X)}")

if X.shape[1] != TOTAL_FEATURES:
    raise ValueError(
        f"Expected {TOTAL_FEATURES} features, "
        f"but found {X.shape[1]}"
    )

print("\nLabel distribution:")
print(y.value_counts().sort_index())


# ============================================================
# SAME SPLIT FOR EVERY MODEL
# ============================================================

indices = np.arange(len(df))

train_idx, test_idx = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n============================================================")
print(" CONTROLLED SPLIT")
print("============================================================")

print(f"Training samples: {len(train_idx)}")
print(f"Testing samples : {len(test_idx)}")

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

print("\nTraining labels:")
print(y_train.value_counts().sort_index())

print("\nTesting labels:")
print(y_test.value_counts().sort_index())


# ============================================================
# FEATURE GROUPS
# ============================================================

X_train_engineered = X_train.iloc[:, :ENGINEERED_FEATURES]
X_test_engineered = X_test.iloc[:, :ENGINEERED_FEATURES]

X_train_bert = X_train.iloc[:, ENGINEERED_FEATURES:]
X_test_bert = X_test.iloc[:, ENGINEERED_FEATURES:]

X_train_hybrid = X_train
X_test_hybrid = X_test


# ============================================================
# METRIC FUNCTION
# ============================================================

def evaluate_model(name, model, Xtr, Xte):

    print("\n------------------------------------------------------------")
    print(f"TRAINING: {name}")
    print("------------------------------------------------------------")

    model.fit(Xtr, y_train)

    predictions = model.predict(Xte)

    probabilities = model.predict_proba(Xte)[:, 1]

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions
    ).ravel()

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )
    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )
    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )
    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )
    pr_auc = average_precision_score(
        y_test,
        probabilities
    )
    mcc = matthews_corrcoef(
        y_test,
        predictions
    )
    kappa = cohen_kappa_score(
        y_test,
        predictions
    )
    specificity = tn / (tn + fp)
    logloss = log_loss(
        y_test,
        probabilities
    )

    results = {
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
        "LogLoss": logloss,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }

    print(f"Accuracy      : {accuracy:.4f}")
    print(f"Precision     : {precision:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"F1 Score      : {f1:.4f}")
    print(f"ROC AUC       : {roc_auc:.4f}")
    print(f"PR AUC        : {pr_auc:.4f}")
    print(f"MCC           : {mcc:.4f}")
    print(f"Cohen Kappa   : {kappa:.4f}")
    print(f"Specificity   : {specificity:.4f}")
    print(f"Log Loss      : {logloss:.4f}")

    print("\nConfusion Matrix:")
    print(
        np.array([
            [tn, fp],
            [fn, tp]
        ])
    )

    return results


# ============================================================
# RESULTS STORAGE
# ============================================================

results = []


# ============================================================
# 1. RANDOM FOREST
# ============================================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

results.append(
    evaluate_model(
        "Random Forest - 23 Engineered",
        rf,
        X_train_engineered,
        X_test_engineered
    )
)


# ============================================================
# 2. XGBOOST
# ============================================================

if XGB_AVAILABLE:

    xgb = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1
    )

    results.append(
        evaluate_model(
            "XGBoost - 23 Engineered",
            xgb,
            X_train_engineered,
            X_test_engineered
        )
    )

else:
    print("\nXGBoost is not installed. Skipping XGBoost.")


# ============================================================
# 3. CATBOOST
# ============================================================

if CATBOOST_AVAILABLE:

    cat = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=RANDOM_STATE,
        verbose=False
    )

    results.append(
        evaluate_model(
            "CatBoost - 23 Engineered",
            cat,
            X_train_engineered,
            X_test_engineered
        )
    )

else:
    print("\nCatBoost is not installed. Skipping CatBoost.")


# ============================================================
# 4. LIGHTGBM - ENGINEERED ONLY
# ============================================================

if LIGHTGBM_AVAILABLE:

    lgb_engineered = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        verbosity=-1
    )

    results.append(
        evaluate_model(
            "LightGBM - 23 Engineered",
            lgb_engineered,
            X_train_engineered,
            X_test_engineered
        )
    )

else:
    print("\nLightGBM is not installed. Skipping LightGBM.")


# ============================================================
# 5. DISTILBERT ONLY
# ============================================================

if LIGHTGBM_AVAILABLE:

    lgb_bert = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        verbosity=-1
    )

    results.append(
        evaluate_model(
            "LightGBM - 768 DistilBERT",
            lgb_bert,
            X_train_bert,
            X_test_bert
        )
    )


# ============================================================
# 6. PHISHXPERT HYBRID
# ============================================================

if LIGHTGBM_AVAILABLE:

    lgb_hybrid = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        verbosity=-1
    )

    results.append(
        evaluate_model(
            "PhishXpert Hybrid - 791 Features",
            lgb_hybrid,
            X_train_hybrid,
            X_test_hybrid
        )
    )


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "controlled_baseline_results.csv",
    index=False
)

print("\n============================================================")
print(" FINAL CONTROLLED COMPARISON")
print("============================================================\n")

display_columns = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall",
    "F1",
    "ROC_AUC",
    "PR_AUC",
    "MCC",
    "Kappa",
    "Specificity",
    "LogLoss"
]

print(
    results_df[display_columns].to_string(
        index=False
    )
)

print("\n============================================================")
print(" SAVED")
print("============================================================")
print("controlled_baseline_results.csv")
print("\nEvaluation complete.\n")