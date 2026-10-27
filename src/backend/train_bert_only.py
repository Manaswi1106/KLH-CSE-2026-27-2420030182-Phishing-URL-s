import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

from lightgbm import LGBMClassifier


# Load hybrid dataset
df = pd.read_csv("data/hybrid_dataset.csv")

# DistilBERT features only
X = df[[f"f{i}" for i in range(768)]]
y = df["label"]

# Same split protocol as Hybrid model
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Same LightGBM configuration
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

print("\n========== DISTILBERT-ONLY RESULTS ==========\n")

print("Accuracy      :", round(accuracy_score(y_test, pred), 4))
print("Precision     :", round(precision_score(y_test, pred), 4))
print("Recall        :", round(recall_score(y_test, pred), 4))
print("F1 Score      :", round(f1_score(y_test, pred), 4))
print("ROC AUC       :", round(roc_auc_score(y_test, prob), 4))
print("PR AUC        :", round(average_precision_score(y_test, prob), 4))

print("\nConfusion Matrix:")
print([[tn, fp], [fn, tp]])