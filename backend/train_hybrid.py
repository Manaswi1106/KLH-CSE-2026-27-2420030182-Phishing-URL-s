import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    matthews_corrcoef, cohen_kappa_score,
    confusion_matrix, log_loss
)

from lightgbm import LGBMClassifier

# Load hybrid dataset
df = pd.read_csv("data/hybrid_dataset.csv")

X = df.drop("label", axis=1)
y = df["label"]

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Hybrid ML classifier
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
print("Confusion Matrix:")
print([[tn, fp], [fn, tp]])

specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

print("\n========== PHISHXPERT HYBRID RESULTS ==========\n")
print("Accuracy      :", round(accuracy_score(y_test,pred),4))
print("Precision     :", round(precision_score(y_test,pred),4))
print("Recall        :", round(recall_score(y_test,pred),4))
print("F1 Score      :", round(f1_score(y_test,pred),4))
print("ROC AUC       :", round(roc_auc_score(y_test,prob),4))
print("PR AUC        :", round(average_precision_score(y_test,prob),4))
print("MCC           :", round(matthews_corrcoef(y_test,pred),4))
print("Cohen Kappa   :", round(cohen_kappa_score(y_test,pred),4))
print("Specificity   :", round(specificity,4))
print("Sensitivity   :", round(sensitivity,4))
print("Log Loss      :", round(log_loss(y_test,prob),4))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/hybrid_model.pkl")

print("\nHybrid model saved successfully!")