import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

INPUT = "evaluation/compphish_3000_predictions.csv"
OUTPUT = "evaluation/compphish_3000_threshold_analysis.csv"

df = pd.read_csv(INPUT)

y = df["True_Label"].values
prob = df["Probability_Phishing"].values

results = []

for threshold in np.arange(0.10, 0.91, 0.01):

    pred = (prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred
    ).ravel()

    accuracy = accuracy_score(y, pred)

    precision = precision_score(
        y,
        pred,
        zero_division=0
    )

    recall = recall_score(
        y,
        pred,
        zero_division=0
    )

    f1 = f1_score(
        y,
        pred,
        zero_division=0
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    results.append({
        "Threshold": round(float(threshold), 2),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Specificity": specificity,
        "FPR": fpr,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    })

result = pd.DataFrame(results)

result.to_csv(
    OUTPUT,
    index=False
)

print("=" * 70)
print("COMPPHISH THRESHOLD ANALYSIS")
print("=" * 70)

print(
    result.sort_values(
        "F1",
        ascending=False
    ).head(10).to_string(index=False)
)

print("\nBest F1 threshold:")

print(
    result.loc[
        result["F1"].idxmax()
    ].to_string()
)

print("\nBest accuracy threshold:")

print(
    result.loc[
        result["Accuracy"].idxmax()
    ].to_string()
)

print("\nSaved:")
print(OUTPUT)