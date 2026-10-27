import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef
)

INPUT = "evaluation/compphish_3000_predictions.csv"
OUTPUT = "evaluation/compphish_3000_bootstrap_ci.csv"

N_BOOTSTRAP = 2000
RANDOM_STATE = 42

df = pd.read_csv(INPUT)

y = df["True_Label"].to_numpy()
pred = df["Predicted_Label"].to_numpy()
prob = df["Probability_Phishing"].to_numpy()

rng = np.random.default_rng(RANDOM_STATE)

metrics = {
    "Accuracy": lambda a, b, c: accuracy_score(a, c),
    "Precision": lambda a, b, c: precision_score(
        a, c, zero_division=0
    ),
    "Recall": lambda a, b, c: recall_score(
        a, c, zero_division=0
    ),
    "F1": lambda a, b, c: f1_score(
        a, c, zero_division=0
    ),
    "ROC_AUC": lambda a, b, c: roc_auc_score(a, b),
    "PR_AUC": lambda a, b, c: average_precision_score(a, b),
    "MCC": lambda a, b, c: matthews_corrcoef(a, c)
}

results = []

print("=" * 70)
print("COMPPHISH BOOTSTRAP CONFIDENCE INTERVALS")
print("=" * 70)

for metric_name, function in metrics.items():

    values = []

    for _ in range(N_BOOTSTRAP):

        idx = rng.integers(
            0,
            len(y),
            len(y)
        )

        y_b = y[idx]
        pred_b = pred[idx]
        prob_b = prob[idx]

        if metric_name in ["ROC_AUC", "PR_AUC"]:
            if len(np.unique(y_b)) < 2:
                continue

        try:
            value = function(
                y_b,
                prob_b,
                pred_b
            )

            values.append(value)

        except Exception:
            continue

    values = np.asarray(values)

    estimate = np.mean(values)
    lower = np.percentile(values, 2.5)
    upper = np.percentile(values, 97.5)

    results.append({
        "Dataset": "CompPhish",
        "Samples": len(y),
        "Metric": metric_name,
        "Estimate": estimate,
        "CI_2.5": lower,
        "CI_97.5": upper
    })

    print(
        f"{metric_name:10s}: "
        f"{estimate:.4f} "
        f"(95% CI: {lower:.4f} - {upper:.4f})"
    )

pd.DataFrame(results).to_csv(
    OUTPUT,
    index=False
)

print("\nSaved:")
print(OUTPUT)