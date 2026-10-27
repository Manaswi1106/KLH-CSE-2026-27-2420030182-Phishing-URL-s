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

INPUT = "evaluation/controlled_predictions.csv"
OUTPUT = "evaluation/bootstrap_confidence_intervals.csv"

N_BOOTSTRAP = 2000
RANDOM_STATE = 42

df = pd.read_csv(INPUT)

rng = np.random.default_rng(RANDOM_STATE)

metrics = {
    "Accuracy": lambda y, p, pred: accuracy_score(y, pred),
    "Precision": lambda y, p, pred: precision_score(
        y, pred, zero_division=0
    ),
    "Recall": lambda y, p, pred: recall_score(
        y, pred, zero_division=0
    ),
    "F1": lambda y, p, pred: f1_score(
        y, pred, zero_division=0
    ),
    "ROC_AUC": lambda y, p, pred: roc_auc_score(y, p),
    "PR_AUC": lambda y, p, pred: average_precision_score(y, p),
    "MCC": lambda y, p, pred: matthews_corrcoef(y, pred)
}

results = []

for model in df["Model"].unique():

    data = df[df["Model"] == model].reset_index(drop=True)

    y = data["True_Label"].to_numpy()
    pred = data["Predicted_Label"].to_numpy()
    prob = data["Probability_Phishing"].to_numpy()

    print("\n" + "=" * 70)
    print(model)
    print("=" * 70)

    for metric_name, metric_function in metrics.items():

        values = []

        for _ in range(N_BOOTSTRAP):

            sample_idx = rng.integers(
                0,
                len(y),
                len(y)
            )

            y_b = y[sample_idx]
            pred_b = pred[sample_idx]
            prob_b = prob[sample_idx]

            # ROC-AUC and PR-AUC require both classes.
            if metric_name in ["ROC_AUC", "PR_AUC"]:
                if len(np.unique(y_b)) < 2:
                    continue

            try:
                value = metric_function(
                    y_b,
                    prob_b,
                    pred_b
                )
                values.append(value)

            except Exception:
                continue

        values = np.array(values)

        lower = np.percentile(values, 2.5)
        upper = np.percentile(values, 97.5)
        mean = np.mean(values)

        results.append({
            "Model": model,
            "Metric": metric_name,
            "Estimate": mean,
            "CI_2.5": lower,
            "CI_97.5": upper
        })

        print(
            f"{metric_name:10s}: "
            f"{mean:.4f} "
            f"(95% CI: {lower:.4f} - {upper:.4f})"
        )

result_df = pd.DataFrame(results)

result_df.to_csv(
    OUTPUT,
    index=False
)

print("\n" + "=" * 70)
print("BOOTSTRAP EVALUATION COMPLETE")
print("=" * 70)

print("\nSaved:")
print(OUTPUT)