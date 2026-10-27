import os
import numpy as np
import pandas as pd
import joblib
import torch

from bert_encoder import get_bert_embedding
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
    log_loss
)


# ============================================================
# PATHS
# ============================================================

COMP_DIR = r"D:\Documents\Aruna Grace\ALT\CompPhish\CompPhish"

CSV_PATH = os.path.join(
    COMP_DIR,
    "CompPhish_features.csv"
)

HTML_DIR = os.path.join(
    COMP_DIR,
    "All_HTML"
)

MODEL_PATH = "models/hybrid_model.pkl"

OUTPUT_DIR = "evaluation"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

FEATURES = [
    "url_length",
    "domain_length",
    "path_length",
    "query_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_digits",
    "num_special_chars",
    "url_entropy",
    "has_https",
    "has_ip",
    "suspicious_keyword_count",
    "html_length",
    "text_length",
    "num_links",
    "external_links",
    "external_link_ratio",
    "form_count",
    "password_field_count",
    "has_login_form",
    "external_form_count",
    "external_form_ratio"
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PHISHXPERT HYBRID - COMPPHISH EXTERNAL VALIDATION")
print("=" * 70)

print("\nLoading CompPhish dataset...")

df = pd.read_csv(CSV_PATH)

print("Total records:", len(df))


# ============================================================
# MAP HTML FILES RECURSIVELY
# ============================================================

print("\nIndexing HTML files...")

html_map = {}

for root, dirs, files in os.walk(HTML_DIR):

    for filename in files:

        if filename.lower().endswith(".txt"):

            file_id = os.path.splitext(filename)[0]

            html_map[file_id] = os.path.join(
                root,
                filename
            )

print("HTML files found:", len(html_map))


# ============================================================
# MATCH RECORDS WITH HTML
# ============================================================

df["id_str"] = df["id"].astype(str)

df["html_path"] = df["id_str"].map(
    html_map
)

usable = df[
    df["html_path"].notna()
].copy()

missing = df[
    df["html_path"].isna()
].copy()

print("\nRecords with HTML:", len(usable))
print("Records without HTML:", len(missing))

missing.to_csv(
    f"{OUTPUT_DIR}/compphish_missing_html.csv",
    index=False
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading trained Hybrid model...")

model = joblib.load(MODEL_PATH)

print("Model loaded.")


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)


# ============================================================
# GENERATE HYBRID FEATURES
# ============================================================

print("\nGenerating DistilBERT embeddings...")

X_engineered = []
X_bert = []
y_true = []
ids = []

total = len(usable)

for counter, (_, row) in enumerate(
    usable.iterrows(),
    start=1
):

    # --------------------------------------------------------
    # 23 engineered features
    # --------------------------------------------------------

    engineered = row[FEATURES].astype(float).values

    # --------------------------------------------------------
    # Read HTML text
    # --------------------------------------------------------

    try:

        with open(
            row["html_path"],
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            html_text = f.read()

    except Exception:

        html_text = ""

    # --------------------------------------------------------
    # DistilBERT
    # --------------------------------------------------------

    try:

        bert_vector = get_bert_embedding(
            html_text
        )

    except Exception as e:

        print(
            f"\nEmbedding error for ID {row['id']}: {e}"
        )

        continue

    X_engineered.append(
        engineered
    )

    X_bert.append(
        bert_vector
    )

    y_true.append(
        int(row["label"])
    )

    ids.append(
        row["id"]
    )

    if counter % 250 == 0 or counter == total:

        print(
            f"\rProcessed {counter}/{total}",
            end=""
        )


print("\n\nFeature generation complete.")


# ============================================================
# CREATE HYBRID MATRIX
# ============================================================

X_engineered = np.asarray(
    X_engineered,
    dtype=np.float32
)

X_bert = np.asarray(
    X_bert,
    dtype=np.float32
)

y_true = np.asarray(
    y_true,
    dtype=np.int32
)

ids = np.asarray(
    ids
)

print("\nEngineered shape:", X_engineered.shape)
print("DistilBERT shape:", X_bert.shape)


X_hybrid = np.concatenate(
    [
        X_engineered,
        X_bert
    ],
    axis=1
)

print("Hybrid shape:", X_hybrid.shape)


# ============================================================
# VERIFY FEATURE COUNT
# ============================================================

if X_hybrid.shape[1] != 791:

    raise ValueError(
        f"Expected 791 features, got {X_hybrid.shape[1]}"
    )


# ============================================================
# PREDICTION
# ============================================================

print("\nRunning Hybrid model predictions...")

probability = model.predict_proba(
    X_hybrid
)[:, 1]

prediction = (
    probability >= 0.50
).astype(int)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    prediction
)

precision = precision_score(
    y_true,
    prediction,
    zero_division=0
)

recall = recall_score(
    y_true,
    prediction,
    zero_division=0
)

f1 = f1_score(
    y_true,
    prediction,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_true,
    probability
)

pr_auc = average_precision_score(
    y_true,
    probability
)

mcc = matthews_corrcoef(
    y_true,
    prediction
)

kappa = cohen_kappa_score(
    y_true,
    prediction
)

cm = confusion_matrix(
    y_true,
    prediction
)

tn, fp, fn, tp = cm.ravel()

specificity = tn / (
    tn + fp
)

sensitivity = recall

logloss = log_loss(
    y_true,
    probability
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("COMPPHISH EXTERNAL VALIDATION RESULTS")
print("=" * 70)

print(
    f"Accuracy    : {accuracy * 100:.2f}%"
)

print(
    f"Precision   : {precision * 100:.2f}%"
)

print(
    f"Recall      : {recall * 100:.2f}%"
)

print(
    f"F1          : {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC     : {roc_auc * 100:.2f}%"
)

print(
    f"PR-AUC      : {pr_auc * 100:.2f}%"
)

print(
    f"MCC         : {mcc:.4f}"
)

print(
    f"Kappa       : {kappa:.4f}"
)

print(
    f"Specificity : {specificity * 100:.2f}%"
)

print(
    f"Log Loss    : {logloss:.4f}"
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions = pd.DataFrame({

    "id": ids,

    "True_Label": y_true,

    "Predicted_Label": prediction,

    "Probability_Phishing": probability

})

predictions.to_csv(
    f"{OUTPUT_DIR}/compphish_hybrid_predictions.csv",
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame([{

    "Dataset": "CompPhish",

    "Samples": len(y_true),

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

    "LogLoss": logloss,

    "TN": tn,

    "FP": fp,

    "FN": fn,

    "TP": tp

}])

metrics.to_csv(
    f"{OUTPUT_DIR}/compphish_hybrid_metrics.csv",
    index=False
)


print("\nSaved:")

print(
    f"  {OUTPUT_DIR}/compphish_hybrid_predictions.csv"
)

print(
    f"  {OUTPUT_DIR}/compphish_hybrid_metrics.csv"
)

print(
    f"  {OUTPUT_DIR}/compphish_missing_html.csv"
)

print("\nExternal validation completed.")