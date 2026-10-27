import os
import numpy as np
import pandas as pd
import joblib
import torch

from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
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
# CONFIG
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

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "distilbert-base-uncased"

SAMPLES_PER_CLASS = 1500

BATCH_SIZE = 8

RANDOM_STATE = 42


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
print("PHISHXPERT HYBRID - COMPPHISH 3000-SAMPLE VALIDATION")
print("=" * 70)

df = pd.read_csv(CSV_PATH)

df["id_str"] = df["id"].astype(str)


# ============================================================
# INDEX HTML RECURSIVELY
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

df["html_path"] = df["id_str"].map(html_map)

usable = df[
    df["html_path"].notna()
].copy()

print("Usable records:", len(usable))


# ============================================================
# STRATIFIED BALANCED SAMPLE
# ============================================================

print("\nSelecting reproducible external-validation sample...")

phishing = usable[
    usable["label"] == 1
]

benign = usable[
    usable["label"] == 0
]

print("Available phishing:", len(phishing))
print("Available benign :", len(benign))

phishing_sample = phishing.sample(
    n=SAMPLES_PER_CLASS,
    random_state=RANDOM_STATE
)

benign_sample = benign.sample(
    n=SAMPLES_PER_CLASS,
    random_state=RANDOM_STATE
)

sample = pd.concat(
    [
        phishing_sample,
        benign_sample
    ]
).sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

print("\nSelected records:", len(sample))

print(
    sample["label"]
    .value_counts()
    .sort_index()
)


# Save exact external-validation IDs
sample[
    ["id", "label"]
].to_csv(
    f"{OUTPUT_DIR}/compphish_3000_sample_ids.csv",
    index=False
)


# ============================================================
# LOAD DISTILBERT
# ============================================================

print("\nLoading DistilBERT...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

bert_model = AutoModel.from_pretrained(
    MODEL_NAME
)

device = torch.device("cpu")

bert_model.to(device)

bert_model.eval()

print("Device: CPU")
print("Batch size:", BATCH_SIZE)


# ============================================================
# READ HTML
# ============================================================

texts = []

engineered = []

labels = []

ids = []

print("\nReading HTML...")

for counter, (_, row) in enumerate(
    sample.iterrows(),
    start=1
):

    try:

        with open(
            row["html_path"],
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            html = f.read()

    except Exception:

        html = ""

    # Match project encoder behavior:
    # first 512 characters passed to tokenizer.
    text = html[:512]

    texts.append(text)

    engineered.append(
        row[FEATURES]
        .astype(float)
        .values
    )

    labels.append(
        int(row["label"])
    )

    ids.append(
        row["id"]
    )

    if counter % 250 == 0 or counter == len(sample):

        print(
            f"Prepared {counter}/{len(sample)}"
        )


X_engineered = np.asarray(
    engineered,
    dtype=np.float32
)

y_true = np.asarray(
    labels,
    dtype=np.int32
)


# ============================================================
# BATCHED DISTILBERT
# ============================================================

print("\nGenerating DistilBERT embeddings...")

bert_vectors = []

total = len(texts)

for start in range(
    0,
    total,
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        total
    )

    batch = texts[start:end]

    encoded = tokenizer(
        batch,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = bert_model(
            **encoded
        )

    vectors = (
        outputs
        .last_hidden_state[:, 0, :]
        .numpy()
        .astype(np.float32)
    )

    bert_vectors.append(vectors)

    if end % 250 == 0 or end == total:

        print(
            f"Embedded {end}/{total}"
        )


X_bert = np.vstack(
    bert_vectors
)


print("\nEngineered:", X_engineered.shape)

print("DistilBERT :", X_bert.shape)


# ============================================================
# FUSION
# ============================================================

X_hybrid = np.concatenate(
    [
        X_engineered,
        X_bert
    ],
    axis=1
)

print("Hybrid     :", X_hybrid.shape)


if X_hybrid.shape[1] != 791:

    raise ValueError(
        f"Expected 791 features, got "
        f"{X_hybrid.shape[1]}"
    )


# ============================================================
# LOAD D1 TRAINED MODEL
# ============================================================

print("\nLoading D1-trained Hybrid model...")

model = joblib.load(
    MODEL_PATH
)


# ============================================================
# PREDICTION
# ============================================================

print("Running predictions...")

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

specificity = tn / (tn + fp)

logloss = log_loss(
    y_true,
    probability
)


# ============================================================
# PRINT
# ============================================================

print("\n" + "=" * 70)
print("COMPPHISH EXTERNAL VALIDATION RESULTS")
print("=" * 70)

print("Samples     :", len(y_true))
print(f"Accuracy    : {accuracy * 100:.2f}%")
print(f"Precision   : {precision * 100:.2f}%")
print(f"Recall      : {recall * 100:.2f}%")
print(f"F1          : {f1 * 100:.2f}%")
print(f"ROC-AUC     : {roc_auc * 100:.2f}%")
print(f"PR-AUC      : {pr_auc * 100:.2f}%")
print(f"MCC         : {mcc:.4f}")
print(f"Kappa       : {kappa:.4f}")
print(f"Specificity : {specificity * 100:.2f}%")
print(f"Log Loss    : {logloss:.4f}")

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

pd.DataFrame({

    "id": ids,

    "True_Label": y_true,

    "Predicted_Label": prediction,

    "Probability_Phishing": probability

}).to_csv(
    f"{OUTPUT_DIR}/compphish_3000_predictions.csv",
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

pd.DataFrame([{

    "Dataset": "CompPhish",
    "Evaluation_Type": "Balanced external validation",
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
    "LogLoss": logloss,

    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp

}]).to_csv(
    f"{OUTPUT_DIR}/compphish_3000_metrics.csv",
    index=False
)


print("\nResults saved successfully.")