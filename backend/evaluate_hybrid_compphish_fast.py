import os
import re
import numpy as np
import pandas as pd
import joblib
import torch

from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel

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

BATCH_SIZE = 16


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
# TEXT CLEANING
# ============================================================

def clean_html_text(html):

    if not html:
        return ""

    try:
        soup = BeautifulSoup(
            html,
            "html.parser"
        )

        for tag in soup([
            "script",
            "style",
            "noscript"
        ]):
            tag.decompose()

        text = soup.get_text(
            separator=" ",
            strip=True
        )

    except Exception:
        text = html

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PHISHXPERT HYBRID - FAST COMPPHISH EXTERNAL VALIDATION")
print("=" * 70)

print("\nLoading CompPhish dataset...")

df = pd.read_csv(CSV_PATH)

df["id_str"] = df["id"].astype(str)

print("Total records:", len(df))


# ============================================================
# INDEX HTML
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
# MATCH
# ============================================================

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
# LOAD DISTILBERT
# ============================================================

print("\nLoading DistilBERT...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

bert_model = AutoModel.from_pretrained(
    MODEL_NAME
)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

bert_model.to(device)
bert_model.eval()

print("Device:", device)


# ============================================================
# READ HTML + PREPARE TEXT
# ============================================================

print("\nReading webpage contents...")

texts = []
engineered_features = []
labels = []
ids = []

for counter, (_, row) in enumerate(
    usable.iterrows(),
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

    text = clean_html_text(html)

    # Same first-512-character treatment used
    # by the project's bert_encoder.py.
    text = text[:512]

    texts.append(text)

    engineered_features.append(
        row[FEATURES].astype(float).values
    )

    labels.append(
        int(row["label"])
    )

    ids.append(
        row["id"]
    )

    if counter % 1000 == 0 or counter == len(usable):

        print(
            f"Prepared {counter}/{len(usable)}"
        )


X_engineered = np.asarray(
    engineered_features,
    dtype=np.float32
)

y_true = np.asarray(
    labels,
    dtype=np.int32
)

ids = np.asarray(ids)


print(
    "\nEngineered feature matrix:",
    X_engineered.shape
)


# ============================================================
# BATCHED DISTILBERT
# ============================================================

print("\nGenerating batched DistilBERT embeddings...")

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

    batch_texts = texts[start:end]

    encoded = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.no_grad():

        outputs = bert_model(
            **encoded
        )

    embeddings = (
        outputs
        .last_hidden_state[:, 0, :]
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    bert_vectors.append(
        embeddings
    )

    if end % 256 == 0 or end == total:

        print(
            f"Embedded {end}/{total}"
        )


X_bert = np.vstack(
    bert_vectors
)


print(
    "\nDistilBERT matrix:",
    X_bert.shape
)


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

print(
    "Hybrid matrix:",
    X_hybrid.shape
)


if X_hybrid.shape[1] != 791:

    raise ValueError(
        f"Expected 791 features, got "
        f"{X_hybrid.shape[1]}"
    )


# ============================================================
# LOAD TRAINED HYBRID
# ============================================================

print("\nLoading trained LightGBM Hybrid model...")

model = joblib.load(
    MODEL_PATH
)

print("Model loaded.")


# ============================================================
# PREDICTION
# ============================================================

print("\nRunning external predictions...")

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

logloss = log_loss(
    y_true,
    probability
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("COMPPHISH EXTERNAL VALIDATION RESULTS")
print("=" * 70)

print(f"Samples     : {len(y_true)}")
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

    "Sensitivity": recall,

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
    "evaluation/compphish_hybrid_predictions.csv"
)
print(
    "evaluation/compphish_hybrid_metrics.csv"
)
print(
    "evaluation/compphish_missing_html.csv"
)

print("\nExternal validation completed.")