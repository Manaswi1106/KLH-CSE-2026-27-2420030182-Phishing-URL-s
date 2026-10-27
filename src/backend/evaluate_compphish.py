import os
import torch
import torch.nn as nn
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


DATASET = r"D:\Documents\Aruna Grace\ALT\CompPhish\CompPhish\CompPhish_features.csv"

MODEL_PATH = r"D:\Documents\Aruna Grace\ALT\PhishXpert\models\phishxpert_optimized.pth"

SCALER_PATH = r"D:\Documents\Aruna Grace\ALT\PhishXpert\models\phishxpert_scaler.pkl"


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


class PhishXpert(nn.Module):

    def __init__(self, url_features=13, html_features=10):

        super().__init__()

        self.url_encoder = nn.Sequential(
            nn.Linear(url_features, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(48, 24),
            nn.ReLU()
        )

        self.html_encoder = nn.Sequential(
            nn.Linear(html_features, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(48, 24),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(48, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(48, 24),
            nn.ReLU()
        )

        self.classifier = nn.Linear(24, 1)


    def forward(self, url_x, html_x):

        url_out = self.url_encoder(url_x)

        html_out = self.html_encoder(html_x)

        combined = torch.cat(
            (url_out, html_out),
            dim=1
        )

        fused = self.fusion(combined)

        return self.classifier(fused)


print("=" * 60)
print("PHISHXPERT → COMPPHISH EXTERNAL EVALUATION")
print("=" * 60)


print("\nLoading dataset...")

df = pd.read_csv(DATASET)

print("Total records:", len(df))


# Remove records without HTML
valid_df = df[df["html_length"] > 0].copy()

print("Records with HTML:", len(valid_df))
print("Records without HTML:", len(df) - len(valid_df))


X = valid_df[FEATURES]

y = valid_df["label"]


print("\nLabels:")

print(y.value_counts().sort_index())


print("\nLoading scaler...")

scaler = joblib.load(SCALER_PATH)

X_scaled = scaler.transform(X)


url_x = torch.tensor(
    X_scaled[:, :13],
    dtype=torch.float32
)

html_x = torch.tensor(
    X_scaled[:, 13:],
    dtype=torch.float32
)


print("Loading PhishXpert model...")

model = PhishXpert()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()


print("\nRunning prediction...")


with torch.no_grad():

    outputs = model(
        url_x,
        html_x
    )

    probabilities = torch.sigmoid(
        outputs
    ).numpy().flatten()


predictions = (
    probabilities >= 0.5
).astype(int)


accuracy = accuracy_score(
    y,
    predictions
)

precision = precision_score(
    y,
    predictions,
    zero_division=0
)

recall = recall_score(
    y,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y,
    probabilities
)

pr_auc = average_precision_score(
    y,
    probabilities
)

cm = confusion_matrix(
    y,
    predictions
)


print("\n" + "=" * 60)
print("COMPPHISH RESULTS")
print("=" * 60)

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC   : {roc_auc * 100:.2f}%"
)

print(
    f"PR-AUC    : {pr_auc * 100:.2f}%"
)


print("\nConfusion Matrix:")

print(cm)


print("\nClassification Report:")

print(
    classification_report(
        y,
        predictions,
        target_names=[
            "Benign",
            "Phishing"
        ],
        digits=4
    )
)


print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)