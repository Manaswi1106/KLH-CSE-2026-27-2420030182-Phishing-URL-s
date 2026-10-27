import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import joblib


BASE = r"D:\Documents\Aruna Grace\ALT"

COMPPHISH_DIR = os.path.join(
    BASE,
    "CompPhish",
    "CompPhish"
)

FEATURE_FILE = os.path.join(
    COMPPHISH_DIR,
    "CompPhish_features.csv"
)

MAPPING_FILE = os.path.join(
    COMPPHISH_DIR,
    "Mapping_File.xlsx"
)

MODEL_PATH = os.path.join(
    BASE,
    "PhishXpert",
    "models",
    "phishxpert_domain_generalized.pth"
)

SCALER_PATH = os.path.join(
    BASE,
    "PhishXpert",
    "models",
    "phishxpert_domain_generalized_scaler.pkl"
)

SEED = 42


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

    def __init__(
        self,
        url_features=13,
        html_features=10
    ):

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


def get_hostname(url):

    from urllib.parse import urlparse

    try:

        url = str(url).strip()

        if not url.startswith(
            ("http://", "https://")
        ):
            url = "http://" + url

        hostname = urlparse(url).hostname

        if not hostname:
            return ""

        return hostname.lower().strip(".")

    except Exception:

        return ""


def get_registrable_domain(hostname):

    if not hostname:
        return ""

    if hostname.startswith("www."):
        hostname = hostname[4:]

    parts = hostname.split(".")

    if len(parts) < 2:
        return hostname

    common_two_level_suffixes = {
        "co.uk",
        "org.uk",
        "ac.uk",
        "gov.uk",
        "com.au",
        "net.au",
        "org.au",
        "com.br",
        "com.cn",
        "com.sg",
        "co.in",
        "firm.in",
        "net.in",
        "org.in",
        "gen.in",
        "ind.in",
        "co.jp",
        "ne.jp",
        "or.jp",
        "co.nz",
        "com.mx",
        "com.tr",
        "com.ar",
        "com.my",
        "com.ph",
        "co.za"
    }

    last_two = ".".join(parts[-2:])

    if (
        last_two in common_two_level_suffixes
        and len(parts) >= 3
    ):
        return ".".join(parts[-3:])

    return last_two


print("=" * 70)
print("PHISHXPERT THRESHOLD ANALYSIS")
print("=" * 70)


print("\nLoading data...")

features = pd.read_csv(FEATURE_FILE)

mapping = pd.read_excel(MAPPING_FILE)


usable_mask = features["html_length"] > 0

features = features[
    usable_mask
].copy().reset_index(drop=True)

mapping = mapping[
    usable_mask.values
].copy().reset_index(drop=True)


mapping["hostname"] = mapping["url"].apply(
    get_hostname
)

mapping["registrable_domain"] = mapping[
    "hostname"
].apply(
    get_registrable_domain
)


valid = mapping["registrable_domain"] != ""

features = features[
    valid.values
].reset_index(drop=True)

mapping = mapping[
    valid.values
].reset_index(drop=True)


groups = mapping[
    "registrable_domain"
]


splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=SEED
)


_, test_idx = next(
    splitter.split(
        features,
        features["label"],
        groups=groups
    )
)


X_test = features.iloc[
    test_idx
][FEATURES]

y_test = features.iloc[
    test_idx
]["label"]


print(
    "Domain-unseen test samples:",
    len(X_test)
)


print("\nLoading model and scaler...")

scaler = joblib.load(
    SCALER_PATH
)


model = PhishXpert()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()


X_scaled = scaler.transform(
    X_test
)


url_x = torch.tensor(
    X_scaled[:, :13],
    dtype=torch.float32
)

html_x = torch.tensor(
    X_scaled[:, 13:],
    dtype=torch.float32
)


with torch.no_grad():

    outputs = model(
        url_x,
        html_x
    )

    probabilities = torch.sigmoid(
        outputs
    ).numpy().flatten()


thresholds = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]


results = []


print("\n" + "=" * 70)
print("THRESHOLD RESULTS")
print("=" * 70)


for threshold in thresholds:

    predictions = (
        probabilities >= threshold
    ).astype(int)


    accuracy = accuracy_score(
        y_test,
        predictions
    )

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


    cm = confusion_matrix(
        y_test,
        predictions
    )


    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn)


    results.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr
    })


    print(
        f"\nThreshold: {threshold:.2f}"
    )

    print(
        f"Accuracy : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision: {precision * 100:.2f}%"
    )

    print(
        f"Recall   : {recall * 100:.2f}%"
    )

    print(
        f"F1       : {f1 * 100:.2f}%"
    )

    print(
        f"FPR      : {fpr * 100:.2f}%"
    )


results_df = pd.DataFrame(results)


best_f1 = results_df.loc[
    results_df["f1"].idxmax()
]

best_accuracy = results_df.loc[
    results_df["accuracy"].idxmax()
]


print("\n" + "=" * 70)
print("BEST THRESHOLD BY F1")
print("=" * 70)

print(
    f"Threshold : {best_f1['threshold']:.2f}"
)

print(
    f"Accuracy  : {best_f1['accuracy'] * 100:.2f}%"
)

print(
    f"Precision : {best_f1['precision'] * 100:.2f}%"
)

print(
    f"Recall    : {best_f1['recall'] * 100:.2f}%"
)

print(
    f"F1        : {best_f1['f1'] * 100:.2f}%"
)

print(
    f"FPR       : {best_f1['fpr'] * 100:.2f}%"
)


print("\n" + "=" * 70)
print("BEST THRESHOLD BY ACCURACY")
print("=" * 70)

print(
    f"Threshold : {best_accuracy['threshold']:.2f}"
)

print(
    f"Accuracy  : {best_accuracy['accuracy'] * 100:.2f}%"
)

print(
    f"Precision : {best_accuracy['precision'] * 100:.2f}%"
)

print(
    f"Recall    : {best_accuracy['recall'] * 100:.2f}%"
)

print(
    f"F1        : {best_accuracy['f1'] * 100:.2f}%"
)

print(
    f"FPR       : {best_accuracy['fpr'] * 100:.2f}%"
)


OUTPUT_FILE = os.path.join(
    BASE,
    "PhishXpert",
    "threshold_results.csv"
)


results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nResults saved to:")

print(
    OUTPUT_FILE
)

print("\n" + "=" * 70)
print("THRESHOLD ANALYSIS COMPLETE")
print("=" * 70)