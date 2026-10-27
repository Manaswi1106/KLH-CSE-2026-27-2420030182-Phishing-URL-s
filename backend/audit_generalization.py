import pandas as pd
import numpy as np
import joblib
from pathlib import Path

DATASET = Path(r"D:\Documents\Aruna Grace\ALT\Dataset(D1)")

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

# D1 training data
train = pd.read_csv(DATASET / "X_train.csv")

print("=" * 70)
print("D1 TRAINING DISTRIBUTION")
print("=" * 70)

print(f"\nTraining samples: {len(train)}")

print("\nFeature statistics:\n")

stats = train[FEATURES].describe().T[
    ["mean", "std", "min", "25%", "50%", "75%", "max"]
]

print(stats.to_string())

# Check constant features
print("\n" + "=" * 70)
print("CONSTANT FEATURES")
print("=" * 70)

for feature in FEATURES:
    unique = train[feature].nunique()

    if unique <= 1:
        print(f"{feature}: CONSTANT ({unique} unique value)")

# Show scaler information
print("\n" + "=" * 70)
print("SCALER INFORMATION")
print("=" * 70)

scaler_path = Path("models/phishxpert_scaler.pkl")

if scaler_path.exists():
    scaler = joblib.load(scaler_path)

    print("\nFeature means and scales:\n")

    for feature, mean, scale in zip(
        FEATURES,
        scaler.mean_,
        scaler.scale_
    ):
        print(f"{feature:30} mean={mean:12.4f}  scale={scale:12.4f}")

else:
    print("Scaler not found.")

print("\n" + "=" * 70)
print("IMPORTANT")
print("=" * 70)

print("""
This audit checks whether the features used by live websites
are represented properly in the D1 training distribution.

Do NOT modify the model yet.

Next we will compare these statistics with the actual
Google/YouTube/GitHub/Wikipedia/CNN feature values.
""")