import os
import pandas as pd
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse

BASE = r"D:\Documents\Aruna Grace\ALT"

MAPPING_FILE = os.path.join(
    BASE,
    "CompPhish",
    "CompPhish",
    "Mapping_File.xlsx"
)

FEATURE_FILE = os.path.join(
    BASE,
    "CompPhish",
    "CompPhish",
    "CompPhish_features.csv"
)

SEED = 42


def get_domain(url):
    try:
        url = str(url).strip()

        if not url.startswith(("http://", "https://")):
            url = "http://" + url

        hostname = urlparse(url).hostname

        if not hostname:
            return ""

        hostname = hostname.lower().strip(".")

        if hostname.startswith("www."):
            hostname = hostname[4:]

        return hostname

    except Exception:
        return ""


print("=" * 70)
print("COMPPHISH DOMAIN OVERLAP AUDIT")
print("=" * 70)


print("\nLoading Mapping_File.xlsx...")

mapping = pd.read_excel(MAPPING_FILE)

print("Mapping records:", len(mapping))
print("Mapping columns:", list(mapping.columns))


print("\nLoading CompPhish features...")

features = pd.read_csv(FEATURE_FILE)

print("Feature records:", len(features))


# Same filtering used in train_generalized.py
features = features[
    features["html_length"] > 0
].copy()

print("Usable records:", len(features))


# Reproduce the exact split used in train_generalized.py
train_idx, test_idx = train_test_split(
    features.index,
    test_size=0.20,
    random_state=SEED,
    stratify=features["label"]
)


print("\nReproduced split:")
print("CompPhish training:", len(train_idx))
print("CompPhish test    :", len(test_idx))


# ---------------------------------------------------------
# IMPORTANT:
# CompPhish_features.csv does not contain serial_number.
# The feature file was created from Mapping_File.xlsx
# in the same record order, so we align by row position.
# ---------------------------------------------------------

if len(mapping) != 15358:
    print("\nWARNING: Unexpected mapping size.")

if len(features) == 15124:

    # The original feature file contains 15358 rows.
    # We need to reproduce the same usable-row filtering
    # against the mapping file.

    original_features = pd.read_csv(FEATURE_FILE)

    usable_mask = (
        original_features["html_length"] > 0
    )

    mapping_usable = mapping.loc[
        usable_mask.values
    ].copy()

else:

    raise ValueError(
        "Unexpected number of usable feature records."
    )


if len(mapping_usable) != len(features):

    raise ValueError(
        f"Alignment failed: "
        f"mapping usable={len(mapping_usable)}, "
        f"features usable={len(features)}"
    )


print(
    "\nURLs aligned with feature records:",
    len(mapping_usable)
)


# Extract domains using exactly the same row order
domains = mapping_usable[
    "url"
].apply(
    get_domain
).reset_index(drop=True)


# Convert feature indices to positions
feature_indices = list(features.index)

train_positions = [
    feature_indices.index(i)
    for i in train_idx
]

test_positions = [
    feature_indices.index(i)
    for i in test_idx
]


train_domains = set(
    domains.iloc[train_positions]
)

test_domains = set(
    domains.iloc[test_positions]
)


train_domains.discard("")
test_domains.discard("")


overlap = train_domains.intersection(
    test_domains
)


print("\n" + "=" * 70)
print("DOMAIN RESULTS")
print("=" * 70)


print(
    "\nUnique training domains:",
    len(train_domains)
)

print(
    "Unique test domains    :",
    len(test_domains)
)

print(
    "Overlapping domains    :",
    len(overlap)
)


if len(test_domains) > 0:

    percentage = (
        len(overlap)
        / len(test_domains)
        * 100
    )

    print(
        f"Test domains also in training: "
        f"{percentage:.2f}%"
    )


print("\n" + "=" * 70)


if overlap:

    print("⚠️ DOMAIN OVERLAP DETECTED")

    print("=" * 70)

    print(
        "\nFirst 100 overlapping domains:\n"
    )

    for domain in sorted(overlap)[:100]:

        print(domain)

else:

    print("✅ NO DOMAIN OVERLAP FOUND")

    print("=" * 70)

    print(
        "\nNo common hostnames were found between "
        "the CompPhish training and test splits."
    )


print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)