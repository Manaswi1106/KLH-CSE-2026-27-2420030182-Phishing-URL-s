import os
import pandas as pd
from urllib.parse import urlparse
from sklearn.model_selection import GroupShuffleSplit

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


def get_hostname(url):
    try:
        url = str(url).strip()

        if not url.startswith(("http://", "https://")):
            url = "http://" + url

        hostname = urlparse(url).hostname

        if not hostname:
            return ""

        return hostname.lower().strip(".")

    except Exception:
        return ""


def get_registrable_domain(hostname):
    """
    Approximate eTLD+1 handling for common multi-part
    public suffixes. This is used for the initial audit.
    """

    if not hostname:
        return ""

    if hostname.startswith("www."):
        hostname = hostname[4:]

    parts = hostname.split(".")

    if len(parts) < 2:
        return hostname

    # Common two-level public suffixes
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

    if last_two in common_two_level_suffixes and len(parts) >= 3:
        return ".".join(parts[-3:])

    return last_two


print("=" * 70)
print("COMPPHISH DOMAIN-SEPARATED SPLIT AUDIT")
print("=" * 70)


print("\nLoading Mapping_File.xlsx...")

mapping = pd.read_excel(MAPPING_FILE)

print("Mapping records:", len(mapping))


print("\nLoading CompPhish features...")

features = pd.read_csv(FEATURE_FILE)

print("Feature records:", len(features))


# Same filtering used by train_generalized.py
usable_mask = features["html_length"] > 0

features = features[
    usable_mask
].copy()

mapping = mapping[
    usable_mask.values
].copy()

print("Usable records:", len(features))


# Make sure row alignment is correct
if len(features) != len(mapping):
    raise ValueError(
        f"Alignment error: features={len(features)}, "
        f"mapping={len(mapping)}"
    )


# Extract hostname and registrable domain
mapping["hostname"] = mapping["url"].apply(
    get_hostname
)

mapping["registrable_domain"] = mapping[
    "hostname"
].apply(
    get_registrable_domain
)


# Remove records where domain extraction failed
valid = mapping["registrable_domain"] != ""

features = features.loc[
    valid.values
].reset_index(drop=True)

mapping = mapping.loc[
    valid.values
].reset_index(drop=True)


print(
    "Records with valid domains:",
    len(features)
)


# GroupShuffleSplit ensures the same registrable domain
# cannot appear in both train and test.
groups = mapping[
    "registrable_domain"
]


splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=SEED
)


train_idx, test_idx = next(
    splitter.split(
        features,
        features["label"],
        groups=groups
    )
)


train_domains = set(
    groups.iloc[train_idx]
)

test_domains = set(
    groups.iloc[test_idx]
)


overlap = train_domains.intersection(
    test_domains
)


print("\n" + "=" * 70)
print("DOMAIN-SPLIT RESULTS")
print("=" * 70)


print(
    "\nTraining samples:",
    len(train_idx)
)

print(
    "Test samples    :",
    len(test_idx)
)

print(
    "Training domains:",
    len(train_domains)
)

print(
    "Test domains    :",
    len(test_domains)
)

print(
    "Domain overlap  :",
    len(overlap)
)


print("\nTraining label distribution:")

print(
    features.iloc[train_idx]["label"]
    .value_counts()
    .sort_index()
)


print("\nTest label distribution:")

print(
    features.iloc[test_idx]["label"]
    .value_counts()
    .sort_index()
)


print("\n" + "=" * 70)


if len(overlap) == 0:

    print("✅ ZERO DOMAIN OVERLAP")

    print(
        "\nThe test domains are completely separated "
        "from the training domains."
    )

else:

    print("❌ DOMAIN OVERLAP DETECTED")

    print(
        "\nOverlapping domains:"
    )

    for domain in sorted(overlap)[:50]:
        print(domain)


print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)