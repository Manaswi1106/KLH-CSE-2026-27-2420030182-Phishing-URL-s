import pandas as pd
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
from pathlib import Path

DATASET = Path(r"D:\Documents\Aruna Grace\ALT\Dataset(D1)")

df = pd.read_csv(DATASET / "ID__URLs_labell.csv")

df["label"] = df["typ"].map({-1: 0, 1: 1})

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

def get_domain(url):
    try:
        if not url.startswith(("http://", "https://")):
            url = "http://" + url

        domain = urlparse(url).netloc.lower()

        if domain.startswith("www."):
            domain = domain[4:]

        return domain
    except:
        return ""

train_domains = set(
    get_domain(url)
    for url in train_df["url"]
    if url
)

test_domains = set(
    get_domain(url)
    for url in test_df["url"]
    if url
)

overlap = train_domains.intersection(test_domains)

print("=" * 70)
print("D1 DOMAIN OVERLAP CHECK")
print("=" * 70)

print(f"\nTotal samples       : {len(df)}")
print(f"Training samples    : {len(train_df)}")
print(f"Test samples        : {len(test_df)}")

print(f"\nUnique train domains: {len(train_domains)}")
print(f"Unique test domains : {len(test_domains)}")
print(f"Overlapping domains : {len(overlap)}")

if len(test_domains) > 0:
    percentage = len(overlap) / len(test_domains) * 100
    print(f"Test domains also in training: {percentage:.2f}%")

print("\n" + "=" * 70)

if overlap:
    print("OVERLAPPING DOMAINS")
    print("=" * 70)

    for domain in sorted(overlap):
        print(domain)
else:
    print("No overlapping domains found.")

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)