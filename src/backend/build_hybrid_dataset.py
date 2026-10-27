import os
import pandas as pd
import numpy as np

from app import extract_features
from bert_encoder import get_bert_embedding
from text_preprocessing import clean_webpage_text
from hybrid_fusion import fuse_features

DATASET = r"D:\Documents\Aruna Grace\ALT\Dataset(D1)"
HTML_DIR = os.path.join(DATASET, "sourc_Data(D1)")

df = pd.read_csv(os.path.join(DATASET, "ID__URLs_labell.csv"))

# Convert labels
df["label"] = df["typ"].map({-1:0, 1:1})

rows = []

SAMPLE_SIZE = 5000

# 50% phishing, 50% benign
phishing = df[df["label"] == 1].sample(
    SAMPLE_SIZE // 2,
    random_state=42
)

benign = df[df["label"] == 0].sample(
    SAMPLE_SIZE // 2,
    random_state=42
)

df = pd.concat([phishing, benign]).sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

for i, row in df.iterrows():

    html_path = os.path.join(HTML_DIR, f"{int(row['id'])}.txt")

    if not os.path.exists(html_path):
        continue

    try:
        with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
            html = f.read()

        # Existing 23 features
        feature_dict = extract_features(row["url"], html)
        engineered = list(feature_dict.values())

        # Clean webpage text
        text = clean_webpage_text(html)

        # BERT embedding
        bert = get_bert_embedding(text)

        # Hybrid fusion
        fused = fuse_features(engineered, bert)

        rows.append(
            np.append(fused, row["label"])
        )

        if i % 500 == 0:
            print(i)

    except Exception:
        continue

cols = [f"f{i}" for i in range(791)] + ["label"]

hybrid_df = pd.DataFrame(rows, columns=cols)

os.makedirs("data", exist_ok=True)

hybrid_df.to_csv("data/hybrid_dataset.csv", index=False)

print(hybrid_df.shape)
print("Saved successfully.")