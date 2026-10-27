import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATASET = r"D:\Documents\Aruna Grace\ALT\Dataset(D1)"
MODEL_DIR = r"D:\Documents\Aruna Grace\ALT\PhishXpert\models"
os.makedirs(MODEL_DIR, exist_ok=True)

X_train_full = pd.read_csv(os.path.join(DATASET, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATASET, "X_test.csv"))
y_train_full = pd.read_csv(os.path.join(DATASET, "y_train.csv")).values.ravel()
y_test = pd.read_csv(os.path.join(DATASET, "y_test.csv")).values.ravel()

url_features = [
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
    "suspicious_keyword_count"
]

html_features = [
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

features = url_features + html_features

X_train_full = X_train_full[features]
X_test = X_test[features]

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.15,
    random_state=SEED,
    stratify=y_train_full
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

X_train_url = torch.tensor(X_train[:, :13], dtype=torch.float32)
X_train_html = torch.tensor(X_train[:, 13:], dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_val_url = torch.tensor(X_val[:, :13], dtype=torch.float32)
X_val_html = torch.tensor(X_val[:, 13:], dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

X_test_url = torch.tensor(X_test_scaled[:, :13], dtype=torch.float32)
X_test_html = torch.tensor(X_test_scaled[:, 13:], dtype=torch.float32)


class PhishXpert(nn.Module):

    def __init__(self, hidden=48, latent=24, dropout=0.15):
        super().__init__()

        self.url_encoder = nn.Sequential(
            nn.Linear(13, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent),
            nn.ReLU()
        )

        self.html_encoder = nn.Sequential(
            nn.Linear(10, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(latent * 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, latent),
            nn.ReLU()
        )

        self.classifier = nn.Linear(latent, 1)

    def forward(self, url, html):

        url_rep = self.url_encoder(url)
        html_rep = self.html_encoder(html)

        combined = torch.cat(
            [url_rep, html_rep],
            dim=1
        )

        evidence = self.fusion(combined)

        return self.classifier(evidence)


train_dataset = TensorDataset(
    X_train_url,
    X_train_html,
    y_train_tensor
)

train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True
)


def train_model(lr, weight_decay, dropout):

    model = PhishXpert(dropout=dropout)

    positive = np.sum(y_train == 1)
    negative = np.sum(y_train == 0)

    pos_weight = torch.tensor(
        [negative / positive],
        dtype=torch.float32
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    best_f1 = 0
    best_state = None
    patience = 6
    patience_count = 0

    for epoch in range(40):

        model.train()

        for url_batch, html_batch, labels in train_loader:

            optimizer.zero_grad()

            logits = model(
                url_batch,
                html_batch
            ).squeeze(1)

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

        model.eval()

        with torch.no_grad():

            val_logits = model(
                X_val_url,
                X_val_html
            ).squeeze(1)

            val_prob = torch.sigmoid(
                val_logits
            ).numpy()

        val_pred = (val_prob >= 0.5).astype(int)

        val_f1 = f1_score(
            y_val,
            val_pred
        )

        if val_f1 > best_f1:

            best_f1 = val_f1
            best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

            patience_count = 0

        else:

            patience_count += 1

            if patience_count >= patience:
                break

    model.load_state_dict(best_state)

    model.eval()

    with torch.no_grad():

        val_logits = model(
            X_val_url,
            X_val_html
        ).squeeze(1)

        val_prob = torch.sigmoid(
            val_logits
        ).numpy()

    val_pred = (val_prob >= 0.5).astype(int)

    return model, val_prob, val_pred, best_f1


configs = [
    (0.001, 0.0001, 0.10),
    (0.0005, 0.0001, 0.10),
    (0.001, 0.001, 0.15),
    (0.0005, 0.001, 0.20),
    (0.0003, 0.0001, 0.20)
]

best_model = None
best_config = None
best_score = -1

print("\n========== PhishXpert Optimization ==========\n")

for i, (lr, wd, dropout) in enumerate(configs, 1):

    print(
        f"Configuration {i}/{len(configs)} | "
        f"LR={lr} WD={wd} Dropout={dropout}"
    )

    model, val_prob, val_pred, val_f1 = train_model(
        lr,
        wd,
        dropout
    )

    val_recall = recall_score(
        y_val,
        val_pred
    )

    val_auc = roc_auc_score(
        y_val,
        val_prob
    )

    print(
        f"Validation F1: {val_f1 * 100:.2f}% | "
        f"Recall: {val_recall * 100:.2f}% | "
        f"ROC-AUC: {val_auc * 100:.2f}%"
    )

    score = val_f1 + 0.25 * val_recall

    if score > best_score:

        best_score = score
        best_model = model
        best_config = (lr, wd, dropout)


print("\nBest configuration:")
print(
    f"Learning rate: {best_config[0]}"
)
print(
    f"Weight decay: {best_config[1]}"
)
print(
    f"Dropout: {best_config[2]}"
)


best_model.eval()

with torch.no_grad():

    test_logits = best_model(
        X_test_url,
        X_test_html
    ).squeeze(1)

    test_prob = torch.sigmoid(
        test_logits
    ).numpy()

test_pred = (test_prob >= 0.5).astype(int)

accuracy = accuracy_score(
    y_test,
    test_pred
)

precision = precision_score(
    y_test,
    test_pred
)

recall = recall_score(
    y_test,
    test_pred
)

f1 = f1_score(
    y_test,
    test_pred
)

auc = roc_auc_score(
    y_test,
    test_prob
)

cm = confusion_matrix(
    y_test,
    test_pred
)

print("\n========== Optimized PhishXpert ==========")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")
print(f"ROC-AUC   : {auc * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

torch.save(
    best_model.state_dict(),
    os.path.join(
        MODEL_DIR,
        "phishxpert_optimized.pth"
    )
)

import joblib

joblib.dump(
    scaler,
    os.path.join(
        MODEL_DIR,
        "phishxpert_scaler.pkl"
    )
)

print("\nOptimized model saved successfully.")