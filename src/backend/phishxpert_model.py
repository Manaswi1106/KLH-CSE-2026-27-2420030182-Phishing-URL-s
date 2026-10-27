import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATASET = r"D:\Documents\Aruna Grace\ALT\Dataset(D1)"
SAVE_DIR = r"D:\Documents\Aruna Grace\ALT\PhishXpert\models"
os.makedirs(SAVE_DIR, exist_ok=True)

X_train = pd.read_csv(os.path.join(DATASET, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATASET, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATASET, "y_train.csv")).values.ravel()
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

all_features = url_features + html_features

missing = [f for f in all_features if f not in X_train.columns]

if missing:
    raise ValueError(f"Missing features: {missing}")

X_train = X_train[all_features]
X_test = X_test[all_features]

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_url = X_train_scaled[:, :13]
X_train_html = X_train_scaled[:, 13:]

X_test_url = X_test_scaled[:, :13]
X_test_html = X_test_scaled[:, 13:]

X_train_url = torch.tensor(X_train_url, dtype=torch.float32)
X_train_html = torch.tensor(X_train_html, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

X_test_url = torch.tensor(X_test_url, dtype=torch.float32)
X_test_html = torch.tensor(X_test_html, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


class PhishXpert(nn.Module):

    def __init__(self):
        super().__init__()

        self.url_encoder = nn.Sequential(
            nn.Linear(13, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.html_encoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU()
        )

        self.classifier = nn.Linear(16, 1)

    def forward(self, url, html):
        url_representation = self.url_encoder(url)
        html_representation = self.html_encoder(html)

        combined = torch.cat(
            [url_representation, html_representation],
            dim=1
        )

        evidence = self.fusion(combined)

        output = self.classifier(evidence)

        return output


model = PhishXpert()

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

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 30

print("\nTraining PhishXpert...\n")

for epoch in range(epochs):

    model.train()

    total_loss = 0

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

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1:02d}/{epochs} "
        f"- Loss: {avg_loss:.4f}"
    )


model.eval()

with torch.no_grad():

    logits = model(
        X_test_url,
        X_test_html
    ).squeeze(1)

    probabilities = torch.sigmoid(logits).numpy()

predictions = (probabilities >= 0.5).astype(int)

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)

cm = confusion_matrix(
    y_test,
    predictions
)

print("\n========== PhishXpert Results ==========")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")
print(f"ROC-AUC   : {roc_auc * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

torch.save(
    model.state_dict(),
    os.path.join(SAVE_DIR, "phishxpert_model.pth")
)

import joblib

joblib.dump(
    scaler,
    os.path.join(SAVE_DIR, "phishxpert_scaler.pkl")
)

print("\nModel saved successfully.")