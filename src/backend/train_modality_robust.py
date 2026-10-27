import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
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

torch.manual_seed(42)
np.random.seed(42)

train = pd.read_csv(DATASET / "X_train.csv")
test = pd.read_csv(DATASET / "X_test.csv")

y_train = pd.read_csv(
    DATASET / "y_train.csv"
).values.ravel()

y_test = pd.read_csv(
    DATASET / "y_test.csv"
).values.ravel()

X_train = train[FEATURES].copy()
X_test = test[FEATURES].copy()

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


class PhishXpert(nn.Module):

    def __init__(self):
        super().__init__()

        self.url_encoder = nn.Sequential(
            nn.Linear(13, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(48, 24),
            nn.ReLU()
        )

        self.html_encoder = nn.Sequential(
            nn.Linear(10, 48),
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


model = PhishXpert()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.0001
)

criterion = nn.BCEWithLogitsLoss()

url_train = torch.tensor(
    X_train_scaled[:, :13],
    dtype=torch.float32
)

html_train = torch.tensor(
    X_train_scaled[:, 13:],
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32
).reshape(-1, 1)

url_test = torch.tensor(
    X_test_scaled[:, :13],
    dtype=torch.float32
)

html_test = torch.tensor(
    X_test_scaled[:, 13:],
    dtype=torch.float32
)

print("=" * 70)
print("PHISHXPERT - MODALITY ROBUST TRAINING")
print("=" * 70)

for epoch in range(40):

    model.train()

    optimizer.zero_grad()

    # Clone HTML features
    html_input = html_train.clone()

    # Randomly hide HTML for 30% of training samples
    mask = torch.rand(
        html_input.shape[0]
    ) < 0.30

    html_input[mask] = 0.0

    output = model(
        url_train,
        html_input
    )

    loss = criterion(
        output,
        y_train_tensor
    )

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 5 == 0:

        model.eval()

        with torch.no_grad():

            output = model(
                url_test,
                html_test
            )

            probabilities = torch.sigmoid(
                output
            ).numpy().ravel()

        predictions = (
            probabilities >= 0.5
        ).astype(int)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        f1 = f1_score(
            y_test,
            predictions
        )

        auc = roc_auc_score(
            y_test,
            probabilities
        )

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Loss {loss.item():.4f} | "
            f"Accuracy {accuracy * 100:.2f}% | "
            f"F1 {f1 * 100:.2f}% | "
            f"AUC {auc * 100:.2f}%"
        )


model.eval()

with torch.no_grad():

    output = model(
        url_test,
        html_test
    )

    probabilities = torch.sigmoid(
        output
    ).numpy().ravel()

predictions = (
    probabilities >= 0.5
).astype(int)

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

auc = roc_auc_score(
    y_test,
    probabilities
)

cm = confusion_matrix(
    y_test,
    predictions
)

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")
print(f"ROC-AUC  : {auc * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)

Path("models").mkdir(exist_ok=True)

torch.save(
    model.state_dict(),
    "models/phishxpert_modality_robust.pth"
)

joblib.dump(
    scaler,
    "models/phishxpert_modality_robust_scaler.pkl"
)

joblib.dump(
    FEATURES,
    "models/phishxpert_modality_robust_features.pkl"
)

print("\nSaved:")
print("models/phishxpert_modality_robust.pth")
print("models/phishxpert_modality_robust_scaler.pkl")
print("models/phishxpert_modality_robust_features.pkl")

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)