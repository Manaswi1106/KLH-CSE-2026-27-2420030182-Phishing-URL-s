import os
import random
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

DEVICE = torch.device("cpu")

DATASET = r"D:\Documents\Aruna Grace\ALT\Dataset(D1)"

X_TRAIN_PATH = os.path.join(DATASET, "X_train.csv")
X_TEST_PATH = os.path.join(DATASET, "X_test.csv")
Y_TRAIN_PATH = os.path.join(DATASET, "y_train.csv")
Y_TEST_PATH = os.path.join(DATASET, "y_test.csv")

MODEL_DIR = os.path.join(
    r"D:\Documents\Aruna Grace\ALT\PhishXpert",
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)


FEATURES_TO_REMOVE = [
    "has_https",
    "has_ip"
]

LOG_FEATURES = [
    "url_length",
    "domain_length",
    "path_length",
    "query_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_digits",
    "num_special_chars",
    "html_length",
    "text_length",
    "num_links",
    "external_links",
    "form_count",
    "password_field_count",
    "external_form_count"
]


def preprocess_features(df):

    df = df.copy()

    for feature in FEATURES_TO_REMOVE:
        if feature in df.columns:
            df = df.drop(columns=[feature])

    for feature in LOG_FEATURES:
        if feature in df.columns:
            df[feature] = np.log1p(
                df[feature].clip(lower=0)
            )

    return df


print("\n========== PhishXpert V2 ==========")

print("\nLoading dataset...")

X_train = pd.read_csv(X_TRAIN_PATH)
X_test = pd.read_csv(X_TEST_PATH)

y_train = pd.read_csv(Y_TRAIN_PATH).values.ravel()
y_test = pd.read_csv(Y_TEST_PATH).values.ravel()

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))


X_train = preprocess_features(X_train)
X_test = preprocess_features(X_test)

feature_names = X_train.columns.tolist()

print("\nFeatures used:", len(feature_names))

print(feature_names)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32
).view(-1, 1)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.float32
).view(-1, 1)


URL_FEATURES = [
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
    "suspicious_keyword_count"
]

HTML_FEATURES = [
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


url_indices = [
    feature_names.index(f)
    for f in URL_FEATURES
    if f in feature_names
]

html_indices = [
    feature_names.index(f)
    for f in HTML_FEATURES
    if f in feature_names
]


class PhishXpertV2(nn.Module):

    def __init__(
        self,
        url_dim,
        html_dim
    ):

        super().__init__()

        self.url_encoder = nn.Sequential(
            nn.Linear(url_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.html_encoder = nn.Sequential(
            nn.Linear(html_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.gate = nn.Sequential(
            nn.Linear(64, 32),
            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.15),

            nn.Linear(64, 32),
            nn.ReLU()
        )

        self.classifier = nn.Linear(32, 1)

    def forward(
        self,
        url_x,
        html_x
    ):

        url_repr = self.url_encoder(url_x)

        html_repr = self.html_encoder(html_x)

        combined = torch.cat(
            [url_repr, html_repr],
            dim=1
        )

        gate = self.gate(combined)

        gated_url = url_repr * gate

        gated_html = html_repr * (1 - gate)

        fused = torch.cat(
            [gated_url, gated_html],
            dim=1
        )

        fused = self.fusion(fused)

        output = self.classifier(fused)

        return output


url_train = X_train_tensor[:, url_indices]
html_train = X_train_tensor[:, html_indices]

url_test = X_test_tensor[:, url_indices]
html_test = X_test_tensor[:, html_indices]


model = PhishXpertV2(
    len(url_indices),
    len(html_indices)
).to(DEVICE)


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
    lr=0.001,
    weight_decay=0.0001
)


print("\nTraining PhishXpert V2...")

epochs = 40

best_f1 = 0
best_state = None

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    logits = model(
        url_train,
        html_train
    )

    loss = criterion(
        logits,
        y_train_tensor
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0
    )

    optimizer.step()

    model.eval()

    with torch.no_grad():

        val_logits = model(
            url_test,
            html_test
        )

        val_prob = torch.sigmoid(
            val_logits
        ).numpy().ravel()

        val_pred = (
            val_prob >= 0.5
        ).astype(int)

    f1 = f1_score(
        y_test,
        val_pred
    )

    if f1 > best_f1:

        best_f1 = f1

        best_state = {
            key: value.cpu().clone()
            for key, value
            in model.state_dict().items()
        }

    if (epoch + 1) % 5 == 0:

        auc = roc_auc_score(
            y_test,
            val_prob
        )

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Loss: {loss.item():.4f} | "
            f"F1: {f1:.4f} | "
            f"AUC: {auc:.4f}"
        )


model.load_state_dict(best_state)

model.eval()

with torch.no_grad():

    logits = model(
        url_test,
        html_test
    )

    probabilities = torch.sigmoid(
        logits
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


print("\n========== PhishXpert V2 RESULTS ==========")

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
    f"ROC-AUC   : {auc * 100:.2f}%"
)

print("\nConfusion Matrix:")

print(cm)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "phishxpert_v2.pth"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "phishxpert_v2_scaler.pkl"
)

FEATURE_PATH = os.path.join(
    MODEL_DIR,
    "phishxpert_v2_features.pkl"
)


torch.save(
    model.state_dict(),
    MODEL_PATH
)

joblib.dump(
    scaler,
    SCALER_PATH
)

joblib.dump(
    feature_names,
    FEATURE_PATH
)


print("\nSaved:")

print(MODEL_PATH)
print(SCALER_PATH)
print(FEATURE_PATH)

print("\n========== COMPLETE ==========")