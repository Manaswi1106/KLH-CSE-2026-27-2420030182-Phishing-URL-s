import os
import random
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")

BASE = r"D:\Documents\Aruna Grace\ALT"

D1 = os.path.join(
    BASE,
    "Dataset(D1)"
)

COMP = os.path.join(
    BASE,
    "CompPhish",
    "CompPhish"
)

MODEL_DIR = os.path.join(
    BASE,
    "PhishXpert",
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)


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
    "has_https",
    "has_ip",
    "suspicious_keyword_count"
]


def load_data():

    print("Loading D1...")

    X_train = pd.read_csv(
        os.path.join(D1, "X_train.csv")
    )

    X_test = pd.read_csv(
        os.path.join(D1, "X_test.csv")
    )

    y_train = pd.read_csv(
        os.path.join(D1, "y_train.csv")
    ).values.ravel()

    y_test = pd.read_csv(
        os.path.join(D1, "y_test.csv")
    ).values.ravel()

    print(
        "D1 train:",
        X_train.shape
    )

    print(
        "D1 test:",
        X_test.shape
    )

    print("\nLoading CompPhish...")

    comp = pd.read_csv(
        os.path.join(
            COMP,
            "CompPhish_features.csv"
        )
    )

    comp = comp[
        comp["html_length"] > 0
    ].copy()

    comp_train, comp_test = train_test_split(
        comp,
        test_size=0.20,
        random_state=SEED,
        stratify=comp["label"]
    )

    X_train = X_train[URL_FEATURES]

    X_test = X_test[URL_FEATURES]

    comp_train_X = comp_train[URL_FEATURES]
    comp_test_X = comp_test[URL_FEATURES]

    X_combined = pd.concat(
        [
            X_train,
            comp_train_X
        ],
        ignore_index=True
    )

    y_combined = np.concatenate(
        [
            y_train,
            comp_train["label"].values
        ]
    )

    return (
        X_combined,
        y_combined,
        X_test,
        y_test,
        comp_test_X,
        comp_test["label"].values
    )


class URLOnlyModel(nn.Module):

    def __init__(self, input_dim):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_dim,
                64
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.Dropout(0.15),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Dropout(0.10),

            nn.Linear(
                32,
                16
            ),

            nn.ReLU(),

            nn.Linear(
                16,
                1
            )
        )

    def forward(self, x):

        return self.network(x)


print(
    "\n========== PhishXpert URL-Only =========="
)

(
    X_train,
    y_train,
    X_d1_test,
    y_d1_test,
    X_comp_test,
    y_comp_test
) = load_data()


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_d1_scaled = scaler.transform(
    X_d1_test
)

X_comp_scaled = scaler.transform(
    X_comp_test
)


X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32
).view(-1, 1)

X_d1_tensor = torch.tensor(
    X_d1_scaled,
    dtype=torch.float32
)

X_comp_tensor = torch.tensor(
    X_comp_scaled,
    dtype=torch.float32
)


model = URLOnlyModel(
    len(URL_FEATURES)
).to(DEVICE)


positive = np.sum(
    y_train == 1
)

negative = np.sum(
    y_train == 0
)

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


print(
    "\nTraining URL-only model..."
)

epochs = 50

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()

    logits = model(
        X_train_tensor
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

    if (epoch + 1) % 5 == 0:

        model.eval()

        with torch.no_grad():

            probs = torch.sigmoid(
                model(X_train_tensor)
            ).numpy().ravel()

        auc = roc_auc_score(
            y_train,
            probs
        )

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"Loss: {loss.item():.4f} | "
            f"Train AUC: {auc:.4f}"
        )


def evaluate(
    name,
    X,
    y
):

    model.eval()

    with torch.no_grad():

        probabilities = torch.sigmoid(
            model(X)
        ).numpy().ravel()

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    accuracy = accuracy_score(
        y,
        predictions
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0
    )

    auc = roc_auc_score(
        y,
        probabilities
    )

    pr_auc = average_precision_score(
        y,
        probabilities
    )

    cm = confusion_matrix(
        y,
        predictions
    )

    print(
        f"\n========== {name} =========="
    )

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

    print(
        f"PR-AUC    : {pr_auc * 100:.2f}%"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(cm)


evaluate(
    "D1 TEST",
    X_d1_tensor,
    y_d1_test
)

evaluate(
    "CompPhish RANDOM HOLDOUT",
    X_comp_tensor,
    y_comp_test
)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "phishxpert_url_only.pth"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "phishxpert_url_only_scaler.pkl"
)


torch.save(
    model.state_dict(),
    MODEL_PATH
)

joblib.dump(
    scaler,
    SCALER_PATH
)


print(
    "\nSaved URL-only model:"
)

print(MODEL_PATH)

print(SCALER_PATH)

print(
    "\n========== COMPLETE =========="
)