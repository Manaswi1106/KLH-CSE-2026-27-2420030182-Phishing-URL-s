import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

import joblib


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


BASE = r"D:\Documents\Aruna Grace\ALT"

D1_DIR = os.path.join(
    BASE,
    "Dataset(D1)"
)

COMPPHISH_FILE = os.path.join(
    BASE,
    "CompPhish",
    "CompPhish",
    "CompPhish_features.csv"
)

MODEL_DIR = os.path.join(
    BASE,
    "PhishXpert",
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


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


class PhishXpert(nn.Module):

    def __init__(
        self,
        url_features=13,
        html_features=10
    ):

        super().__init__()

        self.url_encoder = nn.Sequential(
            nn.Linear(url_features, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(48, 24),
            nn.ReLU()
        )

        self.html_encoder = nn.Sequential(
            nn.Linear(html_features, 48),
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

        self.classifier = nn.Linear(
            24,
            1
        )


    def forward(
        self,
        url_x,
        html_x
    ):

        url_out = self.url_encoder(
            url_x
        )

        html_out = self.html_encoder(
            html_x
        )

        combined = torch.cat(
            (
                url_out,
                html_out
            ),
            dim=1
        )

        fused = self.fusion(
            combined
        )

        return self.classifier(
            fused
        )


def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=50
):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_val_scaled = scaler.transform(
        X_val
    )


    url_train = torch.tensor(
        X_train_scaled[:, :13],
        dtype=torch.float32
    )

    html_train = torch.tensor(
        X_train_scaled[:, 13:],
        dtype=torch.float32
    )

    y_train_tensor = torch.tensor(
        y_train.values,
        dtype=torch.float32
    ).view(-1, 1)


    url_val = torch.tensor(
        X_val_scaled[:, :13],
        dtype=torch.float32
    )

    html_val = torch.tensor(
        X_val_scaled[:, 13:],
        dtype=torch.float32
    )

    y_val_tensor = torch.tensor(
        y_val.values,
        dtype=torch.float32
    ).view(-1, 1)


    model = PhishXpert()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )


    criterion = nn.BCEWithLogitsLoss()


    best_f1 = -1

    best_state = None

    patience = 8

    patience_counter = 0


    print("\nTraining generalized PhishXpert...")

    for epoch in range(epochs):

        model.train()

        optimizer.zero_grad()


        outputs = model(
            url_train,
            html_train
        )


        loss = criterion(
            outputs,
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

            val_outputs = model(
                url_val,
                html_val
            )

            val_probs = torch.sigmoid(
                val_outputs
            ).numpy().flatten()


        val_preds = (
            val_probs >= 0.5
        ).astype(int)


        val_f1 = f1_score(
            y_val,
            val_preds,
            zero_division=0
        )


        if (
            epoch + 1
        ) % 5 == 0:

            val_acc = accuracy_score(
                y_val,
                val_preds
            )

            print(
                f"Epoch {epoch + 1:02d} | "
                f"Loss {loss.item():.4f} | "
                f"Val Acc {val_acc * 100:.2f}% | "
                f"Val F1 {val_f1 * 100:.2f}%"
            )


        if val_f1 > best_f1:

            best_f1 = val_f1

            best_state = {
                key: value.cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            patience_counter = 0

        else:

            patience_counter += 1


        if patience_counter >= patience:

            print(
                "Early stopping."
            )

            break


    model.load_state_dict(
        best_state
    )

    return model, scaler


def evaluate(
    model,
    scaler,
    X,
    y,
    name
):

    X_scaled = scaler.transform(
        X
    )


    url_x = torch.tensor(
        X_scaled[:, :13],
        dtype=torch.float32
    )

    html_x = torch.tensor(
        X_scaled[:, 13:],
        dtype=torch.float32
    )


    model.eval()

    with torch.no_grad():

        outputs = model(
            url_x,
            html_x
        )

        probs = torch.sigmoid(
            outputs
        ).numpy().flatten()


    preds = (
        probs >= 0.5
    ).astype(int)


    accuracy = accuracy_score(
        y,
        preds
    )

    precision = precision_score(
        y,
        preds,
        zero_division=0
    )

    recall = recall_score(
        y,
        preds,
        zero_division=0
    )

    f1 = f1_score(
        y,
        preds,
        zero_division=0
    )

    auc = roc_auc_score(
        y,
        probs
    )

    pr_auc = average_precision_score(
        y,
        probs
    )

    cm = confusion_matrix(
        y,
        preds
    )


    print("\n" + "=" * 60)

    print(
        name
    )

    print("=" * 60)

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

    print("\nConfusion Matrix:")

    print(cm)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
        "pr_auc": pr_auc
    }


print("=" * 60)
print("PHISHXPERT GENERALIZATION TRAINING")
print("=" * 60)


print("\nLoading D1...")

X_d1_train = pd.read_csv(
    os.path.join(
        D1_DIR,
        "X_train.csv"
    )
)

y_d1_train = pd.read_csv(
    os.path.join(
        D1_DIR,
        "y_train.csv"
    )
).iloc[:, 0]

X_d1_test = pd.read_csv(
    os.path.join(
        D1_DIR,
        "X_test.csv"
    )
)

y_d1_test = pd.read_csv(
    os.path.join(
        D1_DIR,
        "y_test.csv"
    )
).iloc[:, 0]


print(
    "D1 train:",
    len(X_d1_train)
)

print(
    "D1 test:",
    len(X_d1_test)
)


print("\nLoading CompPhish...")

comp = pd.read_csv(
    COMPPHISH_FILE
)


# Keep only records with usable HTML
comp = comp[
    comp["html_length"] > 0
].copy()


print(
    "CompPhish usable:",
    len(comp)
)


X_comp = comp[
    FEATURES
]

y_comp = comp[
    "label"
]


X_comp_train, X_comp_test, y_comp_train, y_comp_test = train_test_split(
    X_comp,
    y_comp,
    test_size=0.20,
    random_state=SEED,
    stratify=y_comp
)


print(
    "CompPhish train:",
    len(X_comp_train)
)

print(
    "CompPhish unseen test:",
    len(X_comp_test)
)


# Combine training datasets only
X_combined = pd.concat(
    [
        X_d1_train,
        X_comp_train
    ],
    ignore_index=True
)

y_combined = pd.concat(
    [
        y_d1_train,
        y_comp_train
    ],
    ignore_index=True
)


print("\nCombined training set:")

print(
    len(X_combined)
)

print(
    y_combined.value_counts()
)


# Create validation split from combined training only
X_train, X_val, y_train, y_val = train_test_split(
    X_combined,
    y_combined,
    test_size=0.15,
    random_state=SEED,
    stratify=y_combined
)


model, scaler = train_model(
    X_train,
    y_train,
    X_val,
    y_val
)


print("\nEvaluating generalized model...")


d1_results = evaluate(
    model,
    scaler,
    X_d1_test,
    y_d1_test,
    "D1 TEST"
)


comp_results = evaluate(
    model,
    scaler,
    X_comp_test,
    y_comp_test,
    "COMPPHISH UNSEEN TEST"
)


MODEL_PATH = os.path.join(
    MODEL_DIR,
    "phishxpert_generalized.pth"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "phishxpert_generalized_scaler.pkl"
)


torch.save(
    model.state_dict(),
    MODEL_PATH
)

joblib.dump(
    scaler,
    SCALER_PATH
)


print("\n" + "=" * 60)
print("GENERALIZED MODEL SAVED")
print("=" * 60)

print(
    "Model:",
    MODEL_PATH
)

print(
    "Scaler:",
    SCALER_PATH
)


print("\nFINAL COMPARISON")

print(
    f"Original CompPhish accuracy : 79.67%"
)

print(
    f"Generalized CompPhish       : "
    f"{comp_results['accuracy'] * 100:.2f}%"
)

print(
    f"Original D1 accuracy        : 85.68%"
)

print(
    f"Generalized D1              : "
    f"{d1_results['accuracy'] * 100:.2f}%"
)

print("=" * 60)