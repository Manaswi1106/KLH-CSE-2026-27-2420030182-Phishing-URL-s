import torch
import torch.nn as nn
import pandas as pd
import joblib


class PhishXpert(nn.Module):
    def __init__(self, url_features=13, html_features=10):
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


def load_model():

    model = PhishXpert()

    model.load_state_dict(
        torch.load(
            "models/phishxpert_optimized.pth",
            map_location="cpu"
        )
    )

    model.eval()

    scaler = joblib.load(
    "models/phishxpert_scaler.pkl"
)

    return model, scaler


def predict(features):

    model, scaler = load_model()

    data = pd.DataFrame([features])[FEATURES]

    scaled = scaler.transform(data)

    url_x = torch.tensor(
        scaled[:, :13],
        dtype=torch.float32
    )

    html_x = torch.tensor(
        scaled[:, 13:],
        dtype=torch.float32
    )

    with torch.no_grad():

        output = model(url_x, html_x)

        probability = torch.sigmoid(output).item()


    if probability >= 0.5:

        prediction = "Phishing"

        confidence = probability

    else:

        prediction = "Benign"

        confidence = 1 - probability


    return prediction, confidence


if __name__ == "__main__":

    sample_features = {

        "url_length": 60,
        "domain_length": 25,
        "path_length": 20,
        "query_length": 10,
        "num_dots": 3,
        "num_hyphens": 4,
        "num_underscores": 1,
        "num_digits": 5,
        "num_special_chars": 10,
        "url_entropy": 4.2,
        "has_https": 1,
        "has_ip": 0,
        "suspicious_keyword_count": 3,

        "html_length": 50000,
        "text_length": 5000,
        "num_links": 100,
        "external_links": 40,
        "external_link_ratio": 0.4,
        "form_count": 2,
        "password_field_count": 1,
        "has_login_form": 1,
        "external_form_count": 1,
        "external_form_ratio": 0.5
    }


    prediction, confidence = predict(
        sample_features
    )


    print(
        "========== PHISHXPERT MODEL =========="
    )

    print(
        "Prediction:",
        prediction
    )

    print(
        "Confidence:",
        f"{confidence * 100:.2f}%"
    )