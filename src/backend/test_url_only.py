import math
import torch
import joblib
import pandas as pd

from model_predictor import PhishXpert


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


URL_FEATURES = FEATURES[:13]

HTML_FEATURES = FEATURES[13:]


def entropy(value):

    if not value:
        return 0.0

    probabilities = [
        value.count(char) / len(value)
        for char in set(value)
    ]

    return -sum(
        p * math.log2(p)
        for p in probabilities
        if p > 0
    )


def make_url_features(url):

    from urllib.parse import urlparse

    original_url = url

    if not url.startswith(
        ("http://", "https://")
    ):
        url = "https://" + url

    parsed = urlparse(url)

    domain = parsed.netloc
    path = parsed.path
    query = parsed.query

    url_lower = original_url.lower()

    suspicious_words = [
        "login",
        "signin",
        "verify",
        "verification",
        "account",
        "secure",
        "security",
        "update",
        "confirm",
        "password",
        "credential",
        "bank",
        "paypal",
        "payment",
        "wallet",
        "recover",
        "unlock",
        "authentication"
    ]

    return {
        "url_length": len(original_url),

        "domain_length": len(domain),

        "path_length": len(path),

        "query_length": len(query),

        "num_dots": original_url.count("."),

        "num_hyphens": original_url.count("-"),

        "num_underscores": original_url.count("_"),

        "num_digits": sum(
            c.isdigit()
            for c in original_url
        ),

        "num_special_chars": sum(
            not c.isalnum()
            for c in original_url
        ),

        "url_entropy": entropy(
            original_url
        ),

        "has_https": int(
            parsed.scheme == "https"
        ),

        "has_ip": int(
            domain.replace(".", "").isdigit()
        ),

        "suspicious_keyword_count": sum(
            word in url_lower
            for word in suspicious_words
        )
    }


def predict_url_only(url):

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

    url_features = make_url_features(url)

    features = {}

    for feature in URL_FEATURES:
        features[feature] = (
            url_features[feature]
        )

    for feature in HTML_FEATURES:
        features[feature] = 0

    data = pd.DataFrame(
        [features]
    )[FEATURES]

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

        output = model(
            url_x,
            html_x
        )

        probability = torch.sigmoid(
            output
        ).item()

    if probability >= 0.5:

        prediction = "Phishing"

        confidence = probability

    else:

        prediction = "Benign"

        confidence = 1 - probability

    return prediction, confidence


sites = [
    ("Google", "https://www.google.com/"),
    ("YouTube", "https://www.youtube.com/"),
    ("GitHub", "https://github.com/"),
    ("Wikipedia", "https://www.wikipedia.org/"),
    ("CNN", "https://www.cnn.com/")
]


print(
    "\n========== URL-ONLY DIAGNOSTIC ==========\n"
)


for name, url in sites:

    prediction, confidence = (
        predict_url_only(url)
    )

    print(
        f"{name:12} → "
        f"{prediction:8} "
        f"{confidence * 100:.2f}%"
    )


print(
    "\n========== COMPLETE ==========\n"
)