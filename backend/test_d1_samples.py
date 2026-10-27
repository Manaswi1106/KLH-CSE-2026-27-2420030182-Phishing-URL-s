import os
import random
import math
import re
import pandas as pd
from bs4 import BeautifulSoup

from model_predictor import predict

DATASET = r"D:\Documents\Aruna Grace\ALT\Dataset(D1)"
HTML_DIR = os.path.join(DATASET, "sourc_Data(D1)")


FEATURE_NAMES = [
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


SUSPICIOUS_WORDS = [
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


def extract_features(url, html=""):

    from urllib.parse import urlparse

    original_url = url

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)

    domain = parsed.netloc
    path = parsed.path
    query = parsed.query

    url_lower = original_url.lower()

    url_length = len(original_url)
    domain_length = len(domain)
    path_length = len(path)
    query_length = len(query)

    num_dots = original_url.count(".")
    num_hyphens = original_url.count("-")
    num_underscores = original_url.count("_")

    num_digits = sum(
        c.isdigit()
        for c in original_url
    )

    num_special_chars = sum(
        not c.isalnum()
        for c in original_url
    )

    url_entropy = entropy(original_url)

    has_https = int(
        parsed.scheme == "https"
    )

    has_ip = int(
        domain.replace(".", "").isdigit()
    )

    suspicious_keyword_count = sum(
        word in url_lower
        for word in SUSPICIOUS_WORDS
    )

    html_length = len(html)

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    text = soup.get_text(
        " ",
        strip=True
    )

    text_length = len(text)

    links = soup.find_all(
        "a",
        href=True
    )

    num_links = len(links)

    external_links = 0

    for link in links:

        href = link.get("href", "")

        try:

            link_domain = urlparse(
                href
            ).netloc

            if (
                link_domain
                and link_domain != domain
            ):
                external_links += 1

        except Exception:
            pass

    external_link_ratio = (
        external_links / num_links
        if num_links > 0
        else 0
    )

    forms = soup.find_all("form")

    form_count = len(forms)

    password_field_count = len(
        soup.find_all(
            "input",
            {"type": "password"}
        )
    )

    has_login_form = int(
        password_field_count > 0
        or any(
            "login" in str(form).lower()
            or "signin" in str(form).lower()
            for form in forms
        )
    )

    external_form_count = 0

    for form in forms:

        action = form.get(
            "action",
            ""
        )

        if action:

            try:

                action_domain = urlparse(
                    action
                ).netloc

                if (
                    action_domain
                    and action_domain != domain
                ):
                    external_form_count += 1

            except Exception:
                pass

    external_form_ratio = (
        external_form_count / form_count
        if form_count > 0
        else 0
    )

    values = [
        url_length,
        domain_length,
        path_length,
        query_length,
        num_dots,
        num_hyphens,
        num_underscores,
        num_digits,
        num_special_chars,
        url_entropy,
        has_https,
        has_ip,
        suspicious_keyword_count,
        html_length,
        text_length,
        num_links,
        external_links,
        external_link_ratio,
        form_count,
        password_field_count,
        has_login_form,
        external_form_count,
        external_form_ratio
    ]

    return dict(
        zip(
            FEATURE_NAMES,
            values
        )
    )


df = pd.read_csv(
    os.path.join(
        DATASET,
        "ID__URLs_labell.csv"
    )
)

df["label"] = df["typ"].map({
    -1: 0,
    1: 1
})


available_ids = set()

for filename in os.listdir(HTML_DIR):

    if filename.endswith(".txt"):

        try:
            available_ids.add(
                int(filename[:-4])
            )

        except Exception:
            pass


df = df[
    df["id"].isin(available_ids)
]


phishing = df[
    df["label"] == 1
]

benign = df[
    df["label"] == 0
]


random.seed(42)

import os
import random
import math
import re
import pandas as pd
from bs4 import BeautifulSoup

from model_predictor import predict

DATASET = r"D:\Documents\Aruna Grace\ALT\Dataset(D1)"
HTML_DIR = os.path.join(DATASET, "sourc_Data(D1)")


FEATURE_NAMES = [
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


SUSPICIOUS_WORDS = [
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


def extract_features(url, html=""):

    from urllib.parse import urlparse

    original_url = url

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)

    domain = parsed.netloc
    path = parsed.path
    query = parsed.query

    url_lower = original_url.lower()

    url_length = len(original_url)
    domain_length = len(domain)
    path_length = len(path)
    query_length = len(query)

    num_dots = original_url.count(".")
    num_hyphens = original_url.count("-")
    num_underscores = original_url.count("_")

    num_digits = sum(
        c.isdigit()
        for c in original_url
    )

    num_special_chars = sum(
        not c.isalnum()
        for c in original_url
    )

    url_entropy = entropy(original_url)

    has_https = int(
        parsed.scheme == "https"
    )

    has_ip = int(
        domain.replace(".", "").isdigit()
    )

    suspicious_keyword_count = sum(
        word in url_lower
        for word in SUSPICIOUS_WORDS
    )

    html_length = len(html)

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    text = soup.get_text(
        " ",
        strip=True
    )

    text_length = len(text)

    links = soup.find_all(
        "a",
        href=True
    )

    num_links = len(links)

    external_links = 0

    for link in links:

        href = link.get("href", "")

        try:

            link_domain = urlparse(
                href
            ).netloc

            if (
                link_domain
                and link_domain != domain
            ):
                external_links += 1

        except Exception:
            pass

    external_link_ratio = (
        external_links / num_links
        if num_links > 0
        else 0
    )

    forms = soup.find_all("form")

    form_count = len(forms)

    password_field_count = len(
        soup.find_all(
            "input",
            {"type": "password"}
        )
    )

    has_login_form = int(
        password_field_count > 0
        or any(
            "login" in str(form).lower()
            or "signin" in str(form).lower()
            for form in forms
        )
    )

    external_form_count = 0

    for form in forms:

        action = form.get(
            "action",
            ""
        )

        if action:

            try:

                action_domain = urlparse(
                    action
                ).netloc

                if (
                    action_domain
                    and action_domain != domain
                ):
                    external_form_count += 1

            except Exception:
                pass

    external_form_ratio = (
        external_form_count / form_count
        if form_count > 0
        else 0
    )

    values = [
        url_length,
        domain_length,
        path_length,
        query_length,
        num_dots,
        num_hyphens,
        num_underscores,
        num_digits,
        num_special_chars,
        url_entropy,
        has_https,
        has_ip,
        suspicious_keyword_count,
        html_length,
        text_length,
        num_links,
        external_links,
        external_link_ratio,
        form_count,
        password_field_count,
        has_login_form,
        external_form_count,
        external_form_ratio
    ]

    return dict(
        zip(
            FEATURE_NAMES,
            values
        )
    )


df = pd.read_csv(
    os.path.join(
        DATASET,
        "ID__URLs_labell.csv"
    )
)

df["label"] = df["typ"].map({
    -1: 0,
    1: 1
})


available_ids = set()

for filename in os.listdir(HTML_DIR):

    if filename.endswith(".txt"):

        try:
            available_ids.add(
                int(filename[:-4])
            )

        except Exception:
            pass


df = df[
    df["id"].isin(available_ids)
]


phishing = df[
    df["label"] == 1
]

benign = df[
    df["label"] == 0
]


random.seed(42)

samples = df[df["id"] == 68954]


print("\n========== D1 SAMPLE TEST ==========\n")

correct = 0


for _, row in samples.iterrows():

    sample_id = int(row["id"])

    actual = int(row["label"])

    html_path = os.path.join(
        HTML_DIR,
        f"{sample_id}.txt"
    )

    with open(
        html_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        html = f.read()

    features = extract_features(
        row["url"],
        html
    )

    prediction, confidence = predict(
        features
    )

    predicted = (
        1
        if prediction == "Phishing"
        else 0
    )

    if predicted == actual:
        correct += 1

    actual_text = (
        "Phishing"
        if actual == 1
        else "Benign"
    )

    print(
        f"ID: {sample_id}"
    )

    print(
        f"Actual: {actual_text}"
    )

    print(
        f"Predicted: {prediction}"
    )

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )

    print("-" * 40)


print(
    f"\nCorrect: {correct}/10"
)

print(
    f"Sample accuracy: {correct / 10 * 100:.2f}%"
)

print(
    "\n========== COMPLETE =========="
)


print("\n========== D1 SAMPLE TEST ==========\n")

correct = 0


for _, row in samples.iterrows():

    sample_id = int(row["id"])

    actual = int(row["label"])

    html_path = os.path.join(
        HTML_DIR,
        f"{sample_id}.txt"
    )

    with open(
        html_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        html = f.read()

    features = extract_features(
        row["url"],
        html
    )

    prediction, confidence = predict(
        features
    )

    predicted = (
        1
        if prediction == "Phishing"
        else 0
    )

    if predicted == actual:
        correct += 1

    actual_text = (
        "Phishing"
        if actual == 1
        else "Benign"
    )

    print(
        f"ID: {sample_id}"
    )

    print(
        f"Actual: {actual_text}"
    )

    print(
        f"Predicted: {prediction}"
    )

    print(
        f"Confidence: {confidence * 100:.2f}%"
    )

    print("-" * 40)


print(
    f"\nCorrect: {correct}/10"
)

print(
    f"Sample accuracy: {correct / 10 * 100:.2f}%"
)

print(
    "\n========== COMPLETE =========="
)