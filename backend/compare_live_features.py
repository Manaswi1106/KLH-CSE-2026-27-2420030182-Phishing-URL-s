import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from collections import Counter
from pathlib import Path
import joblib
import re
import math

SITES = {
    "Google": "https://www.google.com/",
    "YouTube": "https://www.youtube.com/",
    "GitHub": "https://github.com/",
    "Wikipedia": "https://www.wikipedia.org/",
    "CNN": "https://www.cnn.com/"
}

FEATURES = [
    "url_length", "domain_length", "path_length", "query_length",
    "num_dots", "num_hyphens", "num_underscores", "num_digits",
    "num_special_chars", "url_entropy", "has_https", "has_ip",
    "suspicious_keyword_count", "html_length", "text_length",
    "num_links", "external_links", "external_link_ratio",
    "form_count", "password_field_count", "has_login_form",
    "external_form_count", "external_form_ratio"
]

SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "verification", "account",
    "secure", "update", "confirm", "bank", "paypal", "password",
    "credential", "wallet", "billing", "suspend", "unlock"
]

TRAIN = pd.read_csv(
    Path(r"D:\Documents\Aruna Grace\ALT\Dataset(D1)") / "X_train.csv"
)

scaler = joblib.load("models/phishxpert_scaler.pkl")


def entropy(text):
    if not text:
        return 0.0

    counts = Counter(text)
    total = len(text)

    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
    )


def extract_features(url, html):
    parsed = urlparse(url)

    domain = parsed.netloc
    path = parsed.path
    query = parsed.query

    soup = BeautifulSoup(html, "html.parser")

    text = soup.get_text(" ", strip=True)

    links = soup.find_all("a", href=True)

    external_links = 0

    for link in links:
        href = link.get("href", "")

        if href.startswith(("http://", "https://")):
            link_domain = urlparse(href).netloc

            if link_domain and link_domain != domain:
                external_links += 1

    forms = soup.find_all("form")

    password_fields = soup.find_all(
        "input",
        attrs={"type": re.compile("^password$", re.I)}
    )

    login_forms = 0

    for form in forms:
        form_text = form.get_text(" ", strip=True).lower()

        if any(word in form_text for word in
               ["login", "sign in", "signin", "password"]):
            login_forms += 1

    external_forms = 0

    for form in forms:
        action = form.get("action", "")

        if action.startswith(("http://", "https://")):
            action_domain = urlparse(action).netloc

            if action_domain and action_domain != domain:
                external_forms += 1

    special_chars = sum(
        not c.isalnum() for c in url
    )

    suspicious_count = sum(
        url.lower().count(word)
        for word in SUSPICIOUS_WORDS
    )

    link_count = len(links)
    form_count = len(forms)

    return {
        "url_length": len(url),
        "domain_length": len(domain),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": special_chars,
        "url_entropy": entropy(url),
        "has_https": int(parsed.scheme == "https"),
        "has_ip": int(bool(re.fullmatch(
            r"\d{1,3}(\.\d{1,3}){3}", domain
        ))),
        "suspicious_keyword_count": suspicious_count,
        "html_length": len(html),
        "text_length": len(text),
        "num_links": link_count,
        "external_links": external_links,
        "external_link_ratio": (
            external_links / link_count if link_count else 0
        ),
        "form_count": form_count,
        "password_field_count": len(password_fields),
        "has_login_form": int(login_forms > 0),
        "external_form_count": external_forms,
        "external_form_ratio": (
            external_forms / form_count if form_count else 0
        )
    }


print("=" * 80)
print("LIVE WEBSITE vs D1 DISTRIBUTION")
print("=" * 80)

results = []

for name, url in SITES.items():

    print(f"\nFetching {name}...")

    try:
        response = requests.get(
            url,
            timeout=(3, 5),
            headers={"User-Agent": "Mozilla/5.0"},
            allow_redirects=True
        )

        features = extract_features(
            response.url,
            response.text
        )

        row = {"site": name}

        for feature in FEATURES:
            value = features[feature]
            mean = TRAIN[feature].mean()
            std = TRAIN[feature].std()

            if std == 0:
                z = 0
            else:
                z = (value - mean) / std

            row[feature] = value
            row[feature + "_z"] = z

        results.append(row)

        print("Accessible:", response.status_code)
        print("URL:", response.url)

    except Exception as e:
        print("ERROR:", e)


df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("FEATURE VALUES")
print("=" * 80)

for _, row in df.iterrows():

    print(f"\n--- {row['site']} ---")

    for feature in FEATURES:
        print(
            f"{feature:30} "
            f"{row[feature]:12.3f}"
        )

print("\n" + "=" * 80)
print("MOST OUT-OF-DISTRIBUTION FEATURES")
print("=" * 80)

for _, row in df.iterrows():

    deviations = []

    for feature in FEATURES:

        z = abs(row[feature + "_z"])

        if not np.isfinite(z):
            z = 0

        deviations.append((feature, z))

    deviations.sort(
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\n--- {row['site']} ---")

    for feature, z in deviations[:8]:

        print(
            f"{feature:30} "
            f"| absolute z-score = {z:.2f}"
        )

print("\n" + "=" * 80)
print("COMPLETE")
print("=" * 80)