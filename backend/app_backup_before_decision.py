import requests
import numpy as np

from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from model_predictor import predict
from domain_verifier import verify_domain


app = FastAPI(title="PhishXpert")


class URLRequest(BaseModel):
    url: str


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
        p * np.log2(p)
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

    # =========================================================
    # URL FEATURES
    # =========================================================

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

    url_entropy = entropy(
        original_url
    )

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

    # =========================================================
    # HTML FEATURES
    # =========================================================

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

        href = link.get(
            "href",
            ""
        )

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

    forms = soup.find_all(
        "form"
    )

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


@app.get("/")
def home():

    return {
        "message": "PhishXpert API is running"
    }


@app.get("/app")
def frontend():

    return FileResponse(
        "static/index.html"
    )


@app.post("/predict")
def predict_url(
    request: URLRequest
):

    print(
        "\n========== NEW REQUEST ==========",
        flush=True
    )

    url = request.url.strip()

    print(
        f"URL: {url}",
        flush=True
    )

    # =========================================================
    # DOMAIN VERIFICATION
    # =========================================================

    print(
        "Verifying domain...",
        flush=True
    )

    domain_info = verify_domain(
        url
    )

    print(
        f"Hostname: "
        f"{domain_info['hostname']}",
        flush=True
    )

    print(
        f"Registered domain: "
        f"{domain_info['registered_domain']}",
        flush=True
    )

    print(
        f"Trusted domain: "
        f"{domain_info['trusted']}",
        flush=True
    )

    # =========================================================
    # WEBSITE FETCH
    # =========================================================

    html = ""

    access_status = "Website inaccessible"

    feature_mode = "URL features only"

    try:

        target_url = url

        if not target_url.startswith(
            ("http://", "https://")
        ):
            target_url = (
                "https://" + target_url
            )

        print(
            "Fetching webpage...",
            flush=True
        )

        response = requests.get(
            target_url,
            timeout=(3, 5),
            headers={
                "User-Agent": "Mozilla/5.0"
            },
            allow_redirects=True
        )

        print(
            f"HTTP status: "
            f"{response.status_code}",
            flush=True
        )

        if response.status_code < 400:

            html = response.text

            access_status = (
                "Website accessible"
            )

            feature_mode = (
                "URL + HTML features"
            )

            print(
                f"HTML received: "
                f"{len(html)} characters",
                flush=True
            )

    except Exception as e:

        print(
            f"Website fetch failed: "
            f"{type(e).__name__}: {e}",
            flush=True
        )

        html = ""

    # =========================================================
    # FEATURE EXTRACTION
    # =========================================================

    print(
        "Extracting features...",
        flush=True
    )

    features = extract_features(
        url,
        html
    )

    print(
        "\n---------- FEATURES ----------",
        flush=True
    )

    for name in FEATURE_NAMES:

        print(
            f"{name}: {features[name]}",
            flush=True
        )

    print(
        "------------------------------",
        flush=True
    )

    # =========================================================
    # PHISHXPERT ML MODEL
    # =========================================================

    print(
        "Running PhishXpert model...",
        flush=True
    )

    prediction, confidence = predict(
        features
    )

    print(
        f"ML Prediction: {prediction}",
        flush=True
    )

    print(
        f"ML Confidence: {confidence:.2%}",
        flush=True
    )

    # =========================================================
    # DOMAIN STATUS
    # =========================================================

    if domain_info["trusted"]:

        domain_status = (
            "Trusted legitimate domain"
        )

    else:

        domain_status = (
            "Unknown domain"
        )

    # =========================================================
    # FINAL RESPONSE
    # =========================================================

    print(
        f"Domain status: {domain_status}",
        flush=True
    )

    print(
        "========== REQUEST COMPLETE ==========\n",
        flush=True
    )

    return {

        "prediction": prediction,

        "confidence": round(
            confidence * 100,
            2
        ),

        "access_status": access_status,

        "feature_mode": feature_mode,

        "model": (
            "PhishXpert Neural "
            "Evidence Fusion"
        ),

        "domain_verification": {

            "status": domain_status,

            "hostname": (
                domain_info["hostname"]
            ),

            "registered_domain": (
                domain_info[
                    "registered_domain"
                ]
            ),

            "trusted": (
                domain_info["trusted"]
            )
        },

        "explanation": (
            "The prediction is generated "
            "by the PhishXpert ML model. "
            "Domain verification is provided "
            "as an additional security signal "
            "and does not alter the ML prediction."
        )
    }