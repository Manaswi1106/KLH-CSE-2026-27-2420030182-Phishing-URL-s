import os
import requests
import numpy as np
import ipaddress
import urllib3

from bs4 import BeautifulSoup
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from hybrid_predictor import hybrid_predict
from domain_verifier import verify_domain


app = FastAPI()

urllib3.disable_warnings(
    urllib3.exceptions.InsecureRequestWarning
)


class URLRequest(BaseModel):
    url: str


def extract_features(url, html):

    soup = BeautifulSoup(html, "html.parser")

    parsed_url = requests.utils.urlparse(url)

    hostname = parsed_url.hostname or ""
    path = parsed_url.path or ""
    query = parsed_url.query or ""

    # URL features

    url_length = len(url)

    domain_length = len(hostname)

    path_length = len(path)

    query_length = len(query)

    num_dots = url.count(".")

    num_hyphens = url.count("-")

    num_underscores = url.count("_")

    num_digits = sum(
        c.isdigit()
        for c in url
    )

    special_chars = set(
        "!@#$%^&*()+=[]{}|\\:;\"'<>,?/~`"
    )

    num_special_chars = sum(
        c in special_chars
        for c in url
    )

    # URL entropy

    if url:

        counts = {}

        for c in url:
            counts[c] = counts.get(c, 0) + 1

        entropy = 0.0

        for count in counts.values():

            p = count / len(url)

            entropy -= p * np.log2(p)

    else:

        entropy = 0.0

    # HTTPS

    has_https = int(
        url.lower().startswith("https://")
    )

    # IP address

    try:

        ipaddress.ip_address(hostname)

        has_ip = 1

    except:

        has_ip = 0

    # Suspicious URL keywords

    suspicious_keywords = [
        "login",
        "signin",
        "verify",
        "verification",
        "account",
        "update",
        "secure",
        "security",
        "confirm",
        "password",
        "credential",
        "bank",
        "paypal",
        "wallet",
        "billing",
        "payment",
        "suspend",
        "unlock",
        "alert",
    ]

    url_lower = url.lower()

    suspicious_keyword_count = sum(
        1
        for word in suspicious_keywords
        if word in url_lower
    )

    # HTML features

    html_length = len(html)

    text = soup.get_text(
        " ",
        strip=True
    )

    text_length = len(text)

    # Links

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

        if href.startswith(
            ("http://", "https://")
        ):

            link_host = (
                requests.utils.urlparse(
                    href
                ).hostname
                or ""
            )

            if link_host and hostname:

                link_domain = (
                    link_host
                    .lower()
                    .replace("www.", "")
                )

                page_domain = (
                    hostname
                    .lower()
                    .replace("www.", "")
                )

                if link_domain != page_domain:

                    external_links += 1

    if num_links > 0:

        external_link_ratio = (
            external_links / num_links
        )

    else:

        external_link_ratio = 0.0

    # Forms

    forms = soup.find_all("form")

    form_count = len(forms)

    # Password fields

    password_fields = soup.find_all(
        "input",
        attrs={
            "type": lambda x:
                x and x.lower() == "password"
        }
    )

    password_field_count = len(
        password_fields
    )

    # Login form

    has_login_form = int(
        password_field_count > 0
        or any(
            word in str(form).lower()
            for form in forms
            for word in [
                "login",
                "signin",
                "sign in",
                "password"
            ]
        )
    )

    # External form destinations

    external_form_count = 0

    for form in forms:

        action = form.get(
            "action",
            ""
        )

        if action.startswith(
            ("http://", "https://")
        ):

            action_host = (
                requests.utils.urlparse(
                    action
                ).hostname
                or ""
            )

            if action_host and hostname:

                action_domain = (
                    action_host
                    .lower()
                    .replace("www.", "")
                )

                page_domain = (
                    hostname
                    .lower()
                    .replace("www.", "")
                )

                if action_domain != page_domain:

                    external_form_count += 1

    if form_count > 0:

        external_form_ratio = (
            external_form_count / form_count
        )

    else:

        external_form_ratio = 0.0

    # Return the exact 23 features
    # expected by model_predictor.py

    return {

        "url_length":
            url_length,

        "domain_length":
            domain_length,

        "path_length":
            path_length,

        "query_length":
            query_length,

        "num_dots":
            num_dots,

        "num_hyphens":
            num_hyphens,

        "num_underscores":
            num_underscores,

        "num_digits":
            num_digits,

        "num_special_chars":
            num_special_chars,

        "url_entropy":
            entropy,

        "has_https":
            has_https,

        "has_ip":
            has_ip,

        "suspicious_keyword_count":
            suspicious_keyword_count,

        "html_length":
            html_length,

        "text_length":
            text_length,

        "num_links":
            num_links,

        "external_links":
            external_links,

        "external_link_ratio":
            external_link_ratio,

        "form_count":
            form_count,

        "password_field_count":
            password_field_count,

        "has_login_form":
            has_login_form,

        "external_form_count":
            external_form_count,

        "external_form_ratio":
            external_form_ratio,
    }


def make_final_decision(
    prediction,
    confidence,
    domain_info,
    features
):

    trusted = domain_info.get("trusted", False)

    suspicious_keywords = features["suspicious_keyword_count"]
    password_fields = features["password_field_count"]
    login_form = features["has_login_form"]
    external_forms = features["external_form_count"]
    html_length = features["html_length"]

    if trusted:
        return {
            "decision": "Benign",
            "decision_reason": (
                "The registered domain is verified as a trusted "
                "legitimate domain. The ML prediction is retained "
                "separately for transparency."
            )
        }

    if html_length == 0:
        return {
            "decision": "Suspicious",
            "decision_reason": (
                "The website could not be accessed, so webpage-level "
                "evidence could not be evaluated. The ML model produced "
                "a URL-based prediction, but the available evidence "
                "is insufficient for a definitive classification."
            )
        }

    strong_phishing_evidence = (
        suspicious_keywords >= 2
        or password_fields > 0
        or login_form == 1
        or external_forms > 0
    )

    if (
    prediction == "Phishing"
    and confidence >= 0.70
    and (
        strong_phishing_evidence
        or confidence >= 95.0
    )
        ):
        return {
            "decision": "Phishing",
            "decision_reason": (
                "The ML model predicts phishing with high confidence, "
                "the domain is not trusted, and the accessible webpage "
                "contains additional phishing-related indicators."
            )
        }

    if (
        prediction == "Benign"
        and confidence >= 0.70
        and not strong_phishing_evidence
    ):
        return {
            "decision": "Benign",
            "decision_reason": (
                "The ML model predicts the website as benign with high "
                "confidence, and the accessible webpage does not contain "
                "the evaluated phishing-related indicators. The domain "
                "is unverified, so domain status is reported separately."
            )
        }

    return {
        "decision": "Suspicious",
        "decision_reason": (
            "The available ML, domain, and webpage evidence is not "
            "sufficient for a high-confidence final classification."
        )
    }
def root():

    return {

        "message":
            "PhishXpert API is running"
    }


@app.get("/app")
def app_page():

    return FileResponse(
        "static/index.html"
    )


@app.post("/predict")
def predict_url(
    request: URLRequest
):

    url = request.url.strip()

    if not url.startswith(
        ("http://", "https://")
    ):

        url = "https://" + url

    access_status = (
        "Website accessible"
    )

    feature_mode = (
        "URL + HTML features"
    )

    html = ""

    # Fetch webpage

    try:

        response = requests.get(

            url,

            headers={

                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/153.0.0.0 "
                    "Safari/537.36"
                )
            },

            timeout=(10, 20),

            allow_redirects=True,

            # Allows analysis of pages with
            # self-signed certificates.
            # Certificate status is not treated
            # as trusted because of this.
            verify=False
        )

        html = response.text

        if not html:

            access_status = (
                "Website inaccessible"
            )

            feature_mode = (
                "URL-only features"
            )

    except Exception:

        html = ""

        access_status = (
            "Website inaccessible"
        )

        feature_mode = (
            "URL-only features"
        )

    # Extract features

    features = extract_features(
        url,
        html
    )

    # ML prediction

    ml_prediction, ml_confidence, shap_features = hybrid_predict(features, html)

    # Domain verification

    domain_info = verify_domain(
        url
    )

    # Final decision

    final_decision = make_final_decision(

        ml_prediction,

        ml_confidence,

        domain_info,

        features
    )

    return {

        # ML result

        "prediction":
            ml_prediction,

        "confidence":
            round(
                float(
                    ml_confidence
                ) * 100,
                2
            ),

        # Final decision

        "final_decision":
            final_decision[
                "decision"
            ],

        "decision_reason":
            final_decision[
                "decision_reason"
            ],

        # Website information

        "access_status":
            access_status,

        "feature_mode":
            feature_mode,

        "model": "PhishXpert Hybrid (DistilBERT + LightGBM)",
        "shap_explanation": shap_features,


        # Domain verification

        "domain_verification":
            domain_info,

        # Explainable evidence

        "evidence": {

            "suspicious_keyword_count":
                features[
                    "suspicious_keyword_count"
                ],

            "html_length":
                features[
                    "html_length"
                ],

            "text_length":
                features[
                    "text_length"
                ],

            "num_links":
                features[
                    "num_links"
                ],

            "external_links":
                features[
                    "external_links"
                ],

            "external_link_ratio":
                round(
                    float(
                        features[
                            "external_link_ratio"
                        ]
                    ),
                    4
                ),

            "form_count":
                features[
                    "form_count"
                ],

            "password_field_count":
                features[
                    "password_field_count"
                ],

            "has_login_form":
                features[
                    "has_login_form"
                ],

            "external_form_count":
                features[
                    "external_form_count"
                ],

            "external_form_ratio":
                round(
                    float(
                        features[
                            "external_form_ratio"
                        ]
                    ),
                    4
                ),
        },

        # Overall explanation

        "explanation": (
            "PhishXpert ML prediction and domain "
            "verification are reported separately. "
            "The final decision combines the learned "
            "ML signal with domain and webpage evidence."
        )
    }