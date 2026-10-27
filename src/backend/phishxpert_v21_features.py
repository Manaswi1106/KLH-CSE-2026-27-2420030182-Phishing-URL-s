import os
import re
import math
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

SEED = 42

D1_DIR = r"D:\Documents\Aruna Grace\ALT\Dataset(D1)"
D1_HTML_DIR = os.path.join(D1_DIR, "sourc_Data(D1)")
D1_MAPPING = os.path.join(D1_DIR, "ID__URLs_labell.csv")

COMP_DIR = r"D:\Documents\Aruna Grace\ALT\CompPhish\CompPhish"
COMP_HTML_DIR = os.path.join(COMP_DIR, "All_HTML")
COMP_MAPPING = os.path.join(COMP_DIR, "Mapping_File.xlsx")

OUTPUT_DIR = r"D:\Documents\Aruna Grace\ALT\PhishXpert\v21_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

SUSPICIOUS_TLDS = {
    ".tk", ".ml", ".ga", ".cf", ".gq",
    ".top", ".xyz", ".click", ".work",
    ".support", ".zip", ".mov"
}


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


def get_domain(url):
    try:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        return urlparse(url).netloc.lower().split("@")[-1].split(":")[0]
    except Exception:
        return ""


def get_hostname(url):
    try:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        return urlparse(url).hostname or ""
    except Exception:
        return ""


def extract_features(url, html=""):
    original_url = str(url)

    if not original_url.startswith(("http://", "https://")):
        working_url = "https://" + original_url
    else:
        working_url = original_url

    parsed = urlparse(working_url)

    domain = parsed.netloc
    path = parsed.path
    query = parsed.query

    url_lower = original_url.lower()

    # =========================================================
    # EXISTING 23 FEATURES
    # =========================================================

    url_length = len(original_url)
    domain_length = len(domain)
    path_length = len(path)
    query_length = len(query)

    num_dots = original_url.count(".")
    num_hyphens = original_url.count("-")
    num_underscores = original_url.count("_")
    num_digits = sum(c.isdigit() for c in original_url)

    num_special_chars = sum(
        not c.isalnum()
        for c in original_url
    )

    url_entropy = entropy(original_url)

    has_https = int(parsed.scheme == "https")

    has_ip = int(
        domain.replace(".", "").isdigit()
    )

    suspicious_keyword_count = sum(
        word in url_lower
        for word in SUSPICIOUS_WORDS
    )

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    html_length = len(html)

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
            link_domain = urlparse(href).netloc

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

    # =========================================================
    # NEW V2.1 FEATURES
    # =========================================================

    hostname = get_hostname(original_url)

    hostname_parts = [
        part for part in hostname.split(".")
        if part
    ]

    subdomain_count = max(
        len(hostname_parts) - 2,
        0
    )

    at_symbol_count = original_url.count("@")

    encoded_char_count = len(
        re.findall(
            r"%[0-9a-fA-F]{2}",
            original_url
        )
    )

    percent_encoded_ratio = (
        encoded_char_count / len(original_url)
        if len(original_url) > 0
        else 0
    )

    punycode_flag = int(
        "xn--" in hostname.lower()
    )

    hostname_digits = sum(
        c.isdigit()
        for c in hostname
    )

    hostname_letters_symbols = sum(
        c.isalnum()
        for c in hostname
    )

    hostname_digit_ratio = (
        hostname_digits / hostname_letters_symbols
        if hostname_letters_symbols > 0
        else 0
    )

    hostname_hyphens = hostname.count("-")

    hostname_hyphen_ratio = (
        hostname_hyphens / len(hostname)
        if len(hostname) > 0
        else 0
    )

    url_tokens = re.split(
        r"[^a-zA-Z0-9]+",
        original_url
    )

    url_token_count = len([
        token
        for token in url_tokens
        if token
    ])

    tld = ""

    if "." in hostname:
        tld = "." + hostname.rsplit(".", 1)[-1].lower()

    suspicious_tld_flag = int(
        tld in SUSPICIOUS_TLDS
    )

    # -------------------------
    # HTML structural features
    # -------------------------

    iframe_count = len(
        soup.find_all("iframe")
    )

    script_count = len(
        soup.find_all("script")
    )

    hidden_element_count = 0

    for element in soup.find_all(True):
        style = str(
            element.get("style", "")
        ).lower()

        hidden_attr = element.has_attr(
            "hidden"
        )

        aria_hidden = (
            str(
                element.get(
                    "aria-hidden",
                    ""
                )
            ).lower()
            == "true"
        )

        if (
            hidden_attr
            or aria_hidden
            or "display:none" in style
            or "display: none" in style
            or "visibility:hidden" in style
            or "visibility: hidden" in style
        ):
            hidden_element_count += 1

    meta_refresh_flag = 0

    for meta in soup.find_all("meta"):
        http_equiv = str(
            meta.get(
                "http-equiv",
                ""
            )
        ).lower()

        if http_equiv == "refresh":
            meta_refresh_flag = 1
            break

    input_count = len(
        soup.find_all("input")
    )

    # =========================================================
    # RETURN 35 FEATURES
    # =========================================================

    return {
        # Existing 23
        "url_length": url_length,
        "domain_length": domain_length,
        "path_length": path_length,
        "query_length": query_length,
        "num_dots": num_dots,
        "num_hyphens": num_hyphens,
        "num_underscores": num_underscores,
        "num_digits": num_digits,
        "num_special_chars": num_special_chars,
        "url_entropy": url_entropy,
        "has_https": has_https,
        "has_ip": has_ip,
        "suspicious_keyword_count": suspicious_keyword_count,
        "html_length": html_length,
        "text_length": text_length,
        "num_links": num_links,
        "external_links": external_links,
        "external_link_ratio": external_link_ratio,
        "form_count": form_count,
        "password_field_count": password_field_count,
        "has_login_form": has_login_form,
        "external_form_count": external_form_count,
        "external_form_ratio": external_form_ratio,

        # New V2.1
        "subdomain_count": subdomain_count,
        "at_symbol_count": at_symbol_count,
        "encoded_char_count": encoded_char_count,
        "percent_encoded_ratio": percent_encoded_ratio,
        "punycode_flag": punycode_flag,
        "hostname_digit_ratio": hostname_digit_ratio,
        "hostname_hyphen_ratio": hostname_hyphen_ratio,
        "url_token_count": url_token_count,
        "suspicious_tld_flag": suspicious_tld_flag,
        "iframe_count": iframe_count,
        "script_count": script_count,
        "hidden_element_count": hidden_element_count,
        "meta_refresh_flag": meta_refresh_flag,
        "input_count": input_count
    }


def process_d1():
    print("\n========== D1 V2.1 EXTRACTION ==========")

    mapping = pd.read_csv(
        D1_MAPPING
    )

    rows = []

    html_files = {}

    for filename in os.listdir(D1_HTML_DIR):
        path = os.path.join(
            D1_HTML_DIR,
            filename
        )

        if os.path.isfile(path):
            stem = os.path.splitext(
                filename
            )[0]

            html_files[stem] = path

    print(
        "D1 records:",
        len(mapping)
    )

    print(
        "D1 HTML files:",
        len(html_files)
    )

    for index, row in mapping.iterrows():

        record_id = str(
            row["id"]
        )

        url = str(
            row["url"]
        )

        label = int(
            1 if int(row["typ"]) == 1 else 0
        )

        html = ""

        html_path = html_files.get(
            record_id
        )

        if html_path:
            try:
                with open(
                    html_path,
                    "r",
                    encoding="utf-8",
                    errors="ignore"
                ) as f:
                    html = f.read()
            except Exception:
                html = ""

        features = extract_features(
            url,
            html
        )

        features["id"] = record_id
        features["url"] = url
        features["label"] = label

        rows.append(features)

        if (index + 1) % 5000 == 0:
            print(
                f"Processed D1: {index + 1}/{len(mapping)}"
            )

    df = pd.DataFrame(rows)

    output = os.path.join(
        OUTPUT_DIR,
        "D1_features_v21.csv"
    )

    df.to_csv(
        output,
        index=False
    )

    print(
        "\nSaved:",
        output
    )

    print(
        "Shape:",
        df.shape
    )


def process_compphish():
    print("\n========== COMPPHISH V2.1 EXTRACTION ==========")

    mapping = pd.read_excel(
        COMP_MAPPING
    )

    rows = []

    html_files = {}

    for root, dirs, files in os.walk(
        COMP_HTML_DIR
    ):
        for filename in files:

            path = os.path.join(
                root,
                filename
            )

            stem = os.path.splitext(
                filename
            )[0]

            html_files[stem] = path

    print(
        "CompPhish records:",
        len(mapping)
    )

    print(
        "CompPhish HTML files:",
        len(html_files)
    )

    for index, row in mapping.iterrows():

        serial = str(
            row["serial_number"]
        )

        url = str(
            row["url"]
        )

        label_value = str(
            row["label"]
        ).strip().lower()

        if label_value in [
            "phishing",
            "phish",
            "1",
            "true"
        ]:
            label = 1
        else:
            label = 0

        html = ""

        html_path = html_files.get(
            serial
        )

        if html_path:
            try:
                with open(
                    html_path,
                    "r",
                    encoding="utf-8",
                    errors="ignore"
                ) as f:
                    html = f.read()
            except Exception:
                html = ""

        features = extract_features(
            url,
            html
        )

        features["serial_number"] = serial
        features["url"] = url
        features["label"] = label

        rows.append(features)

        if (index + 1) % 2000 == 0:
            print(
                f"Processed CompPhish: {index + 1}/{len(mapping)}"
            )

    df = pd.DataFrame(rows)

    output = os.path.join(
        OUTPUT_DIR,
        "CompPhish_features_v21.csv"
    )

    df.to_csv(
        output,
        index=False
    )

    print(
        "\nSaved:",
        output
    )

    print(
        "Shape:",
        df.shape
    )


if __name__ == "__main__":

    process_d1()

    process_compphish()

    print(
        "\n========== V2.1 EXTRACTION COMPLETE =========="
    )