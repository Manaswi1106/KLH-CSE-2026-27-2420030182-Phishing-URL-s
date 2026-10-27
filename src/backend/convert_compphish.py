import os
import re
import math
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse


ALT_DIR = r"D:\Documents\Aruna Grace\ALT"

MAPPING_FILE = None
HTML_DIR = None

for root, dirs, files in os.walk(ALT_DIR):
    if "Mapping_File.xlsx" in files:
        MAPPING_FILE = os.path.join(root, "Mapping_File.xlsx")

    if "All_HTML" in dirs:
        possible = os.path.join(root, "All_HTML")

        if HTML_DIR is None:
            HTML_DIR = possible


if MAPPING_FILE is None:
    raise FileNotFoundError(
        "Mapping_File.xlsx was not found inside ALT."
    )

if HTML_DIR is None:
    raise FileNotFoundError(
        "All_HTML folder was not found inside ALT."
    )


OUTPUT_FILE = os.path.join(
    os.path.dirname(MAPPING_FILE),
    "CompPhish_features.csv"
)


SUSPICIOUS_WORDS = [
    "login",
    "signin",
    "verify",
    "verification",
    "account",
    "update",
    "secure",
    "security",
    "password",
    "credential",
    "confirm",
    "bank",
    "paypal",
    "wallet",
    "recover",
    "unlock",
    "authenticate",
    "authentication"
]


FEATURES = [
    "id",
    "url",
    "label",
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


def entropy(text):

    if not text:
        return 0.0

    counts = {}

    for char in text:
        counts[char] = counts.get(char, 0) + 1

    length = len(text)

    result = 0.0

    for count in counts.values():

        p = count / length

        result -= p * math.log2(p)

    return result


def get_domain(url):

    try:

        if not re.match(r"^[a-zA-Z]+://", url):
            url = "http://" + url

        parsed = urlparse(url)

        return (
            parsed.netloc
            .lower()
            .split(":")[0]
            .replace("www.", "")
        )

    except:

        return ""


def extract_features(url, html):

    if not isinstance(url, str):
        url = ""

    if not isinstance(html, str):
        html = ""

    original_url = url

    if not re.match(r"^[a-zA-Z]+://", url):
        url = "http://" + url

    try:
        parsed = urlparse(url)
    except:
        parsed = urlparse("http://")

    domain = parsed.netloc

    domain_clean = domain.split(":")[0]

    path = parsed.path or ""

    query = parsed.query or ""

    url_lower = original_url.lower()


    url_length = len(original_url)

    domain_length = len(domain_clean)

    path_length = len(path)

    query_length = len(query)


    num_dots = original_url.count(".")

    num_hyphens = original_url.count("-")

    num_underscores = original_url.count("_")

    num_digits = sum(
        c.isdigit()
        for c in original_url
    )


    special_chars = "@$%&=!?;:'_-,"

    num_special_chars = sum(
        original_url.count(c)
        for c in special_chars
    )


    url_entropy = entropy(original_url)


    has_https = int(
        parsed.scheme.lower() == "https"
    )


    has_ip = 0

    try:

        parts = domain_clean.split(".")

        if (
            len(parts) == 4
            and all(
                part.isdigit()
                and 0 <= int(part) <= 255
                for part in parts
            )
        ):
            has_ip = 1

    except:

        has_ip = 0


    suspicious_keyword_count = sum(
        1
        for word in SUSPICIOUS_WORDS
        if word in url_lower
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


    base_domain = get_domain(
        original_url
    )


    external_links = 0


    for link in links:

        href = link.get(
            "href",
            ""
        )

        if not href:
            continue

        if href.startswith(
            (
                "javascript:",
                "#",
                "mailto:"
            )
        ):
            continue

        try:

            absolute = urlparse(href)

            if absolute.netloc:

                link_domain = (
                    absolute.netloc
                    .lower()
                    .split(":")[0]
                    .replace("www.", "")
                )

                if (
                    link_domain
                    and link_domain != base_domain
                ):
                    external_links += 1

        except:

            pass


    external_link_ratio = (
        external_links / num_links
        if num_links > 0
        else 0.0
    )


    forms = soup.find_all("form")

    form_count = len(forms)


    password_field_count = 0


    for inp in soup.find_all("input"):

        input_type = (
            inp.get(
                "type",
                ""
            )
            .lower()
        )

        if input_type == "password":

            password_field_count += 1


    has_login_form = 0


    for form in forms:

        form_text = form.get_text(
            " ",
            strip=True
        ).lower()

        inputs = form.find_all(
            "input"
        )


        has_password = any(
            inp.get(
                "type",
                ""
            ).lower() == "password"
            for inp in inputs
        )


        login_words = [
            "login",
            "log in",
            "signin",
            "sign in",
            "password",
            "username"
        ]


        has_login_word = any(
            word in form_text
            for word in login_words
        )


        if (
            has_password
            or has_login_word
        ):

            has_login_form = 1

            break


    external_form_count = 0


    for form in forms:

        action = form.get(
            "action",
            ""
        )

        if not action:
            continue

        try:

            action_parsed = urlparse(
                action
            )

            if action_parsed.netloc:

                action_domain = (
                    action_parsed.netloc
                    .lower()
                    .split(":")[0]
                    .replace("www.", "")
                )

                if (
                    action_domain
                    and action_domain != base_domain
                ):

                    external_form_count += 1

        except:

            pass


    external_form_ratio = (
        external_form_count / form_count
        if form_count > 0
        else 0.0
    )


    return {
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
        "external_form_ratio": external_form_ratio
    }


print("=" * 60)
print("COMPPHISH → PHISHXPERT FEATURE CONVERSION")
print("=" * 60)

print("\nMapping file:")
print(MAPPING_FILE)

print("\nHTML directory:")
print(HTML_DIR)


print("\nSearching for HTML files...")

html_index = {}


for root, dirs, files in os.walk(HTML_DIR):

    for filename in files:

        full_path = os.path.join(
            root,
            filename
        )

        stem = os.path.splitext(
            filename
        )[0].strip()

        if stem:

            html_index[stem] = full_path


print(
    "HTML files found:",
    len(html_index)
)


if len(html_index) == 0:

    raise RuntimeError(
        "No HTML files were found inside All_HTML."
    )


print("\nExample HTML mappings:")

shown = 0

for key, value in html_index.items():

    print(
        key,
        "→",
        value
    )

    shown += 1

    if shown >= 5:
        break


print("\nReading Mapping_File.xlsx...")

mapping = pd.read_excel(
    MAPPING_FILE
)


print(
    "Columns:",
    list(mapping.columns)
)

print(
    "Rows:",
    len(mapping)
)


results = []

missing_html = 0

failed_html = 0


for index, row in mapping.iterrows():

    serial = str(
        row["serial_number"]
    ).strip()

    url = str(
        row["url"]
    ).strip()

    label = int(
        row["label"]
    )


    html_path = html_index.get(
        serial
    )


    if html_path is None:

        missing_html += 1

        html = ""

    else:

        try:

            with open(
                html_path,
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as f:

                html = f.read()

        except Exception:

            failed_html += 1

            html = ""


    features = extract_features(
        url,
        html
    )


    record = {
        "id": serial,
        "url": url,
        "label": label
    }


    record.update(
        features
    )


    results.append(
        record
    )


    if (
        (index + 1) % 500 == 0
    ):

        print(
            f"Processed {index + 1}/{len(mapping)}"
        )


df = pd.DataFrame(
    results
)


df = df[FEATURES]


df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 60)
print("CONVERSION COMPLETE")
print("=" * 60)

print(
    "Total records:",
    len(df)
)

print(
    "HTML files found:",
    len(html_index)
)

print(
    "Missing HTML:",
    missing_html
)

print(
    "HTML read failures:",
    failed_html
)


print("\nLabel distribution:")

print(
    df["label"]
    .value_counts()
    .sort_index()
)


print("\nOutput:")

print(
    OUTPUT_FILE
)


print("\nFeature count:")

print(
    len(FEATURES) - 3
)


print("=" * 60)