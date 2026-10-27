import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import math
import re
import pandas as pd

SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "verification", "secure",
    "account", "update", "confirm", "bank", "password",
    "credential", "wallet", "payment", "authenticate"
]

def entropy(text):
    if not text:
        return 0

    counts = {}
    for c in text:
        counts[c] = counts.get(c, 0) + 1

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

    links = soup.find_all("a")
    forms = soup.find_all("form")
    password_fields = soup.find_all(
        "input",
        attrs={"type": lambda x: x and x.lower() == "password"}
    )

    suspicious_count = sum(
        url.lower().count(word)
        for word in SUSPICIOUS_WORDS
    )

    external_links = 0

    for link in links:
        href = link.get("href")

        if href and href.startswith(("http://", "https://")):
            link_domain = urlparse(href).netloc

            if link_domain and link_domain != domain:
                external_links += 1

    external_ratio = (
        external_links / len(links)
        if links else 0
    )

    features = {
        "url_length": len(url),
        "domain_length": len(domain),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": sum(
            not c.isalnum() for c in url
        ),
        "url_entropy": entropy(url),
        "has_https": int(url.lower().startswith("https://")),
        "has_ip": int(bool(
            re.match(r"^\d+\.\d+\.\d+\.\d+$", domain)
        )),
        "suspicious_keyword_count": suspicious_count,
        "html_length": len(html),
        "text_length": len(text),
        "num_links": len(links),
        "external_links": external_links,
        "external_link_ratio": external_ratio,
        "form_count": len(forms),
        "password_field_count": len(password_fields),
        "has_login_form": int(
            any(
                "login" in str(form).lower() or
                "signin" in str(form).lower()
                for form in forms
            )
        ),
        "external_form_count": 0,
        "external_form_ratio": 0
    }

    return features


sites = [
    "https://www.google.com/",
    "https://www.youtube.com/",
    "https://www.microsoft.com/",
    "https://github.com/",
    "https://www.wikipedia.org/"
]

rows = []

for url in sites:

    print("\nTesting:", url)

    try:
        response = requests.get(
            url,
            timeout=(3, 5),
            allow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        print("Status:", response.status_code)

        features = extract_features(
            url,
            response.text
        )

        features["website"] = url

        rows.append(features)

    except Exception as e:

        print("ERROR:", e)

        features = extract_features(url, "")

        features["website"] = url

        rows.append(features)


df = pd.DataFrame(rows)

columns = ["website"] + [
    c for c in df.columns
    if c != "website"
]

df = df[columns]

print("\n========== LIVE WEBSITE FEATURES ==========\n")

print(df.to_string(index=False))

df.to_csv(
    "live_site_features.csv",
    index=False
)

print("\nSaved: live_site_features.csv")