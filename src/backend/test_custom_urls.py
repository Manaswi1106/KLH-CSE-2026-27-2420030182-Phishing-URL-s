import requests
import torch
import torch.nn as nn
import pandas as pd
import joblib
import math
from bs4 import BeautifulSoup
from urllib.parse import urlparse


MODEL_PATH = r"D:\Documents\Aruna Grace\ALT\PhishXpert\models\phishxpert_generalized.pth"
SCALER_PATH = r"D:\Documents\Aruna Grace\ALT\PhishXpert\models\phishxpert_generalized_scaler.pkl"


FEATURES = [
    "url_length","domain_length","path_length","query_length",
    "num_dots","num_hyphens","num_underscores","num_digits",
    "num_special_chars","url_entropy","has_https","has_ip",
    "suspicious_keyword_count","html_length","text_length",
    "num_links","external_links","external_link_ratio",
    "form_count","password_field_count","has_login_form",
    "external_form_count","external_form_ratio"
]


SUSPICIOUS_WORDS = [
    "login","signin","verify","verification","account","update",
    "secure","security","password","credential","confirm",
    "bank","paypal","wallet","recover","unlock",
    "authenticate","authentication"
]


class PhishXpert(nn.Module):

    def __init__(self):
        super().__init__()

        self.url_encoder = nn.Sequential(
            nn.Linear(13, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(48, 24),
            nn.ReLU()
        )

        self.html_encoder = nn.Sequential(
            nn.Linear(10, 48),
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


def entropy(text):

    if not text:
        return 0.0

    counts = {}

    for c in text:
        counts[c] = counts.get(c, 0) + 1

    result = 0.0

    for count in counts.values():

        p = count / len(text)

        result -= p * math.log2(p)

    return result


def get_domain(url):

    try:

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

    parsed = urlparse(url)

    domain = (
        parsed.netloc
        .lower()
        .split(":")[0]
    )

    base_domain = domain.replace(
        "www.",
        ""
    )

    path = parsed.path or ""
    query = parsed.query or ""

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    text = soup.get_text(
        " ",
        strip=True
    )

    links = soup.find_all(
        "a",
        href=True
    )

    forms = soup.find_all(
        "form"
    )

    external_links = 0

    for link in links:

        href = link.get(
            "href",
            ""
        )

        try:

            p = urlparse(href)

            if p.netloc:

                d = (
                    p.netloc
                    .lower()
                    .split(":")[0]
                    .replace("www.", "")
                )

                if d and d != base_domain:
                    external_links += 1

        except:
            pass


    password_fields = 0

    for inp in soup.find_all("input"):

        if inp.get(
            "type",
            ""
        ).lower() == "password":

            password_fields += 1


    has_login_form = 0

    for form in forms:

        form_text = form.get_text(
            " ",
            strip=True
        ).lower()

        if (
            password_fields > 0
            or "login" in form_text
            or "sign in" in form_text
            or "signin" in form_text
            or "username" in form_text
        ):

            has_login_form = 1
            break


    external_forms = 0

    for form in forms:

        action = form.get(
            "action",
            ""
        )

        try:

            p = urlparse(action)

            if p.netloc:

                d = (
                    p.netloc
                    .lower()
                    .split(":")[0]
                    .replace("www.", "")
                )

                if d and d != base_domain:
                    external_forms += 1

        except:
            pass


    return {
        "url_length": len(url),
        "domain_length": len(domain),
        "path_length": len(path),
        "query_length": len(query),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": sum(
            url.count(c)
            for c in "@$%&=!?;:'_-,"
        ),
        "url_entropy": entropy(url),
        "has_https": int(
            parsed.scheme.lower() == "https"
        ),
        "has_ip": 0,
        "suspicious_keyword_count": sum(
            1
            for word in SUSPICIOUS_WORDS
            if word in url.lower()
        ),
        "html_length": len(html),
        "text_length": len(text),
        "num_links": len(links),
        "external_links": external_links,
        "external_link_ratio":
            external_links / len(links)
            if links else 0,
        "form_count": len(forms),
        "password_field_count": password_fields,
        "has_login_form": has_login_form,
        "external_form_count": external_forms,
        "external_form_ratio":
            external_forms / len(forms)
            if forms else 0
    }


model = PhishXpert()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

model.eval()

scaler = joblib.load(
    SCALER_PATH
)


print("=" * 65)
print("PHISHXPERT — CUSTOM URL TEST")
print("=" * 65)

while True:

    url = input(
        "\nEnter URL (or type exit): "
    ).strip()

    if url.lower() == "exit":
        break

    if not url.startswith(
        ("http://", "https://")
    ):
        url = "https://" + url

    print("\nFetching website...")

    try:

        response = requests.get(
            url,
            timeout=(3, 7),
            headers={
                "User-Agent":
                "Mozilla/5.0"
            },
            allow_redirects=True
        )

        html = response.text

        features = extract_features(
            response.url,
            html
        )

        data = pd.DataFrame(
            [features]
        )[FEATURES]

        scaled = scaler.transform(
            data
        )

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

            prediction = "PHISHING"

            confidence = probability

        else:

            prediction = "BENIGN"

            confidence = 1 - probability


        print("\n" + "-" * 65)

        print(
            "Final URL:",
            response.url
        )

        print(
            "Website accessible: YES"
        )

        print(
            "HTML size:",
            len(html),
            "characters"
        )

        print(
            "Prediction:",
            prediction
        )

        print(
            "Confidence:",
            f"{confidence * 100:.2f}%"
        )

        print("-" * 65)


    except Exception as e:

        print(
            "\nERROR:",
            str(e)
        )
