import requests
import torch
import torch.nn as nn
import pandas as pd
import joblib
import re
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


SITES = [
    ("Google", "https://www.google.com"),
    ("YouTube", "https://www.youtube.com"),
    ("Microsoft", "https://www.microsoft.com"),
    ("GitHub", "https://github.com"),
    ("Wikipedia", "https://www.wikipedia.org"),
    ("Amazon", "https://www.amazon.com"),
    ("Apple", "https://www.apple.com"),
    ("Netflix", "https://www.netflix.com"),
    ("LinkedIn", "https://www.linkedin.com"),
    ("Reddit", "https://www.reddit.com"),
    ("Yahoo", "https://www.yahoo.com"),
    ("Bing", "https://www.bing.com"),
    ("Adobe", "https://www.adobe.com"),
    ("Oracle", "https://www.oracle.com"),
    ("IBM", "https://www.ibm.com"),
    ("Khan Academy", "https://www.khanacademy.org"),
    ("Coursera", "https://www.coursera.org"),
    ("CNN", "https://www.cnn.com"),
    ("BBC", "https://www.bbc.com"),
    ("NASA", "https://www.nasa.gov")
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

    url_lower = url.lower()


    links = []

    forms = []


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


    feature_values = {

        "url_length": len(url),

        "domain_length": len(domain),

        "path_length": len(path),

        "query_length": len(query),

        "num_dots": url.count("."),

        "num_hyphens": url.count("-"),

        "num_underscores": url.count("_"),

        "num_digits": sum(
            c.isdigit()
            for c in url
        ),

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
            if word in url_lower
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

    return feature_values


print("=" * 60)
print("GENERALIZED PHISHXPERT — LEGITIMATE SITE TEST")
print("=" * 60)


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


results = []


for name, url in SITES:

    print(
        f"\nTesting {name}..."
    )

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

            prediction = "Phishing"

            confidence = probability

        else:

            prediction = "Benign"

            confidence = 1 - probability


        print(
            f"{name:15} → "
            f"{prediction:8} "
            f"{confidence * 100:.2f}%"
        )


        results.append(
            {
                "site": name,
                "prediction": prediction,
                "confidence": confidence * 100,
                "accessible": True
            }
        )


    except Exception as e:

        print(
            f"{name:15} → ERROR"
        )

        results.append(
            {
                "site": name,
                "prediction": "ERROR",
                "confidence": 0,
                "accessible": False
            }
        )


df = pd.DataFrame(
    results
)


successful = df[
    df["prediction"] != "ERROR"
]


benign_correct = (
    successful["prediction"]
    == "Benign"
).sum()


total_successful = len(
    successful
)


print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(
    "Total sites:",
    len(df)
)

print(
    "Successful:",
    total_successful
)

print(
    "Errors:",
    len(df) - total_successful
)

print(
    "Legitimate sites classified Benign:",
    benign_correct,
    "/",
    total_successful
)

if total_successful > 0:

    print(
        "Benign acceptance rate:",
        f"{benign_correct / total_successful * 100:.2f}%"
    )

    print(
        "False-positive rate:",
        f"{(1 - benign_correct / total_successful) * 100:.2f}%"
    )


print("=" * 60)