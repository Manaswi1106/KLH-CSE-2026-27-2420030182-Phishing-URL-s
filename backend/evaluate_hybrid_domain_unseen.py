import os
import re
import joblib
import warnings
import numpy as np
import pandas as pd
import torch

from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
    log_loss,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

COMP_DIR = r"D:\Documents\Aruna Grace\ALT\CompPhish\CompPhish"

FEATURE_FILE = os.path.join(
    COMP_DIR,
    "CompPhish_features.csv"
)

HTML_DIR = os.path.join(
    COMP_DIR,
    "All_HTML"
)

MODEL_FILE = "models/hybrid_model.pkl"

OUTPUT_DIR = "evaluation"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

MODEL_NAME = "distilbert-base-uncased"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PHISHXPERT HYBRID — DOMAIN-UNSEEN COMPPHISH EVALUATION")
print("=" * 70)


# ============================================================
# LOAD HYBRID MODEL
# ============================================================

print("\nLoading Hybrid model...")

# IMPORTANT:
# train_hybrid.py used joblib.dump()
# Therefore we MUST use joblib.load()

model = joblib.load(
    MODEL_FILE
)

print("Hybrid model loaded successfully.")


# ============================================================
# LOAD DISTILBERT
# ============================================================

print("\nLoading DistilBERT...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

bert_model = AutoModel.from_pretrained(
    MODEL_NAME
)

bert_model.eval()

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

bert_model.to(device)

print("Device:", device)


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

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
    "external_form_ratio",
]


# ============================================================
# SUSPICIOUS URL KEYWORDS
# ============================================================

SUSPICIOUS_KEYWORDS = [
    "login",
    "signin",
    "verify",
    "verification",
    "account",
    "secure",
    "update",
    "confirm",
    "password",
    "bank",
    "paypal",
    "credential",
    "wallet",
    "authenticate",
    "security",
    "unlock",
    "suspend",
]


# ============================================================
# ENTROPY
# ============================================================

def entropy(text):

    if not text:
        return 0.0

    counts = {}

    for char in text:

        counts[char] = (
            counts.get(char, 0) + 1
        )

    n = len(text)

    return -sum(
        (count / n)
        * np.log2(count / n)
        for count in counts.values()
    )


# ============================================================
# DOMAIN EXTRACTION
# ============================================================

def extract_domain(url):

    url = str(url).lower()

    url = re.sub(
        r"^https?://",
        "",
        url
    )

    url = url.split("/")[0]

    url = url.split("?")[0]

    return url


# ============================================================
# URL + HTML FEATURE EXTRACTION
# ============================================================

def extract_url_features(
    url,
    html
):

    url = str(url)

    parsed_domain = extract_domain(
        url
    )

    lower_url = url.lower()

    # --------------------------------------------------------
    # URL FEATURES
    # --------------------------------------------------------

    url_length = len(url)

    domain_length = len(
        parsed_domain
    )

    remainder = url

    if "://" in remainder:

        remainder = remainder.split(
            "://",
            1
        )[1]

    path_query = remainder

    if "/" in path_query:

        path_query = path_query.split(
            "/",
            1
        )[1]

    else:

        path_query = ""

    if "?" in path_query:

        path = path_query.split(
            "?",
            1
        )[0]

        query = path_query.split(
            "?",
            1
        )[1]

    else:

        path = path_query

        query = ""

    path_length = len(path)

    query_length = len(query)

    num_dots = url.count(".")

    num_hyphens = url.count("-")

    num_underscores = url.count("_")

    num_digits = sum(
        char.isdigit()
        for char in url
    )

    num_special_chars = sum(
        not char.isalnum()
        for char in url
    )

    url_entropy = entropy(
        url
    )

    has_https = int(
        lower_url.startswith(
            "https://"
        )
    )

    has_ip = int(
        bool(
            re.search(
                r"(?:\d{1,3}\.){3}\d{1,3}",
                parsed_domain
            )
        )
    )

    suspicious_keyword_count = sum(
        keyword in lower_url
        for keyword in SUSPICIOUS_KEYWORDS
    )


    # --------------------------------------------------------
    # HTML FEATURES
    # --------------------------------------------------------

    html = str(html)

    html_length = len(html)

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    visible_text = soup.get_text(
        " ",
        strip=True
    )

    text_length = len(
        visible_text
    )

    # --------------------------------------------------------
    # LINKS
    # --------------------------------------------------------

    links = soup.find_all(
        "a"
    )

    num_links = len(
        links
    )

    external_links = 0

    for link in links:

        href = link.get(
            "href"
        )

        if not href:
            continue

        href = str(
            href
        ).lower()

        if href.startswith(
            (
                "http://",
                "https://"
            )
        ):

            link_domain = extract_domain(
                href
            )

            if (
                link_domain
                and parsed_domain
                and link_domain != parsed_domain
            ):

                external_links += 1

    if num_links > 0:

        external_link_ratio = (
            external_links
            / num_links
        )

    else:

        external_link_ratio = 0.0


    # --------------------------------------------------------
    # FORMS
    # --------------------------------------------------------

    forms = soup.find_all(
        "form"
    )

    form_count = len(
        forms
    )

    password_field_count = len(
        soup.find_all(
            "input",
            attrs={
                "type": re.compile(
                    "^password$",
                    re.I
                )
            }
        )
    )

    has_login_form = int(
        password_field_count > 0
    )

    external_form_count = 0

    for form in forms:

        action = form.get(
            "action"
        )

        if not action:
            continue

        action = str(
            action
        ).lower()

        if action.startswith(
            (
                "http://",
                "https://"
            )
        ):

            action_domain = extract_domain(
                action
            )

            if (
                action_domain
                and parsed_domain
                and action_domain != parsed_domain
            ):

                external_form_count += 1

    if form_count > 0:

        external_form_ratio = (
            external_form_count
            / form_count
        )

    else:

        external_form_ratio = 0.0


    # --------------------------------------------------------
    # RETURN EXACTLY 23 FEATURES
    # --------------------------------------------------------

    return [

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
        external_form_ratio,
    ]


# ============================================================
# DISTILBERT EMBEDDING
# ============================================================

def get_bert_embedding(
    text
):

    text = str(text)

    if not text.strip():

        return np.zeros(
            768,
            dtype=np.float32
        )

    # IMPORTANT:
    # Match existing bert_encoder.py behavior.
    # The first 512 characters are passed to tokenizer.

    text = text[:512]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = bert_model(
            **inputs
        )

        embedding = (
            outputs
            .last_hidden_state[
                :,
                0,
                :
            ]
            .cpu()
            .numpy()
            .flatten()
        )

    return embedding.astype(
        np.float32
    )


# ============================================================
# LOAD COMPPHISH DATA
# ============================================================

print("\nLoading CompPhish features...")

df = pd.read_csv(
    FEATURE_FILE
)

print(
    "Total records:",
    len(df)
)


# ============================================================
# DOMAIN EXTRACTION
# ============================================================

def get_domain_from_url(
    url
):

    url = str(url).lower()

    url = re.sub(
        r"^https?://",
        "",
        url
    )

    url = url.split("/")[0]

    url = url.split("?")[0]

    url = url.split(":")[0]

    return url


df["domain"] = df[
    "url"
].apply(
    get_domain_from_url
)

df = df[
    df["domain"].notna()
    &
    (
        df["domain"]
        .astype(str)
        .str.len()
        > 0
    )
].copy()

print(
    "Records with valid domains:",
    len(df)
)


# ============================================================
# DOMAIN-LEVEL SPLIT
# ============================================================

print(
    "\nCreating domain-separated test set..."
)

rng = np.random.RandomState(
    42
)

domains = df[
    "domain"
].unique()

rng.shuffle(
    domains
)

# Existing audit:
# Training domains = 7308
# Test domains = 1828

test_domain_count = 1828

test_domains = set(
    domains[
        :test_domain_count
    ]
)

test_df = df[
    df["domain"].isin(
        test_domains
    )
].copy()

print(
    "Domain-unseen test records:",
    len(test_df)
)

print(
    "Unique test domains:",
    test_df["domain"].nunique()
)


# ============================================================
# VERIFY ZERO DOMAIN OVERLAP
# ============================================================

test_domain_set = set(
    test_df["domain"]
)

remaining_domains = set(
    domains[
        test_domain_count:
    ]
)

overlap = (
    test_domain_set
    & remaining_domains
)

print(
    "Domain overlap:",
    len(overlap)
)


# ============================================================
# FIND HTML FILES
# ============================================================

print(
    "\nFinding HTML files..."
)


def find_html_file(
    record_id
):

    record_id = str(
        record_id
    )

    direct_path = os.path.join(
        HTML_DIR,
        record_id + ".txt"
    )

    if os.path.exists(
        direct_path
    ):

        return direct_path

    # Recursive search
    filename = (
        record_id
        + ".txt"
    )

    for root, _, files in os.walk(
        HTML_DIR
    ):

        if filename in files:

            return os.path.join(
                root,
                filename
            )

    return None


test_df["html_path"] = test_df[
    "id"
].apply(
    find_html_file
)

test_df = test_df[
    test_df["html_path"].notna()
].copy()

print(
    "Domain-unseen test records with HTML:",
    len(test_df)
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print(
    "\nTest label distribution:"
)

print(
    test_df[
        "label"
    ].value_counts()
)


# ============================================================
# BUILD HYBRID FEATURES
# ============================================================

print(
    "\nGenerating Hybrid features..."
)

X = []

y = []

ids = []

total = len(
    test_df
)

for index, row in enumerate(
    test_df.itertuples(
        index=False
    ),
    start=1
):

    try:

        # ----------------------------------------------------
        # READ HTML
        # ----------------------------------------------------

        with open(
            row.html_path,
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            html = f.read()


        # ----------------------------------------------------
        # 23 ENGINEERED FEATURES
        # ----------------------------------------------------

        engineered_features = (
            extract_url_features(
                row.url,
                html
            )
        )


        # ----------------------------------------------------
        # EXTRACT WEBPAGE TEXT
        # ----------------------------------------------------

        soup = BeautifulSoup(
            html,
            "html.parser"
        )

        text = soup.get_text(
            " ",
            strip=True
        )


        # ----------------------------------------------------
        # 768-DIMENSION DISTILBERT
        # ----------------------------------------------------

        bert_embedding = (
            get_bert_embedding(
                text
            )
        )


        # ----------------------------------------------------
        # HYBRID FUSION
        # ----------------------------------------------------

        combined = np.concatenate(
            [
                np.asarray(
                    engineered_features,
                    dtype=np.float32
                ),
                bert_embedding
            ]
        )


        # Verify dimensions
        if len(combined) != 791:

            raise ValueError(
                f"Expected 791 features, "
                f"got {len(combined)}"
            )


        X.append(
            combined
        )

        y.append(
            int(row.label)
        )

        ids.append(
            row.id
        )


        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        if index % 100 == 0:

            print(
                f"Processed {index}/{total}"
            )


    except Exception as e:

        print(
            f"ERROR ID {row.id}: {e}"
        )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X = np.asarray(
    X,
    dtype=np.float32
)

y = np.asarray(
    y,
    dtype=np.int32
)

print(
    "\nFeature matrix:"
)

print(
    "Shape:",
    X.shape
)

print(
    "Expected dimensions: 791"
)

print(
    "Samples:",
    len(y)
)


# ============================================================
# MODEL PREDICTION
# ============================================================

print(
    "\nRunning Hybrid model prediction..."
)

probabilities = (
    model.predict_proba(X)[:, 1]
)

predictions = (
    probabilities >= 0.50
).astype(int)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y,
    predictions
)

precision = precision_score(
    y,
    predictions,
    zero_division=0
)

recall = recall_score(
    y,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y,
    predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    y,
    probabilities
)

pr_auc = average_precision_score(
    y,
    probabilities
)

mcc = matthews_corrcoef(
    y,
    predictions
)

kappa = cohen_kappa_score(
    y,
    predictions
)

tn, fp, fn, tp = confusion_matrix(
    y,
    predictions
).ravel()

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)

logloss = log_loss(
    y,
    probabilities
)


# ============================================================
# RESULTS
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "DOMAIN-UNSEEN HYBRID RESULTS"
)

print(
    "=" * 70
)

print(
    f"Accuracy    : {accuracy:.4f}"
)

print(
    f"Precision   : {precision:.4f}"
)

print(
    f"Recall      : {recall:.4f}"
)

print(
    f"F1          : {f1:.4f}"
)

print(
    f"ROC-AUC     : {roc_auc:.4f}"
)

print(
    f"PR-AUC      : {pr_auc:.4f}"
)

print(
    f"MCC         : {mcc:.4f}"
)

print(
    f"Kappa       : {kappa:.4f}"
)

print(
    f"Specificity : {specificity:.4f}"
)

print(
    f"Log Loss    : {logloss:.4f}"
)

print(
    "\nConfusion Matrix:"
)

print(
    confusion_matrix(
        y,
        predictions
    )
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame(
    [
        {
            "evaluation":
                "CompPhish_domain_unseen",

            "samples":
                len(y),

            "benign":
                int(
                    (y == 0).sum()
                ),

            "phishing":
                int(
                    (y == 1).sum()
                ),

            "accuracy":
                accuracy,

            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "roc_auc":
                roc_auc,

            "pr_auc":
                pr_auc,

            "mcc":
                mcc,

            "kappa":
                kappa,

            "specificity":
                specificity,

            "log_loss":
                logloss,

            "tn":
                tn,

            "fp":
                fp,

            "fn":
                fn,

            "tp":
                tp,
        }
    ]
)

metrics_file = os.path.join(
    OUTPUT_DIR,
    "hybrid_domain_unseen_metrics.csv"
)

metrics.to_csv(
    metrics_file,
    index=False
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame(
    {
        "id":
            ids,

        "label":
            y,

        "probability_phishing":
            probabilities,

        "prediction":
            predictions,
    }
)

predictions_file = os.path.join(
    OUTPUT_DIR,
    "hybrid_domain_unseen_predictions.csv"
)

predictions_df.to_csv(
    predictions_file,
    index=False
)


# ============================================================
# FINAL STATUS
# ============================================================

print(
    "\nSaved:"
)

print(
    metrics_file
)

print(
    predictions_file
)

print(
    "\n" + "=" * 70
)

print(
    "DOMAIN-UNSEEN EVALUATION COMPLETE"
)

print(
    "=" * 70
)