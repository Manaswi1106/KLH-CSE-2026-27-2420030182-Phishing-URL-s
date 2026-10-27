import os
import re
import pandas as pd
from urllib.parse import urlparse

BASE = r"D:\Documents\Aruna Grace\ALT\CompPhish\CompPhish"

CSV_PATH = os.path.join(BASE, "CompPhish_features.csv")
HTML_DIR = os.path.join(BASE, "All_HTML", "All_HTML")

OUTPUT = os.path.join(BASE, "CompPhish_features_v21.csv")
CHECKPOINT = os.path.join(BASE, "CompPhish_v21_checkpoint.csv")

MAX_HTML_SIZE = 2_000_000


def get_hostname(url):
    try:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        return urlparse(url).hostname or ""
    except Exception:
        return ""


def extract_new_features(url, html):
    hostname = get_hostname(url)

    parts = [x for x in hostname.split(".") if x]

    subdomain_count = max(len(parts) - 2, 0)

    at_symbol_count = url.count("@")

    encoded_char_count = len(
        re.findall(r"%[0-9a-fA-F]{2}", url)
    )

    percent_encoded_ratio = (
        encoded_char_count / len(url)
        if url else 0
    )

    punycode_flag = int(
        "xn--" in hostname.lower()
    )

    hostname_digits = sum(
        c.isdigit() for c in hostname
    )

    hostname_alnum = sum(
        c.isalnum() for c in hostname
    )

    hostname_digit_ratio = (
        hostname_digits / hostname_alnum
        if hostname_alnum else 0
    )

    hostname_hyphen_ratio = (
        hostname.count("-") / len(hostname)
        if hostname else 0
    )

    tokens = re.split(
        r"[^a-zA-Z0-9]+",
        url
    )

    url_token_count = len([
        x for x in tokens if x
    ])

    suspicious_tlds = {
        ".tk", ".ml", ".ga", ".cf",
        ".gq", ".top", ".xyz",
        ".click", ".work",
        ".support", ".zip", ".mov"
    }

    tld = ""

    if "." in hostname:
        tld = "." + hostname.rsplit(
            ".",
            1
        )[-1].lower()

    suspicious_tld_flag = int(
        tld in suspicious_tlds
    )

    # Fast HTML structural extraction
    html_lower = html.lower()

    iframe_count = len(
        re.findall(r"<iframe\b", html_lower)
    )

    script_count = len(
        re.findall(r"<script\b", html_lower)
    )

    input_count = len(
        re.findall(r"<input\b", html_lower)
    )

    hidden_element_count = 0

    hidden_element_count += len(
        re.findall(
            r"<[^>]+\bhidden\b",
            html_lower
        )
    )

    hidden_element_count += len(
        re.findall(
            r'aria-hidden\s*=\s*["\']true["\']',
            html_lower
        )
    )

    hidden_element_count += len(
        re.findall(
            r'display\s*:\s*none',
            html_lower
        )
    )

    hidden_element_count += len(
        re.findall(
            r'visibility\s*:\s*hidden',
            html_lower
        )
    )

    meta_refresh_flag = int(
        bool(
            re.search(
                r'<meta[^>]+http-equiv\s*=\s*["\']?refresh',
                html_lower
            )
        )
    )

    return {
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


print("Loading existing CompPhish features...")

df = pd.read_csv(CSV_PATH)

print(
    "Existing shape:",
    df.shape
)

print("\nIndexing HTML files...")

html_files = {}

for root, dirs, files in os.walk(HTML_DIR):

    for filename in files:

        if filename.lower().endswith(".txt"):

            stem = os.path.splitext(
                filename
            )[0]

            html_files[stem] = os.path.join(
                root,
                filename
            )

print(
    "HTML files found:",
    len(html_files)
)


# ---------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------

completed = {}

if os.path.exists(CHECKPOINT):

    print(
        "\nCheckpoint found. Loading..."
    )

    checkpoint_df = pd.read_csv(
        CHECKPOINT
    )

    for _, row in checkpoint_df.iterrows():

        serial = str(
            row["id"]
        )

        completed[serial] = {
            column: row[column]
            for column in checkpoint_df.columns
            if column != "id"
        }

    print(
        "Already processed:",
        len(completed)
    )


new_rows = []

total = len(df)

for i, row in df.iterrows():

    serial = str(
        row["id"]
    )

    if serial in completed:

        new_rows.append(
            {
                "id": serial,
                **completed[serial]
            }
        )

        continue

    html = ""

    path = html_files.get(serial)

    if path:

        try:

            size = os.path.getsize(path)

            if size <= MAX_HTML_SIZE:

                with open(
                    path,
                    "r",
                    encoding="utf-8",
                    errors="ignore"
                ) as f:

                    html = f.read()

        except Exception:

            html = ""

    features = extract_new_features(
        str(row["url"]),
        html
    )

    new_rows.append(
        {
            "id": serial,
            **features
        }
    )

    if (i + 1) % 500 == 0:

        checkpoint_df = pd.DataFrame(
            new_rows
        )

        checkpoint_df.to_csv(
            CHECKPOINT,
            index=False
        )

        print(
            f"Processed {i + 1}/{total}",
            flush=True
        )


new_df = pd.DataFrame(
    new_rows
)

df["id"] = df["id"].astype(str)
new_df["id"] = new_df["id"].astype(str)

new_df = new_df.set_index("id")
df_indexed = df.set_index("id")

new_df = new_df.reindex(
    df_indexed.index
)

new_df = new_df.reset_index()
df_indexed = df_indexed.reset_index()

new_df = new_df.drop(
    columns=["id"]
)

result = pd.concat(
    [
        df_indexed,
        new_df
    ],
    axis=1
)

result.to_csv(
    OUTPUT,
    index=False
)

print(
    "\n========== COMPLETE =========="
)

print(
    "Original shape:",
    df.shape
)

print(
    "New features:",
    14
)

print(
    "Final shape:",
    result.shape
)

print(
    "Saved:",
    OUTPUT
)