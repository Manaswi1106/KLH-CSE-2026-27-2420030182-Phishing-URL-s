from urllib.parse import urlparse


# Common legitimate domains for live-demo verification.
# This is supporting evidence, NOT the ML training dataset.
TRUSTED_DOMAINS = {
    "google.com",
    "youtube.com",
    "github.com",
    "wikipedia.org",
    "microsoft.com",
    "amazon.com",
    "apple.com",
    "netflix.com",
    "linkedin.com",
    "reddit.com",
    "yahoo.com",
    "bing.com",
    "adobe.com",
    "oracle.com",
    "ibm.com",
    "khanacademy.org",
    "coursera.org",
    "cnn.com",
    "bbc.com",
    "nasa.gov"
}


def normalize_domain(url):
    url = url.strip()

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        hostname = urlparse(url).hostname

        if not hostname:
            return ""

        hostname = hostname.lower().rstrip(".")

        if hostname.startswith("www."):
            hostname = hostname[4:]

        return hostname

    except Exception:
        return ""


def get_registered_domain(hostname):
    """
    Simple registered-domain approximation for the common
    domains used in this project.
    """

    parts = hostname.split(".")

    if len(parts) < 2:
        return hostname

    two_level_suffixes = {
        "co.uk",
        "org.uk",
        "ac.uk",
        "gov.uk",
        "co.in",
        "org.in",
        "ac.in",
        "gov.in",
        "co.jp",
        "co.nz",
        "com.au",
        "net.au",
        "org.au"
    }

    suffix = ".".join(parts[-2:])

    if suffix in two_level_suffixes and len(parts) >= 3:
        return ".".join(parts[-3:])

    return ".".join(parts[-2:])


def verify_domain(url):

    hostname = normalize_domain(url)

    if not hostname:
        return {
            "status": "unknown",
            "hostname": "",
            "registered_domain": "",
            "trusted": False
        }

    registered_domain = get_registered_domain(
        hostname
    )

    trusted = (
        registered_domain
        in TRUSTED_DOMAINS
    )

    if trusted:
        status = "trusted"
    else:
        status = "unknown"

    return {
        "status": status,
        "hostname": hostname,
        "registered_domain": registered_domain,
        "trusted": trusted
    }