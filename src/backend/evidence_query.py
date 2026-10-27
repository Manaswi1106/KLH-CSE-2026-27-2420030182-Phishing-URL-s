def build_evidence_query(features):
    terms = []

    if features.get("has_login_form", 0) == 1:
        terms.append("login form")

    if features.get("password_field_count", 0) > 0:
        terms.append("password field credentials")

    if features.get("external_form_count", 0) > 0:
        terms.append("external form destination")

    if features.get("suspicious_keyword_count", 0) > 0:
        terms.append("suspicious URL keywords")

    if features.get("has_https", 0) == 0:
        terms.append("HTTP insecure connection")

    if features.get("has_ip", 0) == 1:
        terms.append("IP address URL")

    if features.get("url_entropy", 0) > 4:
        terms.append("URL obfuscation")

    if features.get("external_link_ratio", 0) > 0.5:
        terms.append("external links")

    if not terms:
        terms.append("phishing website security indicators")

    return " ".join(terms)


if __name__ == "__main__":
    sample_features = {
        "has_login_form": 1,
        "password_field_count": 1,
        "external_form_count": 1,
        "suspicious_keyword_count": 3,
        "has_https": 1,
        "has_ip": 0,
        "url_entropy": 4.2,
        "external_link_ratio": 0.7
    }

    query = build_evidence_query(sample_features)

    print("========== Evidence Query ==========")
    print(query)