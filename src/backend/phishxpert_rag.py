from model_predictor import predict
from evidence_query import build_evidence_query
from rag_retriever import retrieve


def analyze(features):

    prediction, confidence = predict(features)

    query = build_evidence_query(features)

    evidence = retrieve(query, top_k=3)

    return prediction, confidence, query, evidence


if __name__ == "__main__":

    sample_features = {
        "url_length": 60,
        "domain_length": 25,
        "path_length": 20,
        "query_length": 10,
        "num_dots": 3,
        "num_hyphens": 4,
        "num_underscores": 1,
        "num_digits": 5,
        "num_special_chars": 10,
        "url_entropy": 4.2,
        "has_https": 1,
        "has_ip": 0,
        "suspicious_keyword_count": 3,
        "html_length": 50000,
        "text_length": 5000,
        "num_links": 100,
        "external_links": 40,
        "external_link_ratio": 0.4,
        "form_count": 2,
        "password_field_count": 1,
        "has_login_form": 1,
        "external_form_count": 1,
        "external_form_ratio": 0.5
    }

    prediction, confidence, query, evidence = analyze(
        sample_features
    )

    print("========== PHISHXPERT ANALYSIS ==========")

    print("\nPrediction:")
    print(prediction)

    print("\nConfidence:")
    print(f"{confidence * 100:.2f}%")

    print("\nEvidence Query:")
    print(query)

    print("\nRetrieved Evidence:\n")

    for i, result in enumerate(evidence, 1):

        print(f"{i}. {result['topic']}")
        print(f"Similarity: {result['score']:.4f}")
        print(result["content"])
        print("-" * 60)