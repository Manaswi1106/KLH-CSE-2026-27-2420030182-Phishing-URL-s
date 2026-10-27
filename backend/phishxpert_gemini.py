import os
from google import genai

from model_predictor import predict
from evidence_query import build_evidence_query
from rag_retriever import retrieve


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def analyze_with_gemini(features):

    prediction, confidence = predict(features)

    query = build_evidence_query(features)
    evidence = retrieve(query, top_k=3)

    evidence_text = "\n\n".join(
        f"Topic: {item['topic']}\n{item['content']}"
        for item in evidence
    )

    prompt = f"""
You are an explainable cybersecurity analysis assistant.

Analyze a webpage phishing detection result.

Machine Learning Prediction:
{prediction}

Model Confidence:
{confidence:.2%}

Detected Webpage Features:
{features}

Retrieved Cybersecurity Knowledge:
{evidence_text}

Provide a concise explanation containing:

1. Prediction
2. Main evidence from the webpage features
3. Relevant cybersecurity knowledge
4. Why the evidence supports the prediction
5. A short final risk explanation

Do not invent evidence that is not provided.
Do not claim that one feature alone proves phishing.
"""

    response = client.models.generate_content(
    model="gemini-3.5-flash",
    contents=prompt,
    config={
        "automatic_function_calling": {
            "disable": True
        }
    }
)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "query": query,
        "evidence": evidence,
        "explanation": response.text
    }


if __name__ == "__main__":

    test_features = {
        "url_length": 82,
        "domain_length": 34,
        "path_length": 31,
        "query_length": 8,
        "num_dots": 5,
        "num_hyphens": 6,
        "num_underscores": 2,
        "num_digits": 8,
        "num_special_chars": 15,
        "url_entropy": 4.8,
        "has_https": 0,
        "has_ip": 0,
        "suspicious_keyword_count": 4,
        "html_length": 45000,
        "text_length": 3500,
        "num_links": 80,
        "external_links": 65,
        "external_link_ratio": 0.81,
        "form_count": 2,
        "password_field_count": 1,
        "has_login_form": 1,
        "external_form_count": 1,
        "external_form_ratio": 0.5
    }

    print("\nRunning PhishXpert analysis...", flush=True)

    result = analyze_with_gemini(test_features)

    print("========== PHISHXPERT ==========")
    print("Prediction:", result["prediction"])
    print(f"Confidence: {result['confidence']:.2%}")

    print("\n========== RAG EVIDENCE ==========")

    for item in result["evidence"]:
        print(f"\n{item['topic']} ({item['score']:.4f})")
        print(item["content"])

    print("\n========== GEMINI EXPLANATION ==========")
    print(result["explanation"])