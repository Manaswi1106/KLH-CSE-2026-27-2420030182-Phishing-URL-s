import os
from google import genai

from model_predictor import predict
from evidence_query import build_evidence_query
from rag_retriever import retrieve


client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY")
)


def build_reasoning_prompt(
    prediction,
    confidence,
    features,
    evidence
):

    evidence_text = ""

    for i, item in enumerate(evidence, 1):
        evidence_text += (
            f"\nEvidence {i}: {item['topic']}\n"
            f"{item['content']}\n"
        )

    prompt = f"""
You are a cybersecurity analysis assistant.

Analyze a webpage for possible phishing.

MODEL PREDICTION:
{prediction}

MODEL CONFIDENCE:
{confidence * 100:.2f}%

WEBPAGE EVIDENCE:
- URL length: {features['url_length']}
- Domain length: {features['domain_length']}
- Path length: {features['path_length']}
- Suspicious keyword count: {features['suspicious_keyword_count']}
- URL entropy: {features['url_entropy']}
- HTTPS: {features['has_https']}
- IP address used: {features['has_ip']}
- Number of links: {features['num_links']}
- External links: {features['external_links']}
- External link ratio: {features['external_link_ratio']}
- Forms: {features['form_count']}
- Password fields: {features['password_field_count']}
- Login form detected: {features['has_login_form']}
- External forms: {features['external_form_count']}

RETRIEVED CYBERSECURITY KNOWLEDGE:
{evidence_text}

TASK:

Explain the prediction using only the supplied webpage evidence
and retrieved cybersecurity knowledge.

Identify the strongest evidence and explain how it relates
to the retrieved knowledge.

Do not treat any single feature as proof of phishing.

Do not invent information about the website.

Give a concise security assessment.
"""

    return prompt


def analyze_with_gemini(features):

    prediction, confidence = predict(features)

    query = build_evidence_query(features)

    evidence = retrieve(query, top_k=3)

    evidence_text = ""

    for item in evidence:
        evidence_text += (
            f"{item['topic']}: "
            f"{item['content']}\n"
        )

    prompt = f"""
You are a cybersecurity analysis assistant.

Model prediction: {prediction}
Model confidence: {confidence * 100:.2f}%

Important webpage evidence:
- Suspicious URL keywords: {features['suspicious_keyword_count']}
- Login form: {features['has_login_form']}
- Password fields: {features['password_field_count']}
- External forms: {features['external_form_count']}
- External link ratio: {features['external_link_ratio']}
- URL entropy: {features['url_entropy']}
- HTTPS: {features['has_https']}
- IP address: {features['has_ip']}

Retrieved cybersecurity knowledge:
{evidence_text}

Explain the prediction using the supplied evidence.
Identify the strongest indicators and connect them to the retrieved knowledge.
Do not treat one feature alone as proof of phishing.
Do not invent facts.
Keep the explanation concise.
"""

    print("\nCalling Gemini...")

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt
    )

    print("Gemini response received.")

    return prediction, confidence, query, evidence, response.text

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

    prediction, confidence, query, evidence, explanation = (
        analyze_with_gemini(sample_features)
    )

    print("========== PHISHXPERT ==========")

    print("\nPrediction:")
    print(prediction)

    print("\nConfidence:")
    print(f"{confidence * 100:.2f}%")

    print("\nEvidence Query:")
    print(query)

    print("\nRAG Evidence:")

    for i, item in enumerate(evidence, 1):
        print(f"\n{i}. {item['topic']}")
        print(f"Similarity: {item['score']:.4f}")

    print("\n========== GEMINI REASONING ==========")
    print(explanation)