import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = r"D:\Documents\Aruna Grace\ALT\PhishXpert"
KNOWLEDGE_FILE = os.path.join(
    BASE_DIR,
    "knowledge_base",
    "phishing_knowledge.txt"
)

with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
    knowledge = f.read()

sections = re.split(r"\n(?=TOPIC:)", knowledge)

documents = []

for section in sections:
    section = section.strip()

    if section:
        documents.append(section)

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2)
)

document_vectors = vectorizer.fit_transform(documents)


def retrieve(query, top_k=3):

    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        document_vectors
    )[0]

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append({
            "topic": documents[index].split("\n")[0].replace(
                "TOPIC: ",
                ""
            ),
            "score": float(similarities[index]),
            "content": documents[index]
        })

    return results


if __name__ == "__main__":

    query = (
        "suspicious login page password field "
        "external form URL credentials"
    )

    results = retrieve(query, top_k=3)

    print("\n========== RAG Retrieval Test ==========\n")

    for i, result in enumerate(results, 1):

        print(
            f"{i}. {result['topic']}"
        )

        print(
            f"Similarity: {result['score']:.4f}"
        )

        print(
            result["content"]
        )

        print("-" * 60)