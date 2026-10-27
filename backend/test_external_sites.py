import requests
import time
import pandas as pd

SITES = [
    ("Google", "https://www.google.com/"),
    ("YouTube", "https://www.youtube.com/"),
    ("Microsoft", "https://www.microsoft.com/"),
    ("GitHub", "https://github.com/"),
    ("Wikipedia", "https://www.wikipedia.org/"),

    ("Amazon", "https://www.amazon.com/"),
    ("Apple", "https://www.apple.com/"),
    ("Netflix", "https://www.netflix.com/"),
    ("LinkedIn", "https://www.linkedin.com/"),
    ("Reddit", "https://www.reddit.com/"),

    ("Yahoo", "https://www.yahoo.com/"),
    ("Bing", "https://www.bing.com/"),
    ("Adobe", "https://www.adobe.com/"),
    ("Oracle", "https://www.oracle.com/"),
    ("IBM", "https://www.ibm.com/"),

    ("Khan Academy", "https://www.khanacademy.org/"),
    ("Coursera", "https://www.coursera.org/"),
    ("CNN", "https://www.cnn.com/"),
    ("BBC", "https://www.bbc.com/"),
    ("NASA", "https://www.nasa.gov/")
]


results = []

print("\n========== PHISHXPERT EXTERNAL VALIDATION ==========\n")

for name, url in SITES:

    print(f"Testing: {name}")
    print(f"URL: {url}")

    try:

        start = time.time()

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"url": url},
            timeout=20
        )

        elapsed = time.time() - start

        data = response.json()

        prediction = data.get(
            "prediction",
            "Unknown"
        )

        confidence = data.get(
            "confidence",
            0
        )

        access_status = data.get(
            "access_status",
            "Unknown"
        )

        feature_mode = data.get(
            "feature_mode",
            "Unknown"
        )

        print(
            f"Prediction: {prediction}"
        )

        print(
            f"Confidence: {confidence:.2f}%"
        )

        print(
            f"Access: {access_status}"
        )

        print(
            f"Features: {feature_mode}"
        )

        print(
            f"Time: {elapsed:.2f}s"
        )

        results.append({
            "website": name,
            "url": url,
            "prediction": prediction,
            "confidence": confidence,
            "access_status": access_status,
            "feature_mode": feature_mode,
            "response_time": round(
                elapsed,
                2
            )
        })

    except Exception as e:

        print(
            f"ERROR: {type(e).__name__}: {e}"
        )

        results.append({
            "website": name,
            "url": url,
            "prediction": "ERROR",
            "confidence": 0,
            "access_status": "ERROR",
            "feature_mode": "ERROR",
            "response_time": 0
        })

    print("-" * 50)


df = pd.DataFrame(results)

df.to_csv(
    "external_validation_results.csv",
    index=False
)


benign_count = len(df)

phishing_predictions = (
    df["prediction"] == "Phishing"
).sum()

benign_predictions = (
    df["prediction"] == "Benign"
).sum()

error_count = (
    df["prediction"] == "ERROR"
).sum()


print("\n========== SUMMARY ==========")

print(
    f"Total websites tested : {benign_count}"
)

print(
    f"Predicted Benign      : {benign_predictions}"
)

print(
    f"Predicted Phishing    : {phishing_predictions}"
)

print(
    f"Errors                : {error_count}"
)

if benign_count - error_count > 0:

    valid = df[
        df["prediction"] != "ERROR"
    ]

    benign_rate = (
        (valid["prediction"] == "Benign").sum()
        / len(valid)
        * 100
    )

    print(
        f"Benign classification rate: "
        f"{benign_rate:.2f}%"
    )


print(
    "\nResults saved to:"
)

print(
    "external_validation_results.csv"
)

print(
    "\n========== COMPLETE =========="
)