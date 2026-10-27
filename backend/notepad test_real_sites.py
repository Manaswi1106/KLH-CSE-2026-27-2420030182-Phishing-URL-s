import requests
import time

sites = [
    "https://www.google.com/",
    "https://www.youtube.com/",
    "https://www.microsoft.com/",
    "https://www.github.com/",
    "https://www.wikipedia.org/"
]

for site in sites:

    print("\n" + "=" * 60)
    print("Testing:", site)

    try:

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"url": site},
            timeout=60
        )

        data = response.json()

        print("Prediction:", data.get("prediction"))
        print("Confidence:", data.get("confidence"), "%")
        print("Access:", data.get("access_status"))
        print("Features:", data.get("feature_mode"))

    except Exception as e:

        print("ERROR:", e)

    time.sleep(2)