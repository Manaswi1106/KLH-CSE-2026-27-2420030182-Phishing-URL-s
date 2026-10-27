import requests

sites = [
    "https://www.google.com/",
    "https://www.youtube.com/",
    "https://www.microsoft.com/",
    "https://github.com/",
    "https://www.wikipedia.org/"
]

for url in sites:
    print("\nTesting:", url)

    try:
        r = requests.post(
            "http://127.0.0.1:8000/predict",
            json={"url": url},
            timeout=20
        )

        print(r.json())

    except Exception as e:
        print("ERROR:", e)