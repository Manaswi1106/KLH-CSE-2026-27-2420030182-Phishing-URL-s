import re
from bs4 import BeautifulSoup

def clean_webpage_text(html):
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)

    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9 ]", " ", text)

    return text.lower().strip()