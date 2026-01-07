# src/web_text_extractor.py

import requests
from bs4 import BeautifulSoup
import re


# -------------------------------------------------------------
# 1. HTML laden
# -------------------------------------------------------------
def fetch_html(url: str) -> str:
    """
    Holt den HTML-Content einer Webseite. 
    Wir nutzen requests ohne JavaScript (leicht & schnell).
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()  # falls die Seite nicht erreichbar ist
    return response.text


# -------------------------------------------------------------
# 2. Sichtbaren Text extrahieren
# -------------------------------------------------------------
def extract_visible_text(html: str) -> str:
    """
    Entfernt unnötige Elemente:
    - Skripte
    - CSS
    - Navigation
    - Werbung
    Extrahiert nur den lesbaren Text.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Entferne unsichtbare Tags
    for tag in soup(["script", "style", "header", "footer", "nav", "aside"]):
        tag.decompose()

    # Hol den Text
    text = soup.get_text(separator="\n")

    # Entferne überflüssige Leerzeichen und Zeilen
    text = re.sub(r"\n\s*\n", "\n", text)  # doppelte Zeilen entfernen
    text = text.strip()

    return text


# -------------------------------------------------------------
# 3. End-to-End Funktion: URL -> sauberer Text
# -------------------------------------------------------------
def get_clean_text_from_url(url: str) -> str:
    html = fetch_html(url)
    text = extract_visible_text(html)
    return text
