# Text → Perspective API → Scores

# src/perspective_api_client.py

import os
import requests
import textwrap
from dotenv import load_dotenv

# .env laden
load_dotenv()

PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API_KEY")

API_URL = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"

# Welche Attribute wir analysieren wollen
ATTRIBUTES = {
    "TOXICITY": {},
    "INSULT": {},
    "IDENTITY_ATTACK": {},
    "PROFANITY": {},
    "SEXUAL_CONTENT": {},
}


# -----------------------------------------------------------
# 1. API Anfrage für kurzen Text
# -----------------------------------------------------------
def analyze_text_perspective(text: str) -> dict:
    """
    Sendet einen kurzen Text an die Perspective API.
    Gibt Scores für verschiedene Bias/Toxicity-Kategorien zurück.
    """

    if not PERSPECTIVE_API_KEY:
        raise ValueError("Fehler: Kein PERSPECTIVE_API_KEY in .env gefunden.")

    data = {
        "comment": {"text": text},
        "languages": ["de", "en"],
        "requestedAttributes": ATTRIBUTES,
        "doNotStore": True
    }

    response = requests.post(
        API_URL,
        params={"key": PERSPECTIVE_API_KEY},
        json=data
    )

    response.raise_for_status()
    result = response.json()

    scores = {}
    for attr in result.get("attributeScores", {}):
        scores[attr] = result["attributeScores"][attr]["summaryScore"]["value"]

    return scores


# -----------------------------------------------------------
# 2. Lange Texte aufteilen & komplett analysieren
# -----------------------------------------------------------
def analyze_long_text(text: str, block_size=800) -> dict:
    """
    Lange Texte in Blöcke teilen, jeden Block an Perspective senden,
    dann aggregieren (Durchschnitt + Max).
    """

    blocks = textwrap.wrap(text, block_size)

    block_results = []
    for block in blocks:
        try:
            scores = analyze_text_perspective(block)
            block_results.append(scores)
        except Exception as e:
            print("Fehler bei API-Aufruf:", e)

    # Aggregieren
    aggregated = {}

    for attr in ATTRIBUTES.keys():
        values = [br.get(attr, 0) for br in block_results]
        if values:
            aggregated[attr] = {
                "avg": sum(values) / len(values),
                "max": max(values),
            }
        else:
            aggregated[attr] = {"avg": 0, "max": 0}

    return {
        "blocks": len(block_results),
        "scores": aggregated
    }
