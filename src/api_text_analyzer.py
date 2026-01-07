# src/api_text_analyzer.py

import os
import textwrap
import json
from dotenv import load_dotenv
from openai import OpenAI

# Umgebung laden
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY wurde nicht gefunden. Prüfe deine .env Datei.")

client = OpenAI(api_key=OPENAI_API_KEY)


# -----------------------------------------------------------
# 1. Moderation API (OpenAI v1+)
# -----------------------------------------------------------

# -----------------------------------------------------------
# 1. Moderation API (OpenAI v1+)
# -----------------------------------------------------------

def analyze_moderation(text: str) -> dict:
    """
    Analysiert Hate, Harassment, Sexual Content etc. mit der neuen
    Moderation API (OpenAI >= 1.0).
    """
    response = client.moderations.create(
        model="omni-moderation-latest",
        input=text,
        timeout=20
    )

    result = response.results[0]

    # WICHTIG: .model_dump() → CategoryScores als dict ausgeben
    return {
        "categories": result.categories,
        "scores": result.category_scores.model_dump(),
        "flagged": result.flagged
    }


# ⬇⬇⬇ HIER DIE TIMEOUT-SICHERE FUNKTION (korrekt repariert) ⬇⬇⬇

import time
from openai import APITimeoutError, APIError

def analyze_moderation_long_text(text: str, block_size: int = 500) -> dict:
    """
    Zerlegt Text in kleine Blöcke und nutzt automatische Retry-Logik,
    um APITimeoutError zu verhindern.
    """
    blocks = textwrap.wrap(text, block_size)
    results = []

    for i, block in enumerate(blocks):
        if not block.strip():
            continue

        retry_attempts = 3
        for attempt in range(retry_attempts):
            try:
                res = analyze_moderation(block)
                results.append(res)
                break

            except APITimeoutError:
                print(f"⚠️ Timeout bei Block {i+1}. Versuch {attempt+1}/3…")
                time.sleep(1)

            except APIError as e:
                print(f"⚠️ API-Fehler bei Block {i+1}: {e}. Neuer Versuch…")
                time.sleep(1)

        else:
            print(f"❌ Block {i+1} wurde übersprungen (API reagiert nicht).")

    # Wenn gar keine Ergebnisse:
    if not results:
        return {"blocks": 0, "scores": {}}

    # Scores vom ersten Block → jetzt EIN DICT
    first_scores = results[0]["scores"]
    aggregated = {}

    # ---- WICHTIG: richtige Schleife, da dict → keys() inkl. Fix ----
    for key in first_scores.keys():
        vals = [
            float(r["scores"][key])
            for r in results
        ]
        aggregated[key] = {
            "avg": sum(vals) / len(vals),
            "max": max(vals)
        }

    return {
        "blocks": len(results),
        "scores": aggregated
    }


# -----------------------------------------------------------
# 2. GPT Bias Analyse – Soziologisch + erklärend
# -----------------------------------------------------------

BIAS_SYSTEM_PROMPT = """
Du bist eine Expertin für diskriminierungssensible Sprache, Antidiskriminierung und Soziologie.
Analysiere den Text nach:

- Genderbias
- Rassismus / Ethnisierung
- Ableismus
- Ageism
- Klassismus

Gib IMMER folgendes JSON zurück:

{
  "overall_risk": "low|medium|high",
  "dimensions": {
    "gender": { "risk": "...", "examples": [] },
    "race": { "risk": "...", "examples": [] },
    "disability": { "risk": "...", "examples": [] },
    "age": { "risk": "...", "examples": [] },
    "class": { "risk": "...", "examples": [] }
  },
  "comments": []
}
"""

def analyze_bias_gpt(text: str) -> dict:
    """
    Nutzt die neue OpenAI ChatCompletion API (OpenAI >= 1.0)
    für eine Bias-Analyse mit JSON-Ausgabe.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": BIAS_SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ],
        temperature=0.0
    )

    content = response.choices[0].message.content

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {
            "overall_risk": "unknown",
            "dimensions": {},
            "comments": [
                "JSON konnte nicht geparst werden.",
                content
            ]
        }

    return data
