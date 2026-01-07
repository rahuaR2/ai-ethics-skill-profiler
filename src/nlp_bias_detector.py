# src/nlp_bias_detector.py

import re
from dataclasses import dataclass, asdict
from typing import List, Dict


# -------------------------------------------------------------------
# Datenstruktur für Treffer
# -------------------------------------------------------------------
@dataclass
class BiasHit:
    pattern: str
    match: str
    explanation: str
    suggestion: str


# -------------------------------------------------------------------
# Kategorietexte (für UI)
# -------------------------------------------------------------------
CATEGORY_DESCRIPTIONS = {
    "gender": "Geschlechtsspezifische oder stereotypisierende Formulierungen.",
    "age": "Altersspezifische Einschränkungen oder Bevorzugungen.",
    "ableism": "Ableistische oder problematische Ableismus-Formulierungen.",
    "migration": "Rassifizierende, ethnisierende oder othering-Sprache.",
    "class": "Soziale Herkunft, Klasse oder abwertende Beschreibungen.",
}


# -------------------------------------------------------------------
# Muster & Vorschläge pro Kategorie (MVP, erweiterbar)
# -------------------------------------------------------------------
BIAS_PATTERNS: Dict[str, List[Dict]] = {
    "gender": [
        {
            "pattern": r"\b(junger?|junge)\s+dynamisch\w*",
            "explanation": "Betonung von Jugend und Dynamik kann ältere Personen ausschließen.",
            "suggestion": "z.B. „engagiertes, motiviertes Team“."
        },
        {
            "pattern": r"\bsekretärin\b",
            "explanation": "Berufsbezeichnung nur in weiblicher Form.",
            "suggestion": "z.B. „Sekretariatskraft“ oder „Assistenz (m/w/d)“."
        },
        {
            "pattern": r"\bmuttersprachler(in)?\b|\bmuttersprache\b",
            "explanation": "„Muttersprache“ erzeugt unnötige Hürden für mehrsprachige Personen.",
            "suggestion": "z.B. „sehr gute Deutschkenntnisse in Wort und Schrift“."
        },
    ],
    "age": [
        {
            "pattern": r"\bjung(e|er|es)?\s+team\b",
            "explanation": "Fokus auf „junges Team“ kann ältere Bewerbende ausschließen.",
            "suggestion": "z.B. „diverses, altersgemischtes Team“."
        },
        {
            "pattern": r"\b(max\.?|höchstens)\s*\d{2}\s*jahre\b",
            "explanation": "Altersobergrenzen sind in vielen Fällen nicht AGG-konform.",
            "suggestion": "Fokus auf Erfahrung/Kompetenzen statt Altersgrenzen."
        },
    ],
    "ableism": [
        {
            "pattern": r"\btrotz\s+behinderung\b",
            "explanation": "Formulierung stellt Behinderung als Defizit dar.",
            "suggestion": "z.B. „Menschen mit Behinderung“ ohne „trotz“."
        },
        {
            "pattern": r"\bgesund(?:e|er|es)?\s+(geist|körper)\b",
            "explanation": "Gegenüberstellung „gesund“ vs. „nicht gesund“ kann stigmatisieren.",
            "suggestion": "Neutralere Beschreibungen verwenden, z.B. „belastbar“ nur kontextbezogen."
        },
    ],
    "migration": [
        {
            "pattern": r"\bdeutsch(?:e|er|en)?\s+ohne\s+migrationshintergrund\b",
            "explanation": "Expliziter Ausschluss von Menschen mit Migrationsgeschichte.",
            "suggestion": "Nur relevante Sprach- oder Rechtskenntnisse klar benennen."
        },
        {
            "pattern": r"\bnicht\s+deutsch\b|\bnicht\-deutsch\b",
            "explanation": "Stellt „deutsch“ als Norm und andere als Abweichung dar.",
            "suggestion": "Konkrete Anforderungen formulieren statt Zuschreibungen."
        },
    ],
    "class": [
        {
            "pattern": r"\bbildungsfern(e|er|es)?\b",
            "explanation": "„bildungsfern“ ist stigmatisierend und klassenbezogen abwertend.",
            "suggestion": "Konkrete Zugangsbarrieren benennen statt Abwertung."
        },
        {
            "pattern": r"\bunterschicht\b|\bunterschichten\b",
            "explanation": "Abwertende Beschreibung sozialer Lage.",
            "suggestion": "Neutralere Begriffe wie „Haushalte mit geringem Einkommen“."
        },
    ],
}


def _normalize_text(text: str) -> str:
    return text.lower()


def _find_hits_in_category(text: str, category: str) -> List[BiasHit]:
    hits: List[BiasHit] = []
    for entry in BIAS_PATTERNS.get(category, []):
        pattern = entry["pattern"]
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            hits.append(
                BiasHit(
                    pattern=pattern,
                    match=m.group(0),
                    explanation=entry["explanation"],
                    suggestion=entry["suggestion"],
                )
            )
    return hits


def analyze_text_bias(text: str) -> Dict:
    """
    Analysiert Text auf potenziell diskriminierende Muster.
    Rückgabeformat ist für das Streamlit-Dashboard optimiert.
    """
    norm = _normalize_text(text)

    results = {}
    total_hits = 0

    for category in BIAS_PATTERNS.keys():
        hits = _find_hits_in_category(norm, category)
        total_hits += len(hits)
        results[category] = {
            "description": CATEGORY_DESCRIPTIONS.get(category, ""),
            "n_hits": len(hits),
            "hits": [asdict(h) for h in hits],
        }

    # einfacher Risikoscore: 0–1
    max_theoretical = 10  # skalieren
    overall_score = min(1.0, total_hits / max_theoretical)

    return {
        "overall_score": overall_score,
        "total_hits": total_hits,
        "categories": results,
    }
