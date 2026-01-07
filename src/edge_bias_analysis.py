# src/edge_bias_analysis.py

import numpy as np
import cv2


# -------------------------------------------------------------------
# Vereinfachte Merkmalsextraktion
# Ziel: stabile, messbare Werte für Robustness & Bias
# -------------------------------------------------------------------
def extract_features(image):
    """
    Extrahiert einfache Bildmerkmale:
    - Helligkeit
    - Kontrast
    - Kantenstärke (Canny)
    - Farbsättigung
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    brightness = np.mean(gray)
    contrast = np.std(gray)
    saturation = np.mean(hsv[:, :, 1])

    # Kantenintensität
    edges = cv2.Canny(gray, 80, 140)
    edge_strength = np.mean(edges)

    return {
        "brightness": float(brightness),
        "contrast": float(contrast),
        "saturation": float(saturation),
        "edge_strength": float(edge_strength),
    }


# -------------------------------------------------------------------
# Robustness Score
# Wie stark verändern sich die Merkmale?
# -------------------------------------------------------------------
def compute_robustness_score(original_features, altered_features):
    diffs = []

    for key in original_features.keys():
        orig = original_features[key]
        alt = altered_features[key]

        # relative Veränderung
        diff = abs(orig - alt) / (orig + 1e-6)
        diffs.append(diff)

    # Score 0 = robust, Score 1 = große Veränderung
    score = float(np.mean(diffs))
    return min(1.0, score)


# -------------------------------------------------------------------
# Bias Impact Score
# Prüft, ob Low-Light oder Noise besonders stark wirken.
# (Für MVP: Gleicher Score wie Robustness)
# -------------------------------------------------------------------
def compute_bias_impact(original_features, altered_features):
    """
    Bias-Risiko entsteht, wenn Edge-Bedingungen
    die Merkmale stärker verändern als normal.
    """
    robustness = compute_robustness_score(original_features, altered_features)

    # einfache Heuristik: hoher Robustness-Verlust = Bias-Risiko
    bias_impact = robustness

    return bias_impact


# -------------------------------------------------------------------
# Hauptfunktion: Analyse
# -------------------------------------------------------------------
def analyze_edge_bias(original, low_light, noise):
    f_orig = extract_features(original)
    f_low = extract_features(low_light)
    f_noise = extract_features(noise)

    results = {
        "original": f_orig,
        "low_light": {
            "features": f_low,
            "robustness": compute_robustness_score(f_orig, f_low),
            "bias_impact": compute_bias_impact(f_orig, f_low),
        },
        "noise": {
            "features": f_noise,
            "robustness": compute_robustness_score(f_orig, f_noise),
            "bias_impact": compute_bias_impact(f_orig, f_noise),
        },
    }

    return results
