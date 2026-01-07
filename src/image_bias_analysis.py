import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

# --------------------------------------------------
# Helper: Convert uploaded image to OpenCV format
# --------------------------------------------------
def load_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = np.array(img)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# --------------------------------------------------
# Kategorie 1: Skin Tone Cluster (hell / mittel / dunkel)
# --------------------------------------------------
def analyze_skin_tone(image):
    img = cv2.resize(image, (200, 200))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Flatten
    pixels = lab.reshape((-1, 3))

    # K-Means Clustering
    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(pixels)
    centers = kmeans.cluster_centers_

    # We focus on L channel (lightness)
    lightness = centers[:, 0]

    avg_L = np.mean(lightness)

    if avg_L > 170:
        result = "hell"
    elif avg_L > 120:
        result = "mittel"
    else:
        result = "dunkel"

    return {
        "category": "Skin Tone Cluster",
        "result": result,
        "confidence": 0.8,
        "explanation": f"Durchschnittliche Luminanz (L={avg_L:.1f}) weist auf {result} hin.",
        "ethical_note": "Bewertet nur sichtbare Hautfarb-Helligkeit. Kein Rückschluss auf Ethnie."
    }


# --------------------------------------------------
# Kategorie 2: Age Appearance (Kind / Erwachsen / Alt)
# --------------------------------------------------
def analyze_age(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 150)

    edge_density = np.mean(edges)

    if edge_density < 4:
        result = "Kind / jugendlich erscheinend"
    elif edge_density < 8:
        result = "Erwachsene Person"
    else:
        result = "Ältere Person"

    return {
        "category": "Age Appearance",
        "result": result,
        "confidence": 0.7,
        "explanation": "Anzahl sichtbarer Falten/Kanten im Gesicht wurde analysiert.",
        "ethical_note": "Alter wird nur visuell geschätzt, keine Rückschlüsse auf tatsächliches Alter."
    }


# --------------------------------------------------
# Kategorie 3: Gender Expression (maskulin / feminin / uneindeutig)
# --------------------------------------------------
def analyze_gender_expression(image):
    h, w, _ = image.shape
    upper_part = image[: h//3, :]    # Haarbereich
    lower_part = image[h//3:, :]     # Kleidung

    hair_darkness = np.mean(cv2.cvtColor(upper_part, cv2.COLOR_BGR2GRAY))
    clothing_colorfulness = np.std(lower_part)

    if hair_darkness < 60:
        result = "feminin erscheinend"
    elif hair_darkness < 120:
        result = "uneindeutig"
    else:
        result = "maskulin erscheinend"

    return {
        "category": "Gender Expression",
        "result": result,
        "confidence": 0.6,
        "explanation": "Haarbereich + Kleidungskontraste wurden heuristisch analysiert.",
        "ethical_note": "Es wird keine Geschlechtsidentität festgestellt, nur visuelle Erscheinung."
    }


# --------------------------------------------------
# CENTRAL DISPATCH FUNCTION
# --------------------------------------------------
def analyze_image_bias(image_file, category):
    image = load_image(image_file)

    if category == "Skin Tone Cluster":
        return analyze_skin_tone(image)

    if category == "Age Appearance":
        return analyze_age(image)

    if category == "Gender Expression":
        return analyze_gender_expression(image)

    return {
        "category": category,
        "result": "Noch nicht implementiert",
        "confidence": 0.0,
        "explanation": "Diese Kategorie wird im nächsten Schritt ergänzt.",
        "ethical_note": "Keine Aussage möglich."
    }
# --------------------------------------------------
# Kategorie: Clothing Style / Role Indicators
# --------------------------------------------------
def analyze_clothing_style(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = np.mean(hsv[:, :, 1])
    colorfulness = np.std(image)

    # Einfache Regeln
    if saturation < 40 and colorfulness < 25:
        result = "Business Outfit"
    elif colorfulness > 60:
        result = "Casual / Freizeitkleidung"
    else:
        result = "Sportliche Kleidung"

    return {
        "category": "Clothing Style / Role Indicators",
        "result": result,
        "confidence": 0.6,
        "explanation": "Farbsättigung und Farbkontraste der Kleidung wurden analysiert.",
        "ethical_note": "Es werden keine Berufszuschreibungen gemacht, nur Kleidungsmuster analysiert."
    }
# --------------------------------------------------
# Kategorie: Visible Religious Symbols (Heuristik)
# --------------------------------------------------
def analyze_religious_symbols(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Kantenextraktion
    edges = cv2.Canny(blurred, 80, 130)

    # 1) Kreuz: rechtwinklige Linien erkennen
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60, minLineLength=30, maxLineGap=10)
    has_cross = False

    if lines is not None and len(lines) > 1:
        # vereinfachtes Kreuz-Matching: zwei Linien in unterschiedlicher Orientierung
        orientations = []
        for l in lines[:5]:
            x1, y1, x2, y2 = l[0]
            orientations.append(abs(x2 - x1) > abs(y2 - y1))  # True=horizontal, False=vertikal

        if True in orientations and False in orientations:
            has_cross = True

    # 2) Kopftuch: große einfarbige Fläche im oberen Teil
    upper = image[:image.shape[0]//3]
    color_std = np.std(upper)
    has_headscarf = color_std < 18  # einfarbig

    # 3) Kippa: kleiner dunkler Kreis am Kopf
    head = upper
    circles = cv2.HoughCircles(
        cv2.cvtColor(head, cv2.COLOR_BGR2GRAY),
        cv2.HOUGH_GRADIENT, 1, 20, param1=40, param2=20,
        minRadius=5, maxRadius=30
    )
    has_kippa = circles is not None

    if has_cross:
        result = "Kreuzsymbol sichtbar"
    elif has_headscarf:
        result = "Kopftuch erkennbar"
    elif has_kippa:
        result = "Kippa erkennbar"
    else:
        result = "Keine religiösen Symbole sichtbar"

    return {
        "category": "Visible Religious Symbols",
        "result": result,
        "confidence": 0.5,
        "explanation": "Form- und Farb-Hinweise wurden analysiert, keine Identitätszuordnung.",
        "ethical_note": "Es werden nur sichtbare Accessoires analysiert – keine Rückschlüsse auf Religion."
    }
# --------------------------------------------------
# Kategorie: Body Shape Appearance
# --------------------------------------------------
def analyze_body_shape(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return {
            "category": "Body Shape Appearance",
            "result": "unklar",
            "confidence": 0.0,
            "explanation": "Keine Kontur erkannt.",
            "ethical_note": "Nur visuelle Körperformen werden analysiert."
        }

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)

    ratio = w / h  # Breite zu Höhe

    if ratio < 0.35:
        result = "schlank erscheinend"
    elif ratio < 0.55:
        result = "durchschnittlich erscheinend"
    else:
        result = "kräftig / plus-size erscheinend"

    return {
        "category": "Body Shape Appearance",
        "result": result,
        "confidence": 0.7,
        "explanation": f"Seitenverhältnis des Körpers analysiert (Ratio: {ratio:.2f}).",
        "ethical_note": "Es handelt sich um eine neutrale Silhouette-Analyse, keine Bewertung."
    }
def analyze_image_bias(image_file, category):
    image = load_image(image_file)

    if category == "Skin Tone Cluster":
        return analyze_skin_tone(image)

    if category == "Age Appearance":
        return analyze_age(image)

    if category == "Gender Expression":
        return analyze_gender_expression(image)

    if category == "Clothing Style":
        return analyze_clothing_style(image)

    if category == "Visible Religious Symbols":
        return analyze_religious_symbols(image)

    if category == "Body Shape Appearance":
        return analyze_body_shape(image)

    return {...}
