# src/edge_simulator.py

import cv2
import numpy as np
from PIL import Image


# -----------------------------------------------------------
# Hilfsfunktion: Bild laden und in OpenCV Format bringen
# -----------------------------------------------------------
def load_image(image_file):
    img = Image.open(image_file).convert("RGB")
    img = np.array(img)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# -----------------------------------------------------------
# TEST 1: Low-Light Simulation
# Abdunkelung von Bildern (realistische Edge-Bedingung)
# -----------------------------------------------------------
def simulate_low_light(image, factor=0.4):
    """
    Verdunkelt das Bild, um schlechte Beleuchtung zu simulieren.
    
    factor < 1 -> dunkler
    """
    low_light = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    return low_light


# -----------------------------------------------------------
# TEST 2: Sensor Noise Simulation (Rauschen)
# Simuliert Rauschen auf Low-End Edge-Kameras
# -----------------------------------------------------------
def simulate_noise(image, intensity=25):
    """
    Fügt dem Bild normales Gauß-Rauschen hinzu.
    intensity: Rauschgrad
    """
    noise = np.random.normal(0, intensity, image.shape).astype(np.int16)
    noisy_image = image.astype(np.int16) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image


# -----------------------------------------------------------
# Verpackung: Edge-Simulationen ausführen
# -----------------------------------------------------------
def run_edge_simulations(image_file):
    """
    Führt beide MVP-Simulationen durch und gibt alle Versionen zurück.
    """
    img = load_image(image_file)

    # Original
    original = img

    # Edge-Simulationen
    low_light = simulate_low_light(img)
    noise = simulate_noise(img)

    return {
        "original": original,
        "low_light": low_light,
        "noise": noise,
    }
