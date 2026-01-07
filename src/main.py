from load_model import load_model_and_data
from fairness_metrics import group_accuracy
from bias_analysis import compute_bias
from report_generator import generate_report
import numpy as np
import os

def main():
    # 1) Modell + Daten laden
    model, X_test, y_test, groups = load_model_and_data()

    # 2) Vorhersagen berechnen
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # 3) Fairness-Metriken
    group_acc = group_accuracy(y_test, y_pred, groups)

    # 4) Bias berechnen
    bias = compute_bias(group_acc)

    # 5) Report erzeugen
    report = generate_report(group_acc, bias)

    # 6) Report im Terminal ausgeben
    print(report)

    # 7) Report speichern
    os.makedirs("reports", exist_ok=True)
    with open("reports/fairness_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("\n📄 Report gespeichert unter: reports/fairness_report.txt")

if __name__ == "__main__":
    main()
