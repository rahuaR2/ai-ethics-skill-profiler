import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

# 1) Modell laden
model = tf.keras.models.load_model("models/iris_model.keras")

# 2) Testdaten laden
X_test = np.load("models/X_test.npy")
y_test = np.load("models/y_test.npy")
groups = np.load("models/groups_test.npy")

# 3) Vorhersagen machen
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# 4) Gruppen-Accuracy
def group_accuracy(y_true, y_pred, groups):
    results = {}
    unique_groups = np.unique(groups)
    for g in unique_groups:
        mask = groups == g
        results[g] = accuracy_score(y_true[mask], y_pred[mask])
    return results

group_acc = group_accuracy(y_test, y_pred, groups)

# 5) Bias Score
bias = max(group_acc.values()) - min(group_acc.values())

# 6) Ausgabe
print("\n=== AI Ethics Skill-Profiler ===")
print("Gruppen-Genauigkeiten:")
for g, acc in group_acc.items():
    print(f"  {g}: {acc:.2f}")

print(f"\nBias Score: {bias:.2f}")

if bias > 0.10:
    print("⚠️  Das Modell ist nicht fair.")
else:
    print("✔️  Das Modell ist fair.")
