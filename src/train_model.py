import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

# 1) Daten laden
data = load_iris()
X = data.data
y = data.target

# Gruppenzugehörigkeit simulieren (z. B. "A" und "B")
groups = np.where(y == 0, "A", "B")  # Beispielgruppen

# 2) Split
X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    X, y, groups, test_size=0.3, random_state=42
)

# 3) Modell bauen
model = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),
    Dense(3, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 4) Training
model.fit(X_train, y_train, epochs=10, verbose=0)

# 5) Speichern
model.save("models/iris_model.keras")

# Extra speichern: X_test, y_test, groups_test
np.save("models/X_test.npy", X_test)
np.save("models/y_test.npy", y_test)
np.save("models/groups_test.npy", groups_test)

print("Modell erfolgreich trainiert und gespeichert!")
