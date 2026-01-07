import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_and_save_model():

    # 1. Daten laden
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Skalieren
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Speichern für Dashboard
    np.save("data/X_test.npy", X_test)
    np.save("data/y_test.npy", y_test)

    # Dummy-Gruppen: Blumenart 0,1,2
    np.save("data/groups.npy", y_test)

    # 2. Modell bauen
    model = keras.Sequential([
        keras.layers.Input(shape=(4,)),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(3, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 3. Trainieren
    model.fit(X_train, y_train, epochs=20, verbose=0)

    # 4. Speichern im neuen Keras-Format
    model.save("models/iris_model.keras")

    print("Neues Iris-Modell gespeichert!")

if __name__ == "__main__":
    train_and_save_model()
