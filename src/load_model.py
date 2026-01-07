import os
import numpy as np
import tensorflow as tf

def load_model_and_data():

    project_root = os.path.dirname(os.path.dirname(__file__))

    model_path = os.path.join(project_root, "models", "iris_model.keras")
    data_path  = os.path.join(project_root, "data")

    # WICHTIG: altes Modell richtig laden
    model = tf.keras.models.load_model(model_path, compile=False)

    X_test = np.load(os.path.join(data_path, "X_test.npy"))
    y_test = np.load(os.path.join(data_path, "y_test.npy"))
    groups = np.load(os.path.join(data_path, "groups.npy"))

    return model, X_test, y_test, groups
