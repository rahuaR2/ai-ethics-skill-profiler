import numpy as np
from sklearn.metrics import accuracy_score

def group_accuracy(y_true, y_pred, groups):
    results = {}
    unique_groups = np.unique(groups)

    for g in unique_groups:
        mask = (groups == g)

        # JSON-kompatibel machen
        group_key = int(g)  # numpy → Python int
        acc_value = float(accuracy_score(y_true[mask], y_pred[mask]))  # numpy.float → float

        results[group_key] = acc_value

    return results
