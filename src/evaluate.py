import json
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

mlflow.set_tracking_uri("http://127.0.0.1:5001")
mlflow.set_experiment("assignment-3-mlflow")


def evaluate_model() -> dict[str, Any]:
    """Evaluate the trained model and return metrics."""

    # Load unique classes from the original features file
    classes = pd.read_csv("data/features_iris.csv")["target"].unique().tolist()

    # Load test dataset
    test_dataset = pd.read_csv("data/test.csv")
    y: np.ndarray = test_dataset.loc[:, "target"].values.astype("float32")
    X: np.ndarray = test_dataset.drop("target", axis=1).values

    # Load trained model
    clf = joblib.load("models/model.joblib")

    # Make predictions
    prediction: np.ndarray = clf.predict(X)

    # Calculate metrics
    cm: np.ndarray = confusion_matrix(y, prediction)
    f1: float = f1_score(y_true=y, y_pred=prediction, average="macro")

    return {
        "f1_score": f1,
        "confusion_matrix": {"classes": classes, "matrix": cm.tolist()},
    }


if __name__ == "__main__":
    with mlflow.start_run(run_name="evaluation", nested=True):
        metrics = evaluate_model()

        mlflow.log_metric("eval_f1_score", metrics["f1_score"])

        with open("data/eval.json", "w") as f:
            json.dump(metrics, f, indent=2)

        mlflow.log_artifact("data/eval.json")
