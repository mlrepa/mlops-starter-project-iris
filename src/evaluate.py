import json
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score


def evaluate_model() -> dict[str, Any]:
    """Evaluate the trained model and return metrics.

    Returns:
        Dictionary containing evaluation metrics
    """
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
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("Iris_Classification")

    # Load the parent run ID from the JSON file
    try:
        with open("run_id.json") as f:
            run_data = json.load(f)
            parent_run_id = run_data["run_id"]
    except FileNotFoundError:
        print("Warning: run_id.json not found. Creating a new independent run.")
        parent_run_id = None

    metrics = evaluate_model()

    # Save metrics as JSON
    with open("data/eval.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with mlflow.start_run(parent_run_id=parent_run_id, nested=True) as run:
        print(
            f"Started evaluation run: {run.info.run_id} (nested under {parent_run_id})"
        )

        # Log metrics
        mlflow.log_metric("f1_score", metrics["f1_score"])
        # mlflow.log_metric(
        #   "confusion_matrix.classes", metrics['confusion_matrix']['classes']
        # )
        # mlflow.log_metric(
        #   "confusion_matrix.matrix", metrics['confusion_matrix']['matrix']
        # )

        # Log artifacts
        mlflow.log_artifact("data/eval.json")
        mlflow.log_artifact("data/test.csv")

        # Set tags
        mlflow.set_tag("model_type", "Model Evaluation")
        mlflow.set_tag("dataset", "Iris")

    print("Evaluation completed and metrics logged to MLflow!")
