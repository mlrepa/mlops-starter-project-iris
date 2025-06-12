import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
import mlflow

def evaluate_model():
    # Load test set
    test_dataset = pd.read_csv("data/test.csv")
    y = test_dataset["target"].values.astype("float32")
    X = test_dataset.drop("target", axis=1)

    # Load model
    clf = joblib.load("models/model.joblib")

    # Predict
    y_pred = clf.predict(X)

    # Evaluate
    f1 = f1_score(y, y_pred, average="macro")
    cm = confusion_matrix(y, y_pred)

    return f1, cm.tolist()

if __name__ == "__main__":
    with mlflow.start_run(run_name="evaluation", nested=True):
        f1, cm = evaluate_model()

        # Log evaluation metrics
        mlflow.log_metric("f1_score", f1)

        # Optionally log confusion matrix as artifact
        with open("data/eval.json", "w") as f:
            json.dump({"f1_score": f1, "confusion_matrix": cm}, f, indent=2)
        mlflow.log_artifact("data/eval.json")
