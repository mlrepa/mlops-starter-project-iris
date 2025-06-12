import joblib
import mlflow
import mlflow.models
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score

if __name__ == "__main__":
    # Load train set
    train_dataset = pd.read_csv("data/train.csv")

    # Get X and Y
    y: np.ndarray = train_dataset.loc[:, "target"].values.astype("float32")
    X: np.ndarray = train_dataset.drop("target", axis=1).values

    # Create an instance of Logistic Regression Classifier and fit the data.
    clf = LogisticRegression(C=0.01, solver="lbfgs", max_iter=100)
    clf.fit(X, y)

    joblib.dump(clf, "models/model.joblib")

    prediction: np.ndarray = clf.predict(X)

    # Calculate metrics
    classes = pd.read_csv("data/features_iris.csv")["target"].unique().tolist()
    cm: np.ndarray = confusion_matrix(y, prediction)
    f1: float = f1_score(y_true=y, y_pred=prediction, average="macro")

    # Log the model with MLflow
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment("assignment-3-mlflow")
    with mlflow.start_run() as run:
        # Save run ID in json file
        print(f"Logging run with ID: {run.info.run_id}")
        with open("run_id.json", "w") as f:
            f.write(f'{{"run_id": "{run.info.run_id}"}}')

        # Log parameters
        mlflow.log_param("model", "Logistic Regression")
        mlflow.log_param("parameters", {"C": 0.01, "solver": "lbfgs", "max_iter": 100})

        # Log metrics
        mlflow.log_metric("f1_score", f1)

        # Log artifacts
        mlflow.log_artifact("data/features_iris.csv")
        mlflow.log_artifact("data/train.csv")
        mlflow.log_artifact("models/model.joblib")

        # Set tags
        mlflow.set_tag("model_type", "Logistic Regression")
        mlflow.set_tag("dataset", "Iris")

        # Log the model
        signature = mlflow.models.infer_signature(X, clf.predict(X))
        model_info = mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            signature=signature,
            input_example=X[:5],
            registered_model_name="iris-classifier",
        )

    print("Model trained and logged successfully.")
