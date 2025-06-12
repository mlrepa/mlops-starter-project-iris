import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

mlflow.set_experiment("assignment-3-mlflow")

if __name__ == "__main__":
    # Load train set
    train_dataset = pd.read_csv("data/train.csv")

    # Get X and y
    y = train_dataset["target"].values.astype("float32")
    X = train_dataset.drop("target", axis=1)
    
    with mlflow.start_run(run_name="training") as run:
        # Log hyperparameters
        mlflow.log_param("C", 0.01)
        mlflow.log_param("solver", "lbfgs")
        mlflow.log_param("max_iter", 100)

        # Train model
        clf = LogisticRegression(C=0.01, solver="lbfgs", max_iter=100)
        clf.fit(X, y)

        # Log metrics
        mlflow.log_metric("train_accuracy", clf.score(X, y))

        # Log model with signature
        signature = infer_signature(X, clf.predict(X))
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            signature=signature,
            registered_model_name="iris-logistic-regression"
        )

        # Save model to local file system (still needed for evaluate.py)
        joblib.dump(clf, "models/model.joblib")
