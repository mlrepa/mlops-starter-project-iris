import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.linear_model import LogisticRegression

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5001")
    mlflow.set_experiment("assignment-3-mlflow")

    with mlflow.start_run(run_name="parent_run"):
        mlflow.log_param("parent_info", "orchestrating training + logging")

        # Load train set
        train_dataset = pd.read_csv("data/train.csv")
        y = train_dataset.loc[:, "target"].values.astype("float32")
        X = train_dataset.drop("target", axis=1).values

        # Log training step in nested run
        with mlflow.start_run(run_name="training", nested=True):
            clf = LogisticRegression(C=0.01, solver="lbfgs", max_iter=100)
            clf.fit(X, y)

            # Log metrics or parameters (optional)
            mlflow.log_param("C", 0.01)
            mlflow.log_param("solver", "lbfgs")
            mlflow.log_metric("train_accuracy", clf.score(X, y))

        mlflow.log_metric("final_train_accuracy", clf.score(X, y))

        joblib.dump(clf, "models/model.joblib")

        signature = infer_signature(X, clf.predict(X))

        mlflow.sklearn.log_model(
            sk_model=clf, artifact_path="model", signature=signature
        )

        mlflow.register_model(
            f"runs:/{mlflow.active_run().info.run_id}/model", "assignment-3-model"
        )
