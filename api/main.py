import os
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="Iris Model API",
    description="API for Iris species prediction using trained LogisticRegression model",  # noqa: E501
    version="1.0.0",
)

# Load the trained model
MODEL_PATH = "models/model.joblib"
model = None


def load_model() -> None:
    """Load the trained model from the models directory."""
    global model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
    else:
        print(f"Warning: Model file not found at {MODEL_PATH}")
        model = None


# Load model on startup
load_model()


# Define the input data model using Pydantic
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2,
            }
        }


class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_species: str
    confidence: float
    input_features: dict


# Define the root endpoint
@app.get("/")
def read_root() -> dict[str, Any]:
    return {
        "message": "Welcome to the Iris Prediction API",
        "endpoints": {
            "/predict": "POST - Make a prediction",
            "/health": "GET - Health check",
        },
        "model_loaded": model is not None,
    }


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
    }


# Define the prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict_species(iris_input: IrisInput) -> PredictionResponse:
    """
    Predicts the Iris species based on input features.
    Returns the predicted class (0, 1, 2) and species name.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Create input array in the correct order
        input_features = np.array(
            [
                [
                    iris_input.sepal_length,
                    iris_input.sepal_width,
                    iris_input.petal_length,
                    iris_input.petal_width,
                ]
            ]
        )

        # Make prediction
        prediction = model.predict(input_features)[0]

        # Get prediction probabilities for confidence
        probabilities = model.predict_proba(input_features)[0]
        confidence = float(np.max(probabilities))

        # Map prediction to species name
        species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}
        predicted_species = species_map.get(int(prediction), "unknown")

        return PredictionResponse(
            predicted_class=int(prediction),
            predicted_species=predicted_species,
            confidence=confidence,
            input_features=iris_input.dict(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction error: {str(e)}"
        ) from e


@app.get("/model-info")
def get_model_info() -> dict[str, Any]:
    """Get information about the loaded model."""
    if model is None:
        return {"model_loaded": False}

    return {
        "model_loaded": True,
        "model_type": str(type(model).__name__),
        "feature_names": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "target_names": ["setosa", "versicolor", "virginica"],
    }
