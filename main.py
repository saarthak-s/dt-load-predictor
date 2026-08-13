# Import FastAPI so we can expose a production-ready HTTP API for model inference.
from fastapi import FastAPI
# Import BaseModel from pydantic so request payloads are validated before the model sees them.
from pydantic import BaseModel, Field
# Import pandas so we can reshape a JSON payload into a DataFrame with the exact model feature layout.
import pandas as pd
# Import joblib so we can load the pre-trained scikit-learn/XGBoost model object efficiently from disk.
import joblib

# Create a FastAPI application instance that will serve the prediction endpoint.
app = FastAPI(title="Distribution Transformer Forecast API", version="1.0.0")
# Define the exact feature schema required by the trained model to avoid silent ordering mistakes.
MODEL_FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT_lag_1", "OT_lag_4", "OT_lag_96"]
# Define the model artifact path so the service loads the trained pickle from the same workspace directory.
MODEL_PATH = "dt_model.pkl"

# Define a Pydantic request model to validate and document the nine required numeric inputs.
class PredictionRequest(BaseModel):
    # Validate the first feature as a float so the API rejects strings and invalid values early.
    HUFL: float = Field(..., description="HUFL sensor reading for the transformer asset.")
    # Validate the second feature as a float to ensure consistent numeric input for the model.
    HULL: float = Field(..., description="HULL sensor reading for the transformer asset.")
    # Validate the third feature as a float to preserve the exact ordering expected by the trained estimator.
    MUFL: float = Field(..., description="MUFL sensor reading for the transformer asset.")
    # Validate the fourth feature as a float so missing or malformed values are caught before inference.
    MULL: float = Field(..., description="MULL sensor reading for the transformer asset.")
    # Validate the fifth feature as a float because it is part of the required model input vector.
    LUFL: float = Field(..., description="LUFL sensor reading for the transformer asset.")
    # Validate the sixth feature as a float so the model always receives the exact exogenous feature set.
    LULL: float = Field(..., description="LULL sensor reading for the transformer asset.")
    # Validate the seventh feature as a float because the OT lag at one step is required for temporal autoregression.
    OT_lag_1: float = Field(..., description="Previous 15-minute oil temperature value used as a lag feature.")
    # Validate the eighth feature as a float because the one-hour lag is part of the model contract.
    OT_lag_4: float = Field(..., description="Previous 1-hour oil temperature lag used as a temporal feature.")
    # Validate the ninth feature as a float because the 24-hour lag captures daily seasonality.
    OT_lag_96: float = Field(..., description="Previous 24-hour oil temperature lag used as a daily seasonal feature.")

# Load the trained model once when the application starts so prediction requests do not incur repeated disk I/O.
model = joblib.load(MODEL_PATH)

# Define a response model for the API output to make the contract explicit and self-documenting.
class PredictionResponse(BaseModel):
    # Define a predicted temperature field so the API returns a numeric value in the same scale as the target.
    predicted_temperature: float

# Define the root endpoint to provide health-check information for operational monitoring.
@app.get("/")
# Create a lightweight health endpoint so deployment checks know the service is live and ready for requests.
def health_check():
    # Return a small JSON payload confirming the API is operational.
    return {"status": "ok", "service": "Distribution Transformer Forecast API"}

# Define the prediction endpoint that accepts a structured payload and emits a forecast.
@app.post("/predict", response_model=PredictionResponse)
# Create the endpoint function that receives validated request data and returns a prediction.
def predict_temperature(request: PredictionRequest):
    # Construct a DataFrame from the validated request object while preserving the exact feature order expected by the model.
    features_df = pd.DataFrame([request.model_dump()])[MODEL_FEATURES]
    # Run the model on the single-row DataFrame to produce a forecast for the incoming transformer state.
    prediction = model.predict(features_df)[0]
    # Return the predicted temperature in a JSON-safe object that matches the response schema.
    return {"predicted_temperature": float(prediction)}
