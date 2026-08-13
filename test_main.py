# Import the TestClient to simulate HTTP requests without starting a live Uvicorn server.
from fastapi.testclient import TestClient

# Import the FastAPI application instance from the main module so tests exercise the real API logic.
from main import app

# Initialize the test client that will send in-memory HTTP requests to the FastAPI app.
client = TestClient(app)

# Define the first test to confirm the root endpoint serves the expected health response.
def test_health_check():
    # Send a GET request to the root endpoint to verify the service is live.
    response = client.get("/")

    # Assert that the HTTP status code is 200, which indicates success.
    assert response.status_code == 200
    # Assert that the JSON payload exactly matches the expected health-check contract.
    assert response.json() == {"status": "ok", "service": "Distribution Transformer Forecast API"}

# Define the second test to confirm a valid request produces a model prediction.
def test_valid_prediction():
    # Create a payload that matches the full feature contract required by the trained XGBoost model.
    valid_payload = {
        "HUFL": 12.4,
        "HULL": 2.1,
        "MUFL": 8.3,
        "MULL": 1.2,
        "LUFL": 4.5,
        "LULL": 0.8,
        "OT_lag_1": 34.2,
        "OT_lag_4": 33.8,
        "OT_lag_96": 32.1,
    }

    # Send a POST request to the prediction endpoint using the valid payload.
    response = client.post("/predict", json=valid_payload)

    # Assert that the API accepted the request and returned a successful HTTP response.
    assert response.status_code == 200

    # Parse the JSON payload from the response to validate the output contract.
    data = response.json()

    # Assert that the result includes the predicted_temperature field.
    assert "predicted_temperature" in data
    # Assert that the prediction value is of numeric float type, not a string or integer.
    assert isinstance(data["predicted_temperature"], float)

# Define the third test to validate that missing required input is rejected at the API layer.
def test_invalid_prediction_missing_data():
    # Create an incomplete payload that omits the OT_lag_96 feature required by the model contract.
    broken_payload = {
        "HUFL": 12.4,
        "HULL": 2.1,
        "MUFL": 8.3,
        "MULL": 1.2,
        "LUFL": 4.5,
        "LULL": 0.8,
        "OT_lag_1": 34.2,
        "OT_lag_4": 33.8,
    }

    # Send the invalid request to the prediction endpoint to verify validation triggers.
    response = client.post("/predict", json=broken_payload)

    # Assert that FastAPI returns a 422 validation error when required fields are missing.
    assert response.status_code == 422

    # Extract the validation detail payload to inspect the exact missing field information.
    error_detail = response.json()["detail"]
    # Assert that the location points to the missing field in the request body.
    assert error_detail[0]["loc"] == ["body", "OT_lag_96"]
    # Assert that FastAPI identifies the problem as a missing required field.
    assert error_detail[0]["type"] == "missing"
