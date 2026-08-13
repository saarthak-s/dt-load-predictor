# DT Forecast API

This project provides a FastAPI service for predicting distribution transformer oil temperature using an XGBoost regressor trained on the ETTm1 time-series dataset.

## Project Overview

- Time-series forecasting for transformer oil temperature (`OT`)
- Input features:
  - `HUFL`
  - `HULL`
  - `MUFL`
  - `MULL`
  - `LUFL`
  - `LULL`
  - `OT_lag_1`
  - `OT_lag_4`
  - `OT_lag_96`
- Model artifact: `dt_model.pkl`
- API framework: FastAPI
- Environment manager: uv

## Local Setup

1. Install uv if needed:

   ```bash
   pip install uv
   ```

2. Create and sync the environment:

   ```bash
   uv sync --extra dev
   ```

3. Run the API locally:

   ```bash
   uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

4. Open the docs:

   ```text
   http://localhost:8000/docs
   ```

## Example Prediction Request

```json
{
  "HUFL": 12.4,
  "HULL": 2.1,
  "MUFL": 8.3,
  "MULL": 1.2,
  "LUFL": 4.5,
  "LULL": 0.8,
  "OT_lag_1": 34.2,
  "OT_lag_4": 33.8,
  "OT_lag_96": 32.1
}
```

Example response:

```json
{
  "predicted_temperature": 33.5421
}
```

## Run Tests

```bash
uv run pytest test_main.py -v
```

## Docker

Build the image:

```bash
docker build -t dt-forecast-api .
```

Run the container:

```bash
docker run -p 8000:8000 dt-forecast-api
```

## Notes

- This API expects the exact feature order required by the trained XGBoost model.
- The model file is a deployment artifact and should be managed carefully for production use.
- The project uses a strict chronological split for training and evaluation to avoid time-series leakage.
