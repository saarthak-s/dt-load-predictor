# Import the pandas library so we can read, sort, and transform tabular time-series data cleanly.
import pandas as pd
# Import numpy so we can work with numeric arrays when needed for robust preprocessing and metrics handling.
import numpy as np
# Import joblib to save the trained model object to disk in a standard scikit-learn-compatible format.
import joblib
# Import mean_absolute_error to measure average absolute prediction error in the target unit.
from sklearn.metrics import mean_absolute_error
# Import mean_squared_error to compute RMSE and evaluate the model on the test set.
from sklearn.metrics import mean_squared_error
# Import XGBRegressor from xgboost because it is a strong gradient-boosted tree model suited to tabular forecasting.
from xgboost import XGBRegressor

from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import scipy.stats as stats

# Define the dataset path as a constant so the script is easy to run and maintain.
DATA_PATH = "ETTm1.csv"
# Define the output model path explicitly so the trained estimator is saved in the workspace root.
MODEL_PATH = "dt_model.pkl"
# Define the target variable name to keep the modeling pipeline explicit and readable.
TARGET_COLUMN = "OT"
# Define the raw feature columns that are directly available in the dataset before lag engineering.
RAW_FEATURE_COLUMNS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
# Define the final model input feature list after adding lagged target variables to prevent leakage from future information.
MODEL_FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT_lag_1", "OT_lag_4", "OT_lag_96"]

# Define a function that loads the dataset and prepares the time index for time-series modeling.
def load_and_prepare_data(file_path: str) -> pd.DataFrame:
    # Read the CSV file from disk using pandas so the raw sensor observations are available for analysis.
    df = pd.read_csv(file_path)
    # Convert the date string column to a pandas datetime type so time-based ordering and lag logic are valid.
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    # Sort the rows by timestamp to ensure the dataset is in chronological order before creating lagged features.
    df = df.sort_values("date").reset_index(drop=True)
    # Return the ordered and parsed dataset so downstream steps can safely engineer lags and split by time.
    return df

# Define a function that adds lagged target features representing 15-minute, 1-hour, and 24-hour temporal dependencies.
def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    # Create a 1-step lag so the model uses the previous 15-minute oil temperature as a strong autoregressive signal.
    df["OT_lag_1"] = df[TARGET_COLUMN].shift(1)
    # Create a 4-step lag so the model captures the previous 1-hour temperature pattern at 15-minute resolution.
    df["OT_lag_4"] = df[TARGET_COLUMN].shift(4)
    # Create a 96-step lag so the model captures the previous 24-hour trend, which is common in seasonal time series.
    df["OT_lag_96"] = df[TARGET_COLUMN].shift(96)
    # Remove rows where lagged values are missing because the system cannot create a valid supervised example before the first observations.
    df = df.dropna(subset=["OT_lag_1", "OT_lag_4", "OT_lag_96"]).reset_index(drop=True)
    # Return the dataset with lag-based autoregressive features ready for train/test splitting.
    return df

# Define a function that splits data using a strict chronological cutoff to avoid information leakage from future observations.
def chronological_split(df: pd.DataFrame, test_size: float = 0.2):
    # Validate the test size so the split ratio stays between zero and one, preventing invalid training or test sets.
    if not 0 < test_size < 1:
        # Raise a clear error when the caller provides an invalid temporal split ratio.
        raise ValueError("test_size must be between 0 and 1.")
    # Compute the index that separates the last fraction of observations for the test period using a time-ordered cutoff.
    split_index = int(len(df) * (1 - test_size))
    # Slice the earliest portion of the dataset for training so the model learns from older data only.
    train_df = df.iloc[:split_index].copy()
    # Slice the latest portion of the dataset for testing so evaluation reflects future performance.
    test_df = df.iloc[split_index:].copy()
    # Return both partitions so the model is trained and evaluated on a strict time ordering.
    return train_df, test_df


def optimize_xgboost(X_train, y_train):
    print("Starting Hyperparameter Optimization...")
    
    # 1. Define the baseline model
    base_model = XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1)
    
    # 2. Define a reasonable search space (Do not make this too massive)
    param_distributions = {
        'n_estimators': [200, 400, 600],
        'max_depth': [4, 6, 8],
        'learning_rate': stats.uniform(0.01, 0.1), # Random float between 0.01 and 0.11
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0]
    }
    
    # 3. Use TimeSeriesSplit! 
    # Standard K-Fold cross-validation randomly shuffles data, which causes data leakage.
    # TimeSeriesSplit ensures the model only ever validates on FUTURE data.
    tscv = TimeSeriesSplit(n_splits=3)
    
    # 4. Set up the Randomized Search
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=15, # It will test exactly 15 random combinations
        scoring='neg_mean_absolute_error', # Optimize specifically for MAE
        cv=tscv,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    # 5. Run the search
    random_search.fit(X_train, y_train)
    
    print("\n--- Optimization Complete ---")
    print(f"Best MAE from CV: {-random_search.best_score_:.4f}")
    print("Best Parameters:")
    for key, value in random_search.best_params_.items():
        print(f"  {key}: {value}")
        
    # Return the absolute best model found
    return random_search.best_estimator_

# Define the main orchestration function that builds the full forecasting pipeline.
def main() -> None:
    # Load the raw dataset from the CSV file so the model pipeline has access to all observations.
    df = load_and_prepare_data(DATA_PATH)
    # Add temporal lag features to the dataset so the model can learn short-term and daily dependencies in oil temperature.
    df = add_lag_features(df)
    # Split the data chronologically into train and test subsets without random shuffling to preserve time structure.
    train_df, test_df = chronological_split(df, test_size=0.2)
    # Select the model input feature columns from the training set for supervised learning.
    X_train = train_df[MODEL_FEATURES]
    # Select the target values from the training set to fit the regressor.
    y_train = train_df[TARGET_COLUMN]
    # Select the same feature columns from the test set so evaluation uses the exact same schema as training.
    X_test = test_df[MODEL_FEATURES]
    # Select the target values from the test set to measure out-of-sample accuracy.
    y_test = test_df[TARGET_COLUMN]
    # Initialize the XGBoost regressor with a squared-error objective suited for continuous target prediction.
    model = optimize_xgboost(X_train, y_train)
    # Predict oil temperature values for the unseen test period using the trained model.
    predictions = model.predict(X_test)
    # Compute mean absolute error to summarize the average absolute difference between predictions and actual values.
    mae = mean_absolute_error(y_test, predictions)
    # Compute RMSE by taking the square root of the mean squared error so large errors are penalized more strongly.
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    # Save the trained XGBoost model to disk so it can be reused in inference or deployment workflows later.
    joblib.dump(model, MODEL_PATH)
    # Print the MAE result in a human-readable way to confirm model quality on the test horizon.
    print(f"MAE: {mae:.4f}")
    # Print the RMSE result so stakeholders can see the magnitude of prediction error in the same units as OT.
    print(f"RMSE: {rmse:.4f}")
    # Print the output path to confirm where the saved model artifact was written on disk.
    print(f"Model saved to: {MODEL_PATH}")

# Guard the execution so the script only runs when invoked directly, not when imported elsewhere.
if __name__ == "__main__":
    # Execute the main training pipeline when this file is run as a script.
    main()
