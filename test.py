import pytest
import joblib
import pandas as pd
import numpy as np
import json
import os

# Load once for all tests
@pytest.fixture(scope="module")
def model():
    return joblib.load("artifacts/california_housing_model.joblib")

@pytest.fixture(scope="module")
def sample_data():
    # Load or mock sample data for testing
    df = pd.read_csv("data/entity.csv", parse_dates=["event_timestamp"])
    return df.sample(5, random_state=42)  # Use a small sample

def test_model_prediction(model, sample_data):
    # Simulate Feast feature retrieval if needed
    from feast import FeatureStore
    store = FeatureStore(repo_path="Feast/feature_repo")
    features_df = store.get_historical_features(
        entity_df=sample_data,
        features=store.get_feature_service("feast_model_v1")
    ).to_df()

    X = features_df.drop("median_house_value", axis=1)
    preds = model.predict(X)

    assert isinstance(preds, np.ndarray)
    assert len(preds) == len(X)
    assert np.all(np.isfinite(preds)), "Predictions contain NaN or Inf"

def test_model_artifacts_exist():
    assert os.path.exists("artifacts/california_housing_model.joblib")
    assert os.path.exists("artifacts/model_coefficients.csv")
    assert os.path.exists("artifacts/model_intercept.txt")
    assert os.path.exists("artifacts/evaluation_metrics.json")

def test_metrics_sanity():
    with open("artifacts/evaluation_metrics.json", "r") as f:
        metrics = json.load(f)

    assert "RMSE" in metrics and metrics["RMSE"] > 0
    assert "MAE" in metrics and metrics["MAE"] > 0
    assert "R2" in metrics and -1 <= metrics["R2"] <= 1
