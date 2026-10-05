import os

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from ml.data import apply_label, process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    save_model,
    train_model,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "census.csv")

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


@pytest.fixture(scope="module")
def data():
    """Load a small, reproducible sample of the census data."""
    df = pd.read_csv(DATA_PATH)
    return df.sample(n=2000, random_state=0)


@pytest.fixture(scope="module")
def split(data):
    """Train/test split of the sample data."""
    return train_test_split(
        data, test_size=0.2, random_state=0, stratify=data["salary"]
    )


@pytest.fixture(scope="module")
def processed(split):
    """Processed train and test arrays plus fitted encoder and binarizer."""
    train, test = split
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )
    return X_train, y_train, X_test, y_test, encoder, lb


def test_train_test_split_size_and_type(data, split):
    """
    The train and test datasets are DataFrames with the expected sizes
    (80/20 split) and together contain every row of the input data.
    """
    train, test = split
    assert isinstance(train, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)
    assert len(train) == 1600
    assert len(test) == 400
    assert len(train) + len(test) == len(data)


def test_process_data_output(processed):
    """
    process_data returns numpy arrays with matching row counts, the same
    number of feature columns for train and test, and binary labels.
    """
    X_train, y_train, X_test, y_test, _, _ = processed
    assert isinstance(X_train, np.ndarray)
    assert isinstance(y_train, np.ndarray)
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    assert set(np.unique(y_train)).issubset({0, 1})


def test_train_model_algorithm(processed):
    """
    train_model returns a fitted RandomForestClassifier, and inference
    returns one binary prediction per input row.
    """
    X_train, y_train, X_test, _, _, _ = processed
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
    preds = inference(model, X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_test.shape[0]
    assert set(np.unique(preds)).issubset({0, 1})


def test_compute_model_metrics_values():
    """
    compute_model_metrics returns the expected precision, recall and F1
    for a hand-computed example.
    """
    y = np.array([1, 1, 0, 0, 1])
    preds = np.array([1, 0, 0, 1, 1])
    # TP=2, FP=1, FN=1 -> precision=2/3, recall=2/3, F1=2/3
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(2 / 3)
    assert fbeta == pytest.approx(2 / 3)


def test_save_and_load_model(processed, tmp_path):
    """
    A model saved with save_model and reloaded with load_model produces
    identical predictions.
    """
    X_train, y_train, X_test, _, _, _ = processed
    model = train_model(X_train, y_train)
    path = tmp_path / "model.pkl"
    save_model(model, str(path))
    loaded = load_model(str(path))
    np.testing.assert_array_equal(inference(model, X_test), inference(loaded, X_test))


def test_apply_labels():
    """apply_label converts binary predictions to the salary strings."""
    assert apply_label([1]) == ">50K"
    assert apply_label([0]) == "<=50K"
