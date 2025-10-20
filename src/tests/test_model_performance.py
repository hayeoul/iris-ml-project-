import pytest
from sklearn.metrics import accuracy_score, f1_score

from src.data.loader import load_data, load_test_data
from src.features.preprocessing import preprocess
from src.models.model import IrisModel


@pytest.fixture
def trained_model():
    """Fixture for trained model"""
    df = load_data()
    X, y = preprocess(df[:120])

    model = IrisModel()
    model.train(X, y)
    return model


def test_model_accuracy_threshold(trained_model):
    """Test minimum accuracy threshold"""
    X_test, y_test = load_test_data()
    predictions = trained_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    MIN_ACCURACY = 0.85
    assert (
        accuracy >= MIN_ACCURACY
    ), f"Accuracy {accuracy:.3f} below threshold {MIN_ACCURACY}"


def test_model_f1_score(trained_model):
    """Test F1 score"""
    X_test, y_test = load_test_data()
    predictions = trained_model.predict(X_test)
    f1 = f1_score(y_test, predictions, average="weighted")

    MIN_F1 = 0.80
    assert f1 >= MIN_F1, f"F1-score {f1:.3f} below threshold {MIN_F1}"


def test_model_inference_time(trained_model):
    """Test inference time performance"""
    import time

    X_test, _ = load_test_data()

    start = time.time()
    _ = trained_model.predict(X_test)
    elapsed = time.time() - start

    MAX_INFERENCE_TIME = 0.1
    assert elapsed < MAX_INFERENCE_TIME, f"Inference too slow: {elapsed:.3f}s"
