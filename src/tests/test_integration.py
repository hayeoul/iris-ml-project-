from src.data.loader import load_data
from src.data.validator import validate_dataset
from src.features.preprocessing import preprocess
from src.models.model import IrisModel


def test_full_pipeline():
    """Test complete ML pipeline"""
    # Load data
    df = load_data()
    assert df is not None

    # Validate data
    assert validate_dataset(df) is True

    # Preprocess
    X, y = preprocess(df[:120])
    assert X.shape[0] == y.shape[0]
    assert X.shape[0] == 120

    # Train model
    model = IrisModel()
    model.train(X, y)
    assert model.is_trained is True

    # Make predictions
    predictions = model.predict(X[:10])
    assert len(predictions) == 10
