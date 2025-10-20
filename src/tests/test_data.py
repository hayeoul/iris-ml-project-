import pandas as pd
import pytest

from src.data.loader import load_data
from src.data.validator import validate_dataset


def test_load_data():
    """Test data loading"""
    df = load_data()
    assert df is not None
    assert len(df) == 150
    assert "target" in df.columns


def test_validate_dataset_success():
    """Test successful validation"""
    df = load_data()
    assert validate_dataset(df) is True


def test_validate_dataset_missing_columns():
    """Test validation with missing columns"""
    df = pd.DataFrame({"sepal length (cm)": [1, 2, 3]})

    with pytest.raises(ValueError, match="Missing columns"):
        validate_dataset(df)


def test_validate_dataset_null_values():
    """Test validation with null values"""
    df = load_data()
    df.loc[0, "sepal length (cm)"] = None

    assert validate_dataset(df) is False


def test_validate_dataset_invalid_target():
    """Test validation with invalid target"""
    df = load_data()
    df.loc[0, "target"] = 5

    assert validate_dataset(df) is False
