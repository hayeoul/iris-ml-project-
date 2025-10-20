import pandas as pd
from sklearn.datasets import load_iris


def load_data():
    """Load iris dataset"""
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df["target"] = iris.target
    return df


def load_train_data():
    """Load training data"""
    df = load_data()
    return df[:120]


def load_test_data():
    """Load test data"""
    df = load_data()
    X = df[120:][
        [
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)",
        ]
    ].values
    y = df[120:]["target"].values
    return X, y
