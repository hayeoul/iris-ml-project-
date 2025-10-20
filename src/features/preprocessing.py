from sklearn.preprocessing import StandardScaler


def preprocess(df):
    """Preprocess data for training"""
    feature_cols = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]

    X = df[feature_cols].values
    y = df["target"].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
