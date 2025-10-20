def validate_dataset(df):
    """Validate dataset schema and quality"""
    required_columns = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
        "target",
    ]

    # Check required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    # Check for null values
    if df.isnull().any().any():
        return False

    # Check target range
    if not df["target"].isin([0, 1, 2]).all():
        return False

    return True
