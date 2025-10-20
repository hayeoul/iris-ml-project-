import pickle

from sklearn.ensemble import RandomForestClassifier


class IrisModel:
    def __init__(self, n_estimators=100, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state
        )
        self.is_trained = False

    def train(self, X, y):
        """Train the model"""
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.predict_proba(X)

    def save(self, filepath):
        """Save model to file"""
        with open(filepath, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, filepath):
        """Load model from file"""
        with open(filepath, "rb") as f:
            self.model = pickle.load(f)
        self.is_trained = True
