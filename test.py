# test.py

from src.data.loader import load_data
from src.features.preprocessing import preprocess
from src.models.model import IrisModel

df = load_data()
X, y = preprocess(df)

model = IrisModel()
model.train(X, y)

predictions = model.predict(X[:5])
print(predictions)
