from sklearn.linear_model import LogisticRegression
import numpy as np
import joblib

def train_dummy_model():
    X = np.random.rand(10, 224*224*3)
    y = np.random.randint(0, 2, 10)

    model = LogisticRegression(max_iter=100)
    model.fit(X, y)

    joblib.dump(model, "model.pkl")
