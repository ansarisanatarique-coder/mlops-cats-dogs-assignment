import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import train_dummy_model


def test_model_training_creates_file():
    train_dummy_model()
    assert os.path.exists("model.pkl")
