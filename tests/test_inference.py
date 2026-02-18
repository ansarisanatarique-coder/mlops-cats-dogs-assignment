
from src.model import SimpleCNN
import torch

def test_model_output():
    model = SimpleCNN()
    x = torch.randn(1,3,224,224)
    output = model(x)
    assert output.shape == (1,2)
