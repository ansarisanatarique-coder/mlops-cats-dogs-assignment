
from src.data_preprocessing import preprocess_image
from io import BytesIO
from PIL import Image

def test_preprocess_shape():
    img = Image.new("RGB", (300,300))
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    tensor = preprocess_image(buffer)
    assert tensor.shape[1:] == (224,224)
