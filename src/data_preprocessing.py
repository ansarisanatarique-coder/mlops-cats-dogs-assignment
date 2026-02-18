
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

def preprocess_image(file):
    image = Image.open(file).convert("RGB")
    return transform(image)
