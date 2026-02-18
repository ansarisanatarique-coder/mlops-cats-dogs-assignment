import os
from PIL import Image

DATA_DIR = "data/processed"

def clean_folder(folder):
    removed = 0
    for root, dirs, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            try:
                img = Image.open(path)
                img.verify()  # verifies image
            except:
                print("Removing corrupted:", path)
                os.remove(path)
                removed += 1
    return removed

if __name__ == "__main__":
    total = clean_folder(DATA_DIR)
    print("Total corrupted images removed:", total)
