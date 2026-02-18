import os
import shutil
import random
from pathlib import Path

# Paths
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

SPLITS = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1
}

IMG_SIZE = (224, 224)

def create_folders():
    for split in SPLITS:
        for category in ["cats", "dogs"]:
            os.makedirs(PROCESSED_DIR / split / category, exist_ok=True)

def split_images():
    for category in ["cats", "dogs"]:
        images = list((RAW_DIR / category).glob("*"))
        random.shuffle(images)

        total = len(images)
        train_end = int(total * 0.8)
        val_end = int(total * 0.9)

        splits = {
            "train": images[:train_end],
            "val": images[train_end:val_end],
            "test": images[val_end:]
        }

        for split_name, split_images in splits.items():
            for img_path in split_images:
                dest = PROCESSED_DIR / split_name / category / img_path.name
                shutil.copy(img_path, dest)

if __name__ == "__main__":
    print("Creating folders...")
    create_folders()

    print("Splitting images...")
    split_images()

    print("Done.")
