import os
import shutil
import random

SOURCE = "data/processed/train"
DEST = "data/small_train"

CLASSES = ["cats", "dogs"]
LIMIT = 1000   # per class

for cls in CLASSES:
    src_folder = os.path.join(SOURCE, cls)
    dest_folder = os.path.join(DEST, cls)
    os.makedirs(dest_folder, exist_ok=True)

    images = os.listdir(src_folder)
    random.shuffle(images)

    for img in images[:LIMIT]:
        shutil.copy(
            os.path.join(src_folder, img),
            os.path.join(dest_folder, img)
        )

print("Small dataset created!")
