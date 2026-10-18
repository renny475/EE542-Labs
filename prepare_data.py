from pathlib import Path
import io

import pandas as pd
from PIL import Image

PARQUET_FILES = [
    Path("data/parquet/data/train-00000-of-00002.parquet"),
    Path("data/parquet/data/train-00001-of-00002.parquet"),
]

# The dataset labels are 0 = cat and 1 = dog.
LABEL_NAMES = {0: "cat", 1: "dog"}
TEST_FRACTION = 0.2
SPLIT_SEED = 42

# Collect image bytes by class.
images_by_label = {"cat": [], "dog": []}

for parquet_file in PARQUET_FILES:
    print(f"Reading {parquet_file}...")
    df = pd.read_parquet(parquet_file)

    for _, row in df.iterrows():
        label = int(row["labels"])
        image_value = row["image"]

        # Hugging Face image columns are commonly stored as a dict with
        # the encoded bytes under "bytes".
        image_bytes = image_value["bytes"]
        if image_bytes is None:
            image_path = image_value["path"]
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()

        images_by_label[LABEL_NAMES[label]].append(image_bytes)

# Make a deterministic train/test split within each class.
import random

rng = random.Random(SPLIT_SEED)

for class_name, image_bytes_list in images_by_label.items():
    indices = list(range(len(image_bytes_list)))
    rng.shuffle(indices)
    test_count = int(len(indices) * TEST_FRACTION)
    test_indices = set(indices[:test_count])

    for index, encoded_image in enumerate(image_bytes_list):
        split = "test" if index in test_indices else "train"
        output_dir = Path("data") / split / class_name
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(io.BytesIO(encoded_image)) as image:
                image = image.convert("RGB")
                image.save(output_dir / f"{class_name}_{index:06d}.jpg", format="JPEG")
        except Exception as error:
            print(f"Skipping invalid {class_name} image {index}: {error}")

print("Image extraction and splitting complete.")
