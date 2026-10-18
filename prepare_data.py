"""
把 Hugging Face microsoft/cats_vs_dogs 的 parquet 解成作業要求的資料夾結構:

data/
├── train/
│   ├── cat/
│   └── dog/
└── test/

parquet 裡存的就是原始 JPEG bytes，直接寫檔，不重新編碼 (無損、也快)。
用法: python prepare_data.py [parquet 所在目錄] [輸出目錄]
"""
import io
import json
import os
import sys
from glob import glob

import pyarrow.parquet as pq
from PIL import Image

SRC_DIR = sys.argv[1] if len(sys.argv) > 1 else "data/data"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "data"


def main():
    files = sorted(glob(os.path.join(SRC_DIR, "*.parquet")))
    if not files:
        sys.exit(f"No parquet files found in {SRC_DIR}")

    # 類別名稱從 parquet metadata 讀 (labels: 0=cat, 1=dog)，不寫死
    meta = json.loads(pq.read_schema(files[0]).metadata[b"huggingface"])
    names = meta["info"]["features"]["labels"]["names"]
    for name in names:
        os.makedirs(os.path.join(OUT_DIR, "train", name), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "test"), exist_ok=True)

    # 各 shard 裡的 image.path 都從 0.jpg 開始會撞名，所以用全域流水號命名
    idx, counts, skipped = 0, {n: 0 for n in names}, 0
    for f in files:
        table = pq.read_table(f, columns=["image", "labels"])
        for image, label in zip(table.column("image").to_pylist(),
                                table.column("labels").to_pylist()):
            data = image["bytes"]
            try:
                Image.open(io.BytesIO(data)).verify()   # 跳過壞檔，避免訓練時才爆
            except Exception:
                skipped += 1
                continue
            name = names[label]
            with open(os.path.join(OUT_DIR, "train", name, f"{name}.{idx}.jpg"), "wb") as out:
                out.write(data)
            counts[name] += 1
            idx += 1
        print(f"{os.path.basename(f)} done")

    print(f"Wrote {counts} to {os.path.join(OUT_DIR, 'train')}, skipped {skipped} corrupt images")


if __name__ == "__main__":
    main()
