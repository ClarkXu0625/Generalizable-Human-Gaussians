import os
from PIL import Image
from collections import Counter

INPUT_DIR = "/home/clark/Documents/GitHub/Generalizable-Human-Gaussians/datasets/THuman/train/img/0004_000"

sizes = Counter()

for root, _, files in os.walk(INPUT_DIR):
    for f in files:
        if f.lower().endswith((".jpg", ".png")):
            fp = os.path.join(root, f)
            try:
                with Image.open(fp) as im:
                    sizes[(im.width, im.height)] += 1
            except Exception as e:
                print(f"Could not open {fp}: {e}")

print("Unique image sizes found:")
for (w,h), count in sizes.items():
    print(f"  {w}x{h}: {count} images")
