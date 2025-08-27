from pathlib import Path

root = Path("ml-pipeline/datasets/home_sample_v1")
missing = []

for split in ["train", "val"]:
    imgs = (root / "images" / split).glob("*.jpeg")
    for img in imgs:
        lbl = root / "labels" / split / (img.stem + ".txt")
        if not lbl.exists():
            missing.append((img.name, lbl.name))

if missing:
    print("⚠️ Missing labels:")
    for img, lbl in missing:
        print(f"  {img} → expected {lbl}")
else:
    print("✅ All images have matching labels")