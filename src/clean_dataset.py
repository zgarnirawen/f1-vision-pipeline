# src/clean_dataset.py
from pathlib import Path
from PIL import Image

root = Path("data/processed/classification")
bad_files = []

for path in root.rglob("*.*"):
    if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        continue
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception:
        bad_files.append(path)

print(f"Found {len(bad_files)} unreadable files")
for f in bad_files:
    print(f)
    f.unlink()

print("Done. Removed all unreadable files.")
