"""
split_dataset.py

Splits the raw classification dataset (one folder per team, e.g.
data/raw/classification/Ferrari/, data/raw/classification/Mercedes/, ...)
into train/val/test folders, keeping the class balance the same in each split
(stratified split) so no team ends up over/under-represented in any set.

Usage:
    python src/split_dataset.py \
        --input data/raw/classification \
        --output data/processed/classification \
        --train 0.70 --val 0.15 --test 0.15
"""

import argparse
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split


def collect_images(input_dir: Path):
    """Return {class_name: [file_paths]} for every subfolder in input_dir."""
    valid_ext = {".jpg", ".jpeg", ".png", ".webp"}
    classes = {}
    for class_dir in sorted(input_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        files = [f for f in class_dir.iterdir() if f.suffix.lower() in valid_ext]
        if not files:
            print(f"  [warn] no images found in {class_dir}, skipping")
            continue
        classes[class_dir.name] = files
    return classes


def split_class_files(files, train_ratio, val_ratio, test_ratio, seed=42):
    """Split one class's file list into train/val/test, stratified isn't needed
    per-class since we split each class independently (that IS the stratification)."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train + val + test ratios must sum to 1.0"

    train_files, temp_files = train_test_split(
        files, train_size=train_ratio, random_state=seed
    )
    # val and test share the remaining chunk — split it proportionally
    remaining_ratio = val_ratio / (val_ratio + test_ratio)
    val_files, test_files = train_test_split(
        temp_files, train_size=remaining_ratio, random_state=seed
    )
    return train_files, val_files, test_files


def copy_files(files, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, dest_dir / f.name)


def main():
    parser = argparse.ArgumentParser(description="Split classification dataset into train/val/test")
    parser.add_argument("--input", type=str, required=True, help="Path to raw dataset (one folder per class)")
    parser.add_argument("--output", type=str, required=True, help="Where to write train/val/test folders")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(
            f"{input_dir} does not exist. Did you download and unzip the Kaggle dataset there?"
        )

    print(f"Scanning {input_dir} ...")
    classes = collect_images(input_dir)
    if not classes:
        raise RuntimeError("No class folders with images found. Check --input path and folder structure.")

    print(f"Found {len(classes)} classes: {list(classes.keys())}")

    total_counts = {"train": 0, "val": 0, "test": 0}

    for class_name, files in classes.items():
        train_files, val_files, test_files = split_class_files(
            files, args.train, args.val, args.test, seed=args.seed
        )
        copy_files(train_files, output_dir / "train" / class_name)
        copy_files(val_files, output_dir / "val" / class_name)
        copy_files(test_files, output_dir / "test" / class_name)

        total_counts["train"] += len(train_files)
        total_counts["val"] += len(val_files)
        total_counts["test"] += len(test_files)

        print(f"  {class_name}: {len(files)} total -> "
              f"train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    print("\nDone.")
    print(f"Total: train={total_counts['train']}, val={total_counts['val']}, test={total_counts['test']}")
    print(f"Output written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()