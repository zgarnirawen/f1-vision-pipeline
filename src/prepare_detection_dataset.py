"""
Prepare a Roboflow YOLO dataset for object detection.

This project uses YOLO26 detection (not segmentation). Roboflow exports can
contain polygon rows for segmentation. This script converts polygon rows to
their enclosing axis-aligned bounding boxes and preserves existing detection
box rows.

Example:
    python src/prepare_detection_dataset.py \
        --input data/annotations \
        --output data/processed/detection

Expected input:
    data/annotations/
      data.yaml
      train/images, train/labels
      valid/images, valid/labels
      test/images, test/labels

Output:
    data/processed/detection/
      data.yaml
      train/images, train/labels
      valid/images, valid/labels
      test/images, test/labels
"""

import argparse
import shutil
from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_label_line(line: str, label_path: Path, line_number: int):
    """Return a normalized YOLO detection row from a box or polygon row."""
    parts = line.split()
    location = f"{label_path}:{line_number}"

    if len(parts) < 5:
        raise ValueError(f"{location}: expected at least 5 values, got {len(parts)}")

    try:
        class_id = int(parts[0])
        coordinates = [float(value) for value in parts[1:]]
    except ValueError as exc:
        raise ValueError(f"{location}: class ID and coordinates must be numeric") from exc

    if class_id < 0:
        raise ValueError(f"{location}: class ID must be non-negative")

    if any(value < 0 or value > 1 for value in coordinates):
        raise ValueError(f"{location}: source coordinates must be normalized within [0, 1]")

    if len(coordinates) == 4:
        # Standard YOLO detection: cx, cy, width, height.
        cx, cy, width, height = coordinates
    else:
        # YOLO segmentation polygon: x1, y1, x2, y2, ...
        if len(coordinates) < 6 or len(coordinates) % 2 != 0:
            raise ValueError(
                f"{location}: expected 4 box coordinates or at least "
                "3 polygon points (an even number of coordinates)"
            )

        xs = coordinates[0::2]
        ys = coordinates[1::2]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

    values = [cx, cy, width, height]
    if any(value < 0 or value > 1 for value in values):
        raise ValueError(f"{location}: normalized box values must be within [0, 1]")
    if width <= 0 or height <= 0:
        raise ValueError(f"{location}: box width and height must be positive")

    return class_id, values, len(coordinates) != 4


def convert_label_file(source: Path, destination: Path):
    """Convert every row in one label file and return conversion statistics."""
    converted_polygons = 0
    rows_written = 0
    output_lines = []

    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue

        class_id, box, was_polygon = parse_label_line(line, source, line_number)
        converted_polygons += int(was_polygon)
        rows_written += 1
        output_lines.append(
            f"{class_id} " + " ".join(f"{value:.8f}" for value in box)
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "\n".join(output_lines) + ("\n" if output_lines else ""),
        encoding="utf-8",
    )
    return rows_written, converted_polygons


def main():
    parser = argparse.ArgumentParser(
        description="Convert mixed YOLO box/polygon labels to YOLO detection labels."
    )
    parser.add_argument("--input", type=Path, required=True, help="Extracted Roboflow dataset root.")
    parser.add_argument("--output", type=Path, required=True, help="Prepared detection dataset output.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output directory before preparing it.",
    )
    args = parser.parse_args()

    source_root = args.input.resolve()
    output_root = args.output.resolve()
    source_yaml = source_root / "data.yaml"

    if not source_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {source_yaml}")
    if source_root == output_root or source_root in output_root.parents:
        raise ValueError("Output directory must not be the input directory or inside it.")

    with source_yaml.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    names = config.get("names")
    if isinstance(names, list):
        names = {index: name for index, name in enumerate(names)}
    if not isinstance(names, dict) or not names:
        raise ValueError("data.yaml must define class names as a list or mapping.")

    class_ids = {int(class_id) for class_id in names}
    declared_nc = config.get("nc")
    if declared_nc is not None and int(declared_nc) != len(class_ids):
        raise ValueError("The 'nc' value does not match the number of class names.")

    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: {output_root}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)
    total_images = 0
    total_labels = 0
    total_polygons = 0
    split_summary = {}

    for split in ("train", "valid", "test"):
        image_dir = source_root / split / "images"
        label_dir = source_root / split / "labels"

        if not image_dir.is_dir():
            if split == "test":
                continue
            raise FileNotFoundError(f"Missing image directory: {image_dir}")
        if not label_dir.is_dir():
            raise FileNotFoundError(f"Missing label directory: {label_dir}")

        images = sorted(
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError(f"No supported images found in {image_dir}")

        out_images = output_root / split / "images"
        out_labels = output_root / split / "labels"
        out_images.mkdir(parents=True, exist_ok=True)
        out_labels.mkdir(parents=True, exist_ok=True)

        split_polygons = 0
        split_labels = 0

        for image_path in images:
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(
                    f"Missing label for image {image_path.name}: {label_path}"
                )

            rows, polygons = convert_label_file(
                label_path, out_labels / label_path.name
            )
            # Ensure all source class IDs are declared in data.yaml.
            for line_number, line in enumerate(
                (out_labels / label_path.name).read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if line.strip():
                    class_id = int(line.split()[0])
                    if class_id not in class_ids:
                        raise ValueError(
                            f"{label_path}:{line_number}: class {class_id} "
                            f"is not declared in data.yaml ({sorted(class_ids)})"
                        )

            shutil.copy2(image_path, out_images / image_path.name)
            split_labels += rows
            split_polygons += polygons

        split_summary[split] = {
            "images": len(images),
            "annotation_rows": split_labels,
            "polygons_converted_to_boxes": split_polygons,
        }
        total_images += len(images)
        total_labels += split_labels
        total_polygons += split_polygons

    output_config = {
        "path": str(output_root),
        "train": "train/images",
        "val": "valid/images",
        "names": {int(key): value for key, value in names.items()},
    }
    if (output_root / "test" / "images").is_dir():
        output_config["test"] = "test/images"

    with (output_root / "data.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(output_config, file, sort_keys=False, allow_unicode=True)

    for metadata_name in ("README.dataset.txt", "README.roboflow.txt"):
        metadata_path = source_root / metadata_name
        if metadata_path.is_file():
            shutil.copy2(metadata_path, output_root / metadata_name)

    print("Detection dataset prepared successfully.")
    print(f"Output: {output_root}")
    print(f"Images: {total_images}")
    print(f"Annotation rows: {total_labels}")
    print(f"Polygon rows converted to boxes: {total_polygons}")
    for split, summary in split_summary.items():
        print(f"{split}: {summary}")
    print(f"Dataset YAML: {output_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
