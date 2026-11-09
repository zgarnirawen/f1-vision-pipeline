"""
Evaluate a trained YOLO26 detector on a labeled validation or test split.

Examples:
    python src/evaluate_detector.py \
        --weights runs/detect/f1-yolo26n/weights/best.pt \
        --data data/annotations/data.yaml

    python src/evaluate_detector.py \
        --weights runs/detect/f1-yolo26n/weights/best.pt \
        --data data/annotations/data.yaml \
        --split test \
        --output outputs/detection_test_metrics.json

The test split is optional, but if selected it must be declared in the dataset YAML.
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained F1 car detector.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to best.pt.")
    parser.add_argument("--data", type=Path, required=True, help="Path to YOLO dataset YAML.")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="Examples: 0, cpu. Auto-detected by default.")
    parser.add_argument("--output", type=Path, default=Path("outputs/detection_metrics.json"))
    parser.add_argument("--project", default="runs/val")
    parser.add_argument("--name", default="f1-detector-evaluation")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.weights.is_file():
        raise FileNotFoundError(f"Model weights not found: {args.weights}")
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

    with args.data.open("r", encoding="utf-8") as file:
        dataset = yaml.safe_load(file) or {}

    if not dataset.get(args.split):
        raise ValueError(
            f"The dataset YAML does not define a '{args.split}' split. "
            "For test evaluation, add a labeled test image directory to the YAML."
        )

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    model = YOLO(str(args.weights))
    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        plots=True,
        verbose=True,
        project=args.project,
        name=args.name,
    )

    box = metrics.box
    speed = metrics.speed or {}
    result = {
        "weights": str(args.weights.resolve()),
        "dataset_yaml": str(args.data.resolve()),
        "split": args.split,
        "device": str(device),
        "metrics": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "map50": float(box.map50),
            "map50_95": float(box.map),
        },
        "speed_ms_per_image": {
            key: float(value) for key, value in speed.items()
        },
        "total_reported_ms_per_image": float(sum(speed.values())),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\nDetection evaluation")
    print(json.dumps(result, indent=2))
    print(f"Metrics saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
