"""
Train a YOLO26 detector on an annotated F1-car dataset.

Example:
    python src/train_detector.py \
        --data data/annotations/data.yaml \
        --model yolo26n.pt \
        --epochs 50 \
        --imgsz 640 \
        --batch 16

The dataset must be annotated in Ultralytics YOLO detection format.
This script does not download or fabricate a dataset.
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO


def validate_dataset_yaml(data_yaml: Path) -> dict:
    """Fail early when the dataset config is missing or malformed."""
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml}\n"
            "Create it from configs/f1_detection.yaml.example and update its path."
        )

    with data_yaml.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not config.get("train") or not config.get("val"):
        raise ValueError("Dataset YAML must define both 'train' and 'val' image paths.")

    names = config.get("names")
    if not names:
        raise ValueError("Dataset YAML must define class names under 'names'.")

    number_of_classes = len(names)
    if config.get("nc") is not None and int(config["nc"]) != number_of_classes:
        raise ValueError(
            f"Dataset YAML declares nc={config['nc']} but contains "
            f"{number_of_classes} class names."
        )

    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a YOLO26 model to detect Formula 1 cars."
    )
    parser.add_argument("--data", type=Path, required=True, help="Path to YOLO dataset YAML.")
    parser.add_argument("--model", default="yolo26n.pt", help="Pretrained YOLO26 checkpoint.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="Examples: 0, 0,1, cpu. Auto-detected by default.")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="f1-yolo26n")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume from the supplied --model checkpoint.")
    return parser.parse_args()


def main():
    args = parse_args()
    validate_dataset_yaml(args.data)

    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        raise ValueError("--epochs, --batch must be positive and --imgsz must be at least 32.")

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    print(f"Dataset config: {args.data.resolve()}")
    print(f"Starting model: {args.model}")

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        seed=args.seed,
        deterministic=True,
        plots=True,
        save=True,
        resume=args.resume,
    )

    trainer = getattr(model, "trainer", None)
    if trainer is None or not getattr(trainer, "best", None):
        raise RuntimeError("Training finished but Ultralytics did not expose a best checkpoint.")

    best_weights = Path(trainer.best)
    if not best_weights.is_file():
        raise FileNotFoundError(f"Best checkpoint was not created: {best_weights}")

    print(f"Best checkpoint: {best_weights}")
    print("Running a final validation pass using the best checkpoint...")

    best_model = YOLO(str(best_weights))
    metrics = best_model.val(
        data=str(args.data),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        plots=True,
        verbose=True,
    )

    box = metrics.box
    summary = {
        "model": str(best_weights),
        "dataset_yaml": str(args.data.resolve()),
        "split": "val",
        "epochs_requested": args.epochs,
        "image_size": args.imgsz,
        "batch_size": args.batch,
        "device": str(device),
        "metrics": {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "map50": float(box.map50),
            "map50_95": float(box.map),
        },
        "speed_ms_per_image": {
            key: float(value) for key, value in (metrics.speed or {}).items()
        },
    }

    output_dir = Path(trainer.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nValidation summary")
    print(json.dumps(summary, indent=2))
    print(f"Saved validation summary to: {summary_path}")


if __name__ == "__main__":
    main()
