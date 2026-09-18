"""
train_classifier.py

Fine-tunes a pretrained ResNet18 (via timm) to classify F1 team liveries.
Freezes the backbone, trains only the last block + a new classifier head —
this is the right approach for a small dataset (~350 images total).

Usage:
    python src/train_classifier.py \
        --data data/processed/classification \
        --epochs 25 \
        --batch-size 16 \
        --lr 1e-3 \
        --out models/classifier.pt
"""

import argparse
import json
from pathlib import Path

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def build_transforms():
    # ImageNet normalization stats — required since we're using an ImageNet-pretrained backbone
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        normalize,
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_tf, eval_tf


def build_model(num_classes: int, device):
    model = timm.create_model("resnet18", pretrained=True, num_classes=num_classes)

    # Freeze everything except the final residual block ("layer4") and the classifier head.
    # This is the right amount of fine-tuning for ~300-400 images — training the whole
    # network from scratch (or unfreezing everything) would overfit fast.
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return model.to(device)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    torch.set_grad_enabled(train)
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    torch.set_grad_enabled(True)
    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet18 for F1 team classification")
    parser.add_argument("--data", type=str, required=True, help="Path to processed dataset (train/val/test folders)")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="models/classifier.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cpu":
        print("  [note] No GPU detected — run this in Google Colab (free T4) for reasonable speed.")

    data_dir = Path(args.data)
    train_tf, eval_tf = build_transforms()

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tf)

    class_names = train_ds.classes
    print(f"Classes ({len(class_names)}): {class_names}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = build_model(num_classes=len(class_names), device=device)

    criterion = nn.CrossEntropyLoss()
    # Only optimize the unfrozen parameters (layer4 + fc)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    best_val_acc = 0.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        print(f"Epoch {epoch:2d}/{args.epochs} | "
              f"train_loss={train_loss:.3f} train_acc={train_acc:.3f} | "
              f"val_loss={val_loss:.3f} val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), out_path)
            print(f"  -> new best val_acc={val_acc:.3f}, saved to {out_path}")

    # Save class names alongside the model — predict_classifier.py needs this mapping
    class_map_path = out_path.parent / "class_names.json"
    with open(class_map_path, "w") as f:
        json.dump(class_names, f, indent=2)

    print(f"\nTraining done. Best val accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: {out_path}")
    print(f"Class names saved to: {class_map_path}")


if __name__ == "__main__":
    main()