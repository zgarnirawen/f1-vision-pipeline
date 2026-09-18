import argparse
import json
from pathlib import Path

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to processed/classification folder")
    parser.add_argument("--model", required=True, help="Path to trained model .pt file")
    parser.add_argument("--classes", required=True, help="Path to class_names.json")
    parser.add_argument("--out", default="outputs", help="Output directory for results")
    args = parser.parse_args()

    device = torch.device("cpu")

    with open(args.classes) as f:
        class_names = json.load(f)

    # Same eval transform as used for val during training (resize/normalize, no augmentation)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_dir = Path(args.data) / "test"
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    print(f"Loaded {len(test_dataset)} test images, {len(class_names)} classes")

    model = timm.create_model("resnet18.a1_in1k", pretrained=False, num_classes=len(class_names))
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    print("\n" + "=" * 50)
    print("RESNET18 CLASSIFIER — TEST SET RESULTS")
    print("=" * 50)
    print(f"\nAccuracy:  {acc:.3f}")
    print(f"Macro F1:  {macro_f1:.3f}\n")

    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    # Save summary text
    summary_path = out_dir / "classifier_test_results.txt"
    with open(summary_path, "w") as f:
        f.write(f"Accuracy: {acc:.3f}\nMacro F1: {macro_f1:.3f}\n\n{report}")
    print(f"\nSummary saved to: {summary_path}")

    # Save confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title("ResNet18 Classifier — Confusion Matrix (Test Set)")
    plt.tight_layout()
    cm_path = out_dir / "classifier_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    print(f"Confusion matrix saved to: {cm_path}")


if __name__ == "__main__":
    main()