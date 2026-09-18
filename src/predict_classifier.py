"""
predict_classifier.py

Loads the trained classifier and predicts the team for a single image.

Usage:
    python src/predict_classifier.py --image path/to/car.jpg --model models/classifier.pt
"""

import argparse
import json
from pathlib import Path

import timm
import torch
from PIL import Image
from torchvision import transforms


def load_model(model_path: Path, class_names_path: Path, device):
    with open(class_names_path) as f:
        class_names = json.load(f)

    model = timm.create_model("resnet18", pretrained=False, num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, class_names


def preprocess(image_path: Path):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    image = Image.open(image_path).convert("RGB")
    return tf(image).unsqueeze(0)  # add batch dimension


def main():
    parser = argparse.ArgumentParser(description="Predict F1 team from an image")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default="models/classifier.pt")
    parser.add_argument("--class-names", type=str, default=None,
                         help="Defaults to class_names.json next to --model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(args.model)
    class_names_path = Path(args.class_names) if args.class_names else model_path.parent / "class_names.json"

    model, class_names = load_model(model_path, class_names_path, device)

    image_tensor = preprocess(Path(args.image)).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        top_idx = probs.argmax().item()

    print(f"Predicted team: {class_names[top_idx]}")
    print(f"Confidence: {probs[top_idx].item():.1%}")

    # Show top-3 for context — useful when the model is unsure between similar liveries
    top3 = torch.topk(probs, k=min(3, len(class_names)))
    print("\nTop 3:")
    for prob, idx in zip(top3.values, top3.indices):
        print(f"  {class_names[idx]:15s} {prob.item():.1%}")


if __name__ == "__main__":
    main()