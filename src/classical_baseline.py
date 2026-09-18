"""
classical_baseline.py

A classical (non-deep-learning) computer vision baseline for F1 team classification,
built entirely from hand-crafted features rather than a learned CNN. This exists to
compare against the ResNet18 transfer-learning approach in train_classifier.py:
"how much does deep learning actually buy you over hand-crafted features here?"

Feature pipeline per image:
  1. Grayscale conversion
  2. Sobel gradient (x and y) -> gradient magnitude -> mean/std as texture/edge features
  3. Laplacian -> mean/std as a second edge-response feature
  4. Otsu threshold + morphological closing (dilate then erode) to merge broken edges
     into a single closed car silhouette
  5. Largest contour extraction -> bounding-box aspect ratio, area ratio, and the 7
     Hu moments (log-scaled, so they're invariant to translation/scale/rotation)
  6. HSV color histogram (livery color is one of the most distinctive signals available
     without any learning at all -- Ferrari red vs. Williams blue vs. Mercedes silver)

All of these features are concatenated into one vector per image, then classified with
a classical (non-neural) classifier: SVM (RBF kernel) and k-NN, so you get a direct
comparison of which classical method does better, alongside the deep learning result.

Usage:
    python src/classical_baseline.py --data data/processed/classification
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import matplotlib
matplotlib.use("Agg")  # no display needed, just save figures
import matplotlib.pyplot as plt

IMG_SIZE = 256  # resize target before feature extraction, keeps runtime consistent


def load_image(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img


def extract_edge_features(gray):
    """Sobel + Laplacian summary statistics."""
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    return [
        magnitude.mean(), magnitude.std(),
        np.abs(laplacian).mean(), laplacian.std(),
    ]


def extract_contour_features(gray):
    """
    Otsu threshold -> morphological closing -> largest contour -> shape descriptors.
    Morphological closing (dilate then erode) is what bridges small gaps in the
    thresholded silhouette so the car outline forms one connected contour instead
    of several fragments.
    """
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((7, 7), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # No contour found (blank/corrupt image) -- return zeros rather than crashing
        return [0.0, 0.0] + [0.0] * 7

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    aspect_ratio = w / h if h > 0 else 0.0
    area_ratio = cv2.contourArea(largest) / (IMG_SIZE * IMG_SIZE)

    moments = cv2.moments(largest)
    hu = cv2.HuMoments(moments).flatten()
    # Log-scale the Hu moments -- they span many orders of magnitude otherwise,
    # which would let them dominate the feature vector unfairly.
    hu_log = [-np.sign(h) * np.log10(abs(h) + 1e-10) for h in hu]

    return [aspect_ratio, area_ratio] + hu_log


def extract_color_histogram(img, bins=16):
    """HSV hue histogram -- captures livery color, which is highly distinctive
    for team identification even without any shape/texture information at all."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.tolist()


def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = []
    features += extract_edge_features(gray)
    features += extract_contour_features(gray)
    features += extract_color_histogram(img)
    return features


def load_split(split_dir: Path):
    valid_ext = {".jpg", ".jpeg", ".png", ".webp"}
    X, y, class_names = [], [], []

    class_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]

    for class_idx, class_dir in enumerate(class_dirs):
        files = [f for f in class_dir.iterdir() if f.suffix.lower() in valid_ext]
        for f in files:
            img = load_image(f)
            if img is None:
                print(f"  [warn] could not read {f}, skipping")
                continue
            X.append(extract_features(img))
            y.append(class_idx)

    return np.array(X), np.array(y), class_names


def main():
    parser = argparse.ArgumentParser(description="Classical CV baseline for F1 team classification")
    parser.add_argument("--data", type=str, required=True, help="Path to processed dataset (train/val/test folders)")
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting features from train set...")
    X_train, y_train, class_names = load_split(data_dir / "train")
    print(f"  {len(X_train)} training images, {len(class_names)} classes")

    print("Extracting features from test set...")
    X_test, y_test, _ = load_split(data_dir / "test")
    print(f"  {len(X_test)} test images")

    results = {}

    # --- SVM (RBF kernel) ---
    print("\nTraining SVM (RBF kernel)...")
    svm_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf")),
    ])
    # Small grid search over the two most impactful SVM hyperparameters
    svm_grid = GridSearchCV(
        svm_pipeline,
        param_grid={"svm__C": [1, 10, 100], "svm__gamma": ["scale", 0.01, 0.001]},
        cv=3,
        n_jobs=-1,
    )
    svm_grid.fit(X_train, y_train)
    svm_pred = svm_grid.predict(X_test)
    results["SVM"] = {
        "accuracy": accuracy_score(y_test, svm_pred),
        "macro_f1": f1_score(y_test, svm_pred, average="macro"),
        "best_params": svm_grid.best_params_,
        "predictions": svm_pred,
    }

    # --- k-NN ---
    print("Training k-NN...")
    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ])
    knn_grid = GridSearchCV(
        knn_pipeline,
        param_grid={"knn__n_neighbors": [3, 5, 7, 11]},
        cv=3,
        n_jobs=-1,
    )
    knn_grid.fit(X_train, y_train)
    knn_pred = knn_grid.predict(X_test)
    results["kNN"] = {
        "accuracy": accuracy_score(y_test, knn_pred),
        "macro_f1": f1_score(y_test, knn_pred, average="macro"),
        "best_params": knn_grid.best_params_,
        "predictions": knn_pred,
    }

    # --- Report ---
    print("\n" + "=" * 50)
    print("CLASSICAL CV BASELINE RESULTS")
    print("=" * 50)
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  Accuracy:  {res['accuracy']:.3f}")
        print(f"  Macro F1:  {res['macro_f1']:.3f}")
        print(f"  Best params: {res['best_params']}")

    # Use whichever classical model did better for the detailed report + confusion matrix
    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    best_pred = results[best_name]["predictions"]
    print(f"\nBest classical model: {best_name}")
    print("\n" + classification_report(y_test, best_pred, target_names=class_names))

    cm = confusion_matrix(y_test, best_pred)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(xticks_rotation=45)
    plt.tight_layout()
    cm_path = out_dir / "classical_confusion_matrix.png"
    plt.savefig(cm_path)
    print(f"\nConfusion matrix saved to: {cm_path}")

    # Save a plain-text summary for easy copy into the README
    summary_path = out_dir / "classical_baseline_results.txt"
    with open(summary_path, "w") as f:
        f.write("Classical CV Baseline Results\n")
        f.write("=" * 40 + "\n")
        for name, res in results.items():
            f.write(f"{name}: accuracy={res['accuracy']:.3f}, macro_f1={res['macro_f1']:.3f}, "
                     f"best_params={res['best_params']}\n")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()