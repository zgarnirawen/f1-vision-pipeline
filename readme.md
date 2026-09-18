# F1 Vision Pipeline — Complete Project Scope

## 1. Project Overview

Build an end-to-end Formula 1 computer-vision pipeline combining:

* Classical Computer Vision
* Deep Learning
* Object Detection
* Object Tracking
* Image Classification
* Real F1 telemetry data
* Video processing
* API deployment
* Docker
* A simple interactive demonstration interface

The system should take Formula 1 race footage, detect and track F1 cars, identify the team/livery of each detected car, and optionally associate the video with F1 telemetry such as speed, throttle, and braking.

The final system should demonstrate the complete workflow:

```text
Race Video
    │
    ▼
YOLO Object Detection
    │
    ▼
Multi-Object Tracking
    │
    ▼
Car Crop
    │
    ▼
Team Classification
    │
    ▼
Track ID ↔ Team Association
    │
    ▼
Telemetry Fusion
    │
    ▼
Annotated Video
    │
    ▼
FastAPI + Streamlit Demo
    │
    ▼
Dockerized Deployment
```

The project should also include a classical computer-vision baseline using hand-crafted features and traditional machine-learning classifiers. This baseline must be evaluated on the same classification task as the deep-learning model so that the project can quantitatively demonstrate what transfer learning adds over classical image processing.

The project is intended to be portfolio/CV quality and should emphasize the complete engineering pipeline rather than only model training.

---

# 2. Main Objectives

The project must achieve the following objectives:

1. Build a dataset for F1 team classification.
2. Implement a classical CV classification baseline.
3. Implement a deep-learning classifier using transfer learning.
4. Compare the classical and deep-learning approaches using quantitative metrics.
5. Train an object-detection model for F1 cars.
6. Add multi-object tracking so individual cars retain persistent IDs across frames.
7. Combine detection, tracking, and team classification into one video pipeline.
8. Integrate F1 telemetry data using FastF1.
9. Overlay telemetry information onto processed video.
10. Expose the inference pipeline through a FastAPI service.
11. Build a lightweight Streamlit interface for demonstration.
12. Containerize the application with Docker.
13. Provide reproducible documentation and evaluation results.
14. Publish the project on GitHub with a strong README and demonstration material.

---

# 3. Technology Stack

| Layer               | Technology                       | Purpose                                                           |
| ------------------- | -------------------------------- | ----------------------------------------------------------------- |
| Language            | Python 3.10+                     | Main development language                                         |
| Computer Vision     | OpenCV                           | Image/video processing, gradients, morphology, contours, overlays |
| Classical ML        | scikit-learn                     | SVM and k-NN classifiers                                          |
| Deep Learning       | PyTorch                          | Neural-network training                                           |
| Model library       | timm                             | Pretrained ResNet/EfficientNet models                             |
| Object Detection    | Ultralytics YOLO26               | F1 car detection                                                  |
| Tracking            | ByteTrack through Ultralytics    | Persistent object IDs                                             |
| Annotation          | Roboflow                         | Object-detection annotation/export                                |
| Telemetry           | FastF1                           | F1 session and car telemetry                                      |
| Data                | pandas / NumPy                   | Data processing                                                   |
| Visualization       | Matplotlib                       | Evaluation plots/confusion matrices                               |
| Video               | FFmpeg + OpenCV                  | Frame extraction and video generation                             |
| API                 | FastAPI                          | Model inference API                                               |
| Demo                | Streamlit                        | Interactive frontend                                              |
| Containerization    | Docker                           | Reproducible deployment                                           |
| Experiment Tracking | Weights & Biases                 | Optional experiment tracking                                      |
| GPU                 | Google Colab T4 / Kaggle GPU     | Model training                                                    |
| Version Control     | Git + GitHub                     | Source control and portfolio                                      |
| Deployment          | Hugging Face Spaces / own server | Public demonstration                                              |

---

# 4. Repository Structure

Create the following repository:

```text
f1-vision-pipeline/
│
├── README.md
├── requirements.txt
├── Dockerfile
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── annotations/
│
├── notebooks/
│   └── training notebooks
│
├── src/
│   ├── split_dataset.py
│   ├── clean_dataset.py
│   ├── classical_baseline.py
│   ├── train_classifier.py
│   ├── evaluate_classifier.py
│   ├── predict_classifier.py
│   ├── pipeline.py
│   └── telemetry_fusion.py
│
├── models/
│
├── outputs/
│   ├── confusion matrices
│   ├── evaluation reports
│   ├── demo videos
│   └── sample predictions
│
└── app/
    ├── main.py
    └── streamlit_app.py
```

Large datasets, videos, virtual environments, and model files should not be committed directly to Git unless appropriate storage such as Git LFS is used.

---

# 5. Environment Setup

Create the project:

```bash
mkdir f1-vision-pipeline
cd f1-vision-pipeline
```

Create the directories:

```bash
mkdir -p data/raw data/processed data/annotations notebooks src models outputs app
```

Create a Python virtual environment:

```bash
python3 -m venv venv
```

Activate it:

Linux/macOS:

```bash
source venv/bin/activate
```

Windows:

```powershell
venv\Scripts\activate
```

Install the required packages:

```bash
pip install ultralytics torch torchvision opencv-python fastf1 pandas numpy matplotlib streamlit fastapi uvicorn roboflow timm scikit-learn
```

Save the environment:

```bash
pip freeze > requirements.txt
```

The `.gitignore` should include:

```text
venv/
data/raw/
*.mp4
*.pt
__pycache__/
.ipynb_checkpoints/
```

---

# 6. Phase 1 — F1 Team Image Classification

## Objective

Given a cropped image of an F1 car, predict the team it belongs to.

Use the Formula One Cars dataset containing approximately 350 images across 8 team classes.

The initial dataset classes are:

```text
8 F1 teams
~350 images
```

Download and extract the dataset using Kaggle:

```bash
pip install kaggle
```

Configure the Kaggle API credentials and download the dataset:

```bash
kaggle datasets download -d vesuvius13/formula-one-cars
```

Extract it:

```bash
unzip formula-one-cars.zip -d data/raw/classification
```

---

# 7. Dataset Splitting

Create:

```text
data/processed/classification/
├── train/
├── val/
└── test/
```

Use:

```text
70% training
15% validation
15% testing
```

The split must be performed independently per class to maintain approximately balanced class distributions.

Implement:

```text
src/split_dataset.py
```

using `sklearn.model_selection.train_test_split`.

---

# 8. Dataset Cleaning

Before training, validate every image.

Use PIL image verification to identify corrupted or invalid files.

The existing experiment identified:

* 1 genuinely truncated file
* 2 files with image extensions that were actually HTML pages
* These represented less than 0.2% of the dataset

The cleaning procedure must be reproducible through:

```text
src/clean_dataset.py
```

Do not rely only on file existence or file size. Verify that files are actually valid images.

---

# 9. Classical Computer-Vision Baseline

## Objective

Build a traditional computer-vision classifier without deep learning or learned convolutional filters.

The purpose is to establish a baseline against which the neural-network classifier can be compared.

The classical pipeline must use hand-crafted features.

---

## 9.1 Feature Extraction

For each image, extract the following features.

### Sobel gradients

Calculate:

* Sobel X
* Sobel Y
* gradient magnitude

Extract summary statistics such as:

* mean
* standard deviation

This captures edge strength and texture.

### Laplacian

Calculate the Laplacian response to provide a second edge-related signal.

### Otsu thresholding

Apply Otsu thresholding to create a binary representation.

### Morphological closing

Apply:

```text
dilation → erosion
```

to bridge small gaps in the thresholded representation.

The purpose is to obtain a more coherent silhouette for contour extraction.

### Contours

Use:

```python
cv2.findContours()
```

Extract:

* bounding-box aspect ratio
* contour area ratio
* 7 Hu moments

Apply log scaling to the Hu moments where appropriate to improve numerical stability and provide approximate scale/rotation invariance.

### HSV color histogram

Extract an HSV histogram to capture the dominant livery colors.

This feature is especially important because different teams have visually distinctive liveries.

---

# 10. Classical Classifiers

Train two traditional classifiers:

## SVM

Use an RBF-kernel Support Vector Machine.

Perform a small grid search over:

```text
C
gamma
```

## k-NN

Use k-nearest neighbors.

Perform a grid search over:

```text
n_neighbors
```

All models must be evaluated on the same held-out test set.

Implement:

```text
src/classical_baseline.py
```

Run:

```bash
python src/classical_baseline.py --data data/processed/classification
```

The script must:

1. Load the dataset.
2. Extract classical features.
3. Train SVM.
4. Tune SVM parameters.
5. Train k-NN.
6. Tune k-NN parameters.
7. Calculate accuracy.
8. Calculate macro F1.
9. Generate classification reports.
10. Generate confusion matrices.
11. Save the results to `outputs/`.

---

# 11. Deep-Learning Classification

Implement transfer learning using a pretrained ResNet18 or EfficientNet-B0.

Use:

```python
timm.create_model(
    "resnet18",
    pretrained=True,
    num_classes=8
)
```

The training process should:

1. Load a pretrained model.
2. Freeze early layers initially.
3. Fine-tune the final block and classification head.
4. Use data augmentation.
5. Train for approximately 15–30 epochs.
6. Monitor validation accuracy.
7. Save the best validation checkpoint.

Recommended augmentations:

```text
Random flip
Color jitter
Small rotation
Random crop
```

Implement:

```text
src/train_classifier.py
```

Save:

```text
models/classifier.pt
```

---

# 12. Existing Deep-Learning Results

The ResNet18 model achieved:

```text
Best validation accuracy: 88.9%
```

The final test evaluation achieved:

```text
Accuracy: 86.8%
Macro F1: 0.871
```

The training accuracy eventually reached approximately:

```text
98–99%
```

while validation accuracy remained approximately:

```text
85–89%
```

This indicates some overfitting.

The reported model must use the best-validation checkpoint rather than simply the final epoch.

---

# 13. Classification Evaluation

Implement:

```text
src/evaluate_classifier.py
```

The script must load the trained checkpoint and report:

* Test accuracy
* Macro F1
* Per-class precision
* Per-class recall
* Per-class F1
* Confusion matrix

The classical and deep-learning models must be evaluated on the same test set.

---

# 14. Existing Classification Results

The current results are:

| Model                      | Accuracy | Macro F1 |
| -------------------------- | -------: | -------: |
| Classical SVM              |    39.3% |    0.382 |
| Classical k-NN             |    29.9% |    0.288 |
| ResNet18 Transfer Learning |    86.8% |    0.871 |

The best classical model is the SVM.

The ResNet18 therefore produced approximately a:

```text
47.5 percentage-point accuracy improvement
```

over the classical SVM.

Random chance for 8 classes is:

```text
12.5%
```

The classical model performs substantially above random chance, demonstrating that the hand-crafted features contain useful information, but its performance remains significantly below the transfer-learning model.

---

# 15. Confusion-Matrix Analysis

The project must not report only a single accuracy number.

Analyze the actual errors.

## Classical SVM

The largest observed confusion was:

```text
AlphaTauri → Williams
```

with:

```text
9 of 19 AlphaTauri test images
```

misclassified as Williams.

Another notable confusion was:

```text
Racing Point → Ferrari
```

with:

```text
4 of 41
```

misclassified.

Williams had the lowest recall under the classical model:

```text
14 / 50
28% recall
```

Its errors were distributed across several classes including:

```text
McLaren
Red Bull
Renault
```

This indicates that the classical pipeline was heavily dependent on color information.

---

# 16. ResNet18 Confusion-Matrix Analysis

The ResNet18 model eliminated the two most notable color-driven confusions observed in the classical model:

```text
AlphaTauri ↔ Williams
```

and:

```text
Racing Point ↔ Ferrari
```

AlphaTauri improved from approximately:

```text
16% precision / 16% recall
```

under the classical model to:

```text
86% precision / 95% recall
```

under ResNet18.

This demonstrates that learned spatial features can capture information beyond raw color histograms, such as:

* sponsor placement
* stripe geometry
* car shape
* wing structure
* spatial arrangement of visual features

However, the neural model introduced a different weakness.

Williams became a false-positive sink.

The ResNet18 model produced approximately:

```text
69% precision
92% recall
```

for Williams.

False positives came from:

```text
McLaren
Mercedes
Racing Point
Red Bull Racing
Renault
```

Therefore, the CNN did not simply reduce every type of error. It solved specific color-driven confusions while introducing a different over-prediction pattern.

Mercedes was another relatively weak class, with approximately:

```text
77% recall
```

The final report should present this as an actual model limitation rather than hiding it.

---

# 17. Classification Prediction Script

Implement:

```text
src/predict_classifier.py
```

The script must accept an image path:

```bash
python src/predict_classifier.py path/to/image.jpg
```

and return:

```text
Predicted team
Confidence
```

Example conceptual output:

```text
Predicted team: Ferrari
Confidence: 0.94
```

---

# 18. Phase 2 — F1 Object Detection

## Objective

Detect F1 cars in real race footage.

This phase moves from clean/cropped classification images to real-world video containing:

* motion
* different camera angles
* occlusion
* multiple cars
* changing lighting
* small objects
* background clutter

Use an F1 car object-detection dataset.

A fast approach is to use a Roboflow Universe Formula 1 car detection dataset and export it in YOLO format.

The resulting structure should be:

```text
data/annotations/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

---

# 19. Optional Custom Detection Dataset

If possible, add approximately:

```text
100–200 custom frames
```

from legally reusable F1 footage.

The objective is to make the dataset more original and representative of the intended video pipeline.

Extract frames using FFmpeg:

```bash
ffmpeg -i data/raw/video.mp4 \
-vf "select=not(mod(n\,15))" \
-vsync vfr \
data/raw/frames/frame_%04d.jpg
```

Annotate the selected frames with bounding boxes using Roboflow.

This step is optional if time is limited.

---

# 20. YOLO26 Training

Use the nano model for efficient training:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.train(
    data="data/annotations/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16
)
```

Training should be performed on a GPU such as a Google Colab T4.

Avoid large models unless additional time and compute are available.

---

# 21. Detection Evaluation

Evaluate the trained detector:

```python
metrics = model.val()
```

Record:

```text
mAP50
mAP50-95
Precision
Recall
Inference latency
```

Target metrics:

| Metric            |                    Target |
| ----------------- | ------------------------: |
| mAP50             |                    ≥ 0.80 |
| mAP50-95          |                    ≥ 0.60 |
| Precision         |                    ≥ 0.80 |
| Recall            |                    ≥ 0.80 |
| Inference latency | <100 ms/frame on Colab T4 |

Do not invent values if the model has not yet been trained. Replace the placeholders with real measured results.

---

# 22. Multi-Object Tracking

Use YOLO tracking rather than plain frame-by-frame prediction.

Example:

```python
results = model.track(
    "data/raw/video.mp4",
    save=True,
    tracker="bytetrack.yaml"
)
```

The purpose is to assign persistent IDs to cars.

Conceptually:

```text
Frame 1:
Car ID 3

Frame 2:
Car ID 3

Frame 3:
Car ID 3
```

rather than treating the car as a new object in every frame.

Tracking allows the classification system to associate team predictions with a persistent vehicle.

---

# 23. Phase 3 — Detection + Tracking + Classification

Implement:

```text
src/pipeline.py
```

The end-to-end pipeline must perform the following operations.

### Step 1

Run YOLO detection/tracking.

Obtain:

```text
bounding box
track ID
```

### Step 2

Crop each detected car from the original frame.

### Step 3

Send the crop to the ResNet18 classifier.

### Step 4

Associate the prediction with the corresponding track ID.

Maintain a structure conceptually similar to:

```python
{
    track_id: [
        predictions
    ]
}
```

### Step 5

Use temporal aggregation such as majority voting across frames.

This prevents one incorrectly classified frame from immediately changing the team identity.

### Step 6

Draw onto the frame:

```text
Bounding box
Track ID
Predicted team
Confidence
```

### Step 7

Process the complete video.

### Step 8

Save the annotated video using:

```python
cv2.VideoWriter
```

The final output should look conceptually like:

```text
┌─────────────────────────────────────┐
│                                     │
│        F1 race footage              │
│                                     │
│   ┌───────────────┐                 │
│   │ Car #3        │                 │
│   │ Ferrari 94%   │                 │
│   └───────────────┘                 │
│                                     │
│   ┌───────────────┐                 │
│   │ Car #7        │                 │
│   │ Mercedes 87%  │                 │
│   └───────────────┘                 │
│                                     │
└─────────────────────────────────────┘
```

This annotated video is one of the most important portfolio deliverables.

---

# 24. Phase 4 — Telemetry Fusion

## Objective

Add real F1 telemetry to the computer-vision pipeline.

Use FastF1.

Example:

```python
import fastf1

fastf1.Cache.enable_cache(
    "data/raw/fastf1_cache"
)

session = fastf1.get_session(
    2023,
    "Monza",
    "R"
)

session.load()

lap = session.laps.pick_fastest()

telemetry = lap.get_car_data().add_distance()
```

The resulting telemetry should contain values such as:

```text
Speed
Throttle
Brake
Time
```

---

# 25. Telemetry Synchronization

The project must be explicit about synchronization limitations.

Arbitrary online race footage and FastF1 telemetry are not automatically frame-synchronized.

There are two valid approaches.

## Simplified approach

Use:

```text
race video from a specific race
+
telemetry from the same race
```

and align them approximately by time.

The README must explicitly state that this is illustrative rather than frame-perfect.

## Rigorous approach

Use an identifiable onboard clip and manually establish synchronization points.

For example:

```text
start/finish line
known lap event
specific track position
```

Then align video timestamps with telemetry timestamps.

If the rigorous approach is not implemented, do not claim frame-perfect synchronization.

---

# 26. Telemetry Overlay

Implement:

```text
src/telemetry_fusion.py
```

For every video frame:

1. Determine the corresponding video timestamp.
2. Find the nearest telemetry timestamp.
3. Retrieve telemetry values.
4. Draw the values onto the frame.

The overlay should display values such as:

```text
Speed: 287 km/h
Throttle: 100%
Brake: OFF
```

The overlay should be generated using OpenCV.

The resulting telemetry-enhanced video should be saved in:

```text
outputs/
```

---

# 27. Phase 5 — API

Create:

```text
app/main.py
```

using FastAPI.

Basic structure:

```python
from fastapi import FastAPI, UploadFile

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile):
    # Save upload
    # Run detection
    # Run classification
    # Run telemetry fusion if applicable
    # Return results
    ...
```

The API should expose inference functionality.

The response should contain structured prediction information such as:

```json
{
  "detections": [
    {
      "track_id": 3,
      "team": "Ferrari",
      "confidence": 0.94
    }
  ]
}
```

The exact response format can be adapted to the implemented pipeline.

---

# 28. Streamlit Demo

Create:

```text
app/streamlit_app.py
```

The purpose is to provide a simple recruiter-friendly demonstration without spending time building an unnecessary frontend framework.

The interface should allow the user to:

1. Upload an image or video.
2. Run the model.
3. Display detection results.
4. Display team classification.
5. Display telemetry if available.
6. Display the annotated output.

Conceptually:

```python
import streamlit as st

st.title("F1 Vision Pipeline")

uploaded = st.file_uploader(
    "Upload race footage",
    type=["jpg", "jpeg", "png", "mp4"]
)

# Run inference
# Display results
```

Do not build a React/Angular frontend for this project unless there is a specific reason. The main value of this project is the ML/CV pipeline and deployment engineering.

---

# 29. Dockerization

Create:

```text
Dockerfile
```

Example baseline:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

The container must provide a reproducible environment for inference.

---

# 30. Deployment

Possible deployment targets:

```text
Hugging Face Spaces
Railway
Render
Own server
```

Hugging Face Spaces can be used for a simple public portfolio demonstration.

Alternatively, deployment to a personal/server environment can demonstrate DevOps capabilities.

If deploying to a server, document:

```text
Docker
CI/CD
environment variables
model management
logging
health checks
```

---

# 31. Final Functional Pipeline

The completed project should support this complete flow:

```text
Input F1 Video
       │
       ▼
Frame Extraction
       │
       ▼
YOLO26 Detection
       │
       ▼
ByteTrack Tracking
       │
       ▼
Track ID
       │
       ▼
Car Crop
       │
       ▼
ResNet18 Team Classification
       │
       ▼
Temporal Majority Voting
       │
       ▼
Team + Confidence
       │
       ▼
FastF1 Telemetry Lookup
       │
       ▼
Speed / Throttle / Brake
       │
       ▼
OpenCV Overlay
       │
       ▼
Annotated Video
```

---

# 32. Required Evaluation

The final README must contain quantitative evaluation.

## Classification

Use:

```text
Accuracy
Macro F1
Precision
Recall
Confusion Matrix
```

Current measured results:

| Approach       | Accuracy | Macro F1 |
| -------------- | -------: | -------: |
| Classical SVM  |    39.3% |    0.382 |
| Classical k-NN |    29.9% |    0.288 |
| ResNet18       |    86.8% |    0.871 |

---

## Detection

Report:

```text
mAP50
mAP50-95
Precision
Recall
Latency
```

Do not fill these values until the detection experiment has actually been performed.

---

# 33. Required Confusion Matrices

Save:

```text
outputs/classical_confusion_matrix.png
outputs/classifier_confusion_matrix.png
```

The README should display both matrices side-by-side.

The analysis should explain not only how many predictions were correct, but also what kinds of mistakes the models make.

---

# 34. Limitations

The final project must explicitly document the following limitations.

### Image quality

Classification performance can decrease when:

* cars are partially occluded
* lighting is poor
* the car is viewed from unusual angles
* the livery is not clearly visible

### Tracking

Tracking can lose or swap IDs during heavy occlusion, especially when cars overlap.

### Telemetry

Telemetry synchronization is approximate unless explicit manual synchronization has been performed.

### Season dependence

F1 liveries change between seasons.

Therefore, a classifier trained on one season should not automatically be considered valid for another season.

### Dataset quality

Several invalid images were discovered during dataset cleaning:

```text
1 truncated file
2 HTML files masquerading as image files
```

These were detected through actual image verification.

### Overfitting

The ResNet18 model reached approximately:

```text
98–99% training accuracy
```

while validation accuracy remained around:

```text
85–89%
```

This indicates overfitting.

Potential future improvements include:

```text
Stronger augmentation
Random crops
Color jitter
Early stopping
More training data
Class balancing
Regularization
```

### Williams over-prediction

The ResNet18 model showed:

```text
Williams:
69% precision
92% recall
```

This indicates that while real Williams examples were usually detected correctly, the model also predicted Williams for a significant number of cars from other teams.

This should be presented as a concrete model weakness.

---

# 35. GitHub Quality Requirements

The repository must contain:

```text
README.md
requirements.txt
Dockerfile
.gitignore
src/
app/
outputs/
```

The README must include:

1. Project description.
2. Architecture diagram.
3. Technology stack.
4. Dataset description.
5. Classical baseline.
6. Deep-learning approach.
7. Detection approach.
8. Tracking approach.
9. Telemetry integration.
10. Quantitative evaluation.
11. Confusion matrices.
12. Limitations.
13. Installation instructions.
14. Usage instructions.
15. Demo video/GIF.
16. Deployment information.
17. Future improvements.

The Git history should contain meaningful commits rather than a single final upload.

---

# 36. Portfolio Demonstration

The most important public-facing artifact should be a short GIF or video showing the complete pipeline.

The demonstration should show:

```text
F1 race footage
↓
Detected cars
↓
Persistent track IDs
↓
Predicted teams
↓
Confidence scores
↓
Telemetry overlay
```

A recruiter should be able to understand what the project does within a few seconds.

The README should place this demonstration near the top.

---

# 37. Suggested Final README Structure

```text
# F1 Vision Pipeline

Short demo/GIF

## Overview

## Architecture

## Features

## Tech Stack

## Dataset

## Classical CV Baseline

## Deep Learning Classifier

## Object Detection

## Multi-Object Tracking

## Telemetry Fusion

## End-to-End Pipeline

## Evaluation

### Classification Results

### Confusion Matrix Analysis

### Detection Results

## Limitations

## Installation

## Usage

## API

## Docker

## Deployment

## Future Improvements
```

---

# 38. Development Order

Implement the project in this order:

```text
1. Repository setup
2. Dataset download
3. Dataset cleaning
4. Dataset splitting
5. Classical CV feature extraction
6. SVM baseline
7. k-NN baseline
8. Classical evaluation
9. ResNet18 training
10. ResNet18 evaluation
11. Confusion-matrix comparison
12. Detection dataset preparation
13. YOLO26 training
14. Detection evaluation
15. ByteTrack integration
16. Detection + classification pipeline
17. Temporal team prediction
18. Annotated video generation
19. FastF1 telemetry retrieval
20. Telemetry synchronization
21. Telemetry overlay
22. FastAPI
23. Streamlit
24. Docker
25. Deployment
26. README polishing
27. Demo recording
28. GitHub publication
```

---

# 39. Time-Constraint Strategy

If time becomes limited, prioritize:

```text
Priority 1:
Classification + evaluation

Priority 2:
Object detection

Priority 3:
Detection + tracking + classification

Priority 4:
Telemetry

Priority 5:
API + Streamlit + Docker + deployment
```

Phases 1–3 already constitute a complete and presentable computer-vision project.

Telemetry and deployment are enhancements that strengthen the portfolio but should not compromise the quality of the core ML pipeline.

If annotation becomes too time-consuming, use the existing annotated dataset and document that it was used.

Use the YOLO nano model rather than a larger model when compute/time is limited.

---

# 40. Final Success Criteria

The project is considered complete when:

* [ ] Repository is structured and reproducible.
* [ ] Dataset is cleaned and split.
* [x] Classical SVM baseline implemented.
* [x] Classical k-NN baseline implemented.
* [x] ResNet18 classifier trained.
* [x] Classification evaluation completed.
* [x] Confusion matrices generated.
* [x] Classical vs deep-learning comparison documented.
* [ ] F1 detection dataset prepared.
* [ ] YOLO26 detector trained.
* [ ] Detection metrics measured.
* [ ] ByteTrack tracking implemented.
* [ ] Detection + classification pipeline implemented.
* [ ] Track IDs associated with team predictions.
* [ ] Annotated demonstration video generated.
* [ ] FastF1 telemetry retrieved.
* [ ] Telemetry overlay implemented.
* [ ] Synchronization limitation documented honestly.
* [ ] FastAPI inference endpoint implemented.
* [ ] Streamlit demo implemented.
* [ ] Docker image built successfully.
* [ ] Application deployed or made reproducible locally.
* [ ] README polished.
* [ ] Demo GIF/video added.
* [ ] GitHub repository published.

---

# 41. Current Project Status

Completed:

```text
[x] Project structure
[x] Dataset preparation
[x] Dataset cleaning
[x] Train/validation/test split
[x] Classical CV feature extraction
[x] SVM baseline
[x] k-NN baseline
[x] ResNet18 transfer-learning classifier
[x] Test evaluation
[x] Macro F1 evaluation
[x] Confusion matrices
[x] Classical vs deep-learning analysis
[x] Overfitting analysis
```

Current measured classification results:

```text
SVM:
Accuracy = 39.3%
Macro F1 = 0.382

k-NN:
Accuracy = 29.9%
Macro F1 = 0.288

ResNet18:
Accuracy = 86.8%
Macro F1 = 0.871
```

Remaining:

```text
[ ] YOLO26 detection
[ ] Tracking
[ ] Detection + classification integration
[ ] Video pipeline
[ ] FastF1 telemetry
[ ] Telemetry overlay
[ ] FastAPI
[ ] Streamlit
[ ] Docker
[ ] Deployment
[ ] Final README
[ ] Demo video/GIF
```

---

# 42. Final Deliverable

The final result should not look like a collection of disconnected notebooks.

It should be presented as one coherent engineering system:

```text
F1 Vision Pipeline
│
├── Classical CV baseline
│   ├── Sobel
│   ├── Laplacian
│   ├── Otsu
│   ├── Morphology
│   ├── Contours
│   ├── Hu moments
│   └── HSV histogram
│
├── Deep Learning
│   └── ResNet18 transfer learning
│
├── Detection
│   └── YOLO26
│
├── Tracking
│   └── ByteTrack
│
├── Video Pipeline
│   ├── Detection
│   ├── Tracking
│   ├── Cropping
│   ├── Classification
│   └── Temporal voting
│
├── Telemetry
│   └── FastF1
│
├── Serving
│   ├── FastAPI
│   └── Streamlit
│
└── Deployment
    └── Docker
```

The key portfolio story is not simply:

> "I trained an image classifier."

It is:

> "I built and evaluated a complete computer-vision pipeline that compares classical CV against transfer learning, detects and tracks F1 cars in video, classifies their teams, fuses telemetry data, exposes inference through an API, and packages the system for deployment."

That is the intended scope of the project.
