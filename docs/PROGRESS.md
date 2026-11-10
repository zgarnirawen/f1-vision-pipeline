# F1 Vision Pipeline — Development Progress

> **Status:** In development  
> **Current working branch:** `feature/yolo26-detection`  
> **Last updated:** 10 November 2026

This document records work that has actually been completed and measured so far. Planned features are listed separately and are not presented as implemented.

## 1. Project overview

F1 Vision Pipeline is being developed as an end-to-end computer-vision system for Formula 1 footage. The intended system will detect F1 cars, track them across video frames, classify their teams where suitable labeled data is available, and produce annotated video. Later extensions include FastF1 telemetry, an API, a demonstration interface, and containerization.

The project currently contains two distinct machine-learning tasks:

1. **Team image classification** — classify a cropped F1 car image into one of the team classes.
2. **F1 car object detection** — locate F1 cars in an image using bounding boxes.

These tasks use different datasets and label spaces. The current detection dataset has only one class, `Formula-1-cars`; it does not identify teams or drivers.

## 2. Completed: image classification

The earlier classification phase implemented and evaluated classical computer-vision baselines and a ResNet18 transfer-learning model.

### Methods

- Hand-crafted computer-vision features with an SVM classifier.
- Hand-crafted computer-vision features with a k-nearest-neighbors (k-NN) classifier.
- ResNet18 transfer learning using PyTorch.

### Reported results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Classical SVM | 39.3% | 0.382 |
| Classical k-NN | 29.9% | 0.288 |
| ResNet18 | 86.8% | 0.871 |

The reported best validation accuracy for ResNet18 was 88.9%. Training accuracy reached approximately 98–99%, indicating a gap between training and validation performance and a risk of overfitting.

These are previously reported project results; they have not been independently rerun as part of the current detection work.

## 3. Completed: detection training and evaluation code

The branch `feature/yolo26-detection` contains the initial YOLO26 detection workflow:

| File | Purpose |
|---|---|
| `src/train_detector.py` | Train a YOLO detector and run validation. |
| `src/evaluate_detector.py` | Evaluate saved weights and export metrics. |
| `src/prepare_detection_dataset.py` | Prepare a Roboflow dataset for YOLO detection by converting polygon annotations to bounding boxes. |
| `configs/f1_detection.yaml.example` | Example dataset configuration. |
| `requirements.txt` | Python project dependencies. |

### Initial dataset

The original Roboflow export contained 116 images:

| Split | Images | Label files |
|---|---:|---:|
| Train | 73 | 73 |
| Validation | 20 | 20 |
| Test | 23 | 23 |
| **Total** | **116** | **116** |

The export used one class, `Formula-1-cars`. It contained ordinary YOLO box annotations and six polygon annotations. The preparation script converts each polygon into its enclosing axis-aligned rectangle, while preserving existing box annotations.

### Initial YOLO26n baseline

A short five-epoch baseline was trained on the original prepared dataset using:

- Model: `yolo26n.pt`
- Epochs: 5
- Image size: 416
- Batch size: 2
- Workers: 0
- Device: CPU

Reported validation metrics:

| Metric | Result |
|---|---:|
| Precision | 57.59% |
| Recall | 60.00% |
| mAP50 | 63.64% |
| mAP50-95 | 24.52% |
| Inference time reported by validation | 26.81 ms/image |

The checkpoint was saved locally at:

```text
runs/detect/runs/detect/f1-yolo26n/weights/best.pt
```

This is a preliminary baseline, not a final model. It was trained for only five epochs on a very small dataset, and its metrics are based on 20 validation images/instances. The measurements should therefore be interpreted cautiously. The local checkpoint and run artifacts are not implied to be committed to GitHub.

## 4. Dataset annotation audit

The original annotations were inspected visually and statistically. The review identified inconsistent bounding-box conventions: some boxes included the complete visible car (wings and tires), while others covered only part of the body.

A consistent annotation convention was established:

- Enclose all visible parts of the F1 car: front wing, rear wing, visible tires, body/cockpit, and halo.
- For a partially cropped car, extend the box to the image boundary; do not invent invisible parts.
- Do not label non-F1 vehicles such as safety cars, motorbikes, or ordinary road cars.

A manually selected review sample contained 13 images: 8 were judged to need relabeling and 5 were considered acceptable. This was a targeted sample, not a random sample, so the proportion must not be treated as a dataset-wide error rate. One missing F1-car annotation was also identified in the reviewed sample.

The original dataset's annotation statistics and outlier visualizations were used as screening aids. Small boxes were not automatically considered incorrect because distant cars can legitimately occupy few pixels.

## 5. Corrected dataset preparation

A new Roboflow export was obtained for the corrected annotations.

- Roboflow project: [Formula-1-car corrected dataset](https://app.roboflow.com/ivanarr/formula-1-car-izjor/1)
- Export version: 1
- License shown in export metadata: CC BY 4.0
- Classes: one (`Formula-1-cars`)

The corrected export contains:

| Split | Images | Label files | Annotation rows | Polygon rows converted |
|---|---:|---:|---:|---:|
| Train | 73 | 73 | 76 | 4 |
| Validation | 20 | 20 | 20 | 1 |
| Test | 23 | 23 | 25 | 1 |
| **Total** | **116** | **116** | **121** | **6** |

The mixed box/polygon export was prepared locally with:

```powershell
python src/prepare_detection_dataset.py --input data/corrected --output data/processed/detection_corrected
```

The prepared dataset is located locally at:

```text
data/processed/detection_corrected/
```

The preparation process produced 121 annotation rows and converted all six polygon rows into standard YOLO detection boxes. The source export was kept separate from the prepared output.

### Technical validation

The converted labels were checked using `src/validate_converted_dataset.py`.

Validation results:

- 116 images
- 116 label files
- 121 annotation rows
- 0 format or coordinate issues
- Every annotation row has five values: class ID, center x, center y, width, height.
- Class ID and normalized box coordinates were checked.

**Result: PASS.**

This confirms that the label files are structurally valid. It does not prove that every rectangle accurately encloses the visible car.

### Visual verification status

Visual inspection of the corrected training annotations is the next quality-control step. It must be completed before the corrected dataset is treated as final for model training. The visual audit is intended to verify that boxes cover all visible car parts consistently and that missing or incorrectly labeled objects have been addressed.

The local visualization script is `src/visualize_annotations.py`. The corrected dataset paths and a separate output folder are used for this audit during local work. The audit result is not yet recorded as passed in this document.

## 6. Development environment

The initial detector baseline was run locally with:

- Windows and PowerShell
- Python 3.13.5
- Ultralytics 8.4.155
- PyTorch 2.14.0+cpu
- Intel Core i7-13700H
- Intel Iris Xe integrated graphics
- Approximately 23.6 GB RAM
- No CUDA-compatible NVIDIA GPU detected

The baseline therefore ran on CPU. Training settings and timing should be considered in that hardware context.

## 7. Current status

### Completed

- [x] Classical SVM and k-NN classification baselines.
- [x] ResNet18 transfer-learning classifier and reported evaluation.
- [x] Classification metrics and error analysis documented in project work.
- [x] YOLO26 detector training and evaluation scripts added.
- [x] Initial YOLO26n five-epoch baseline trained and evaluated.
- [x] Original detection dataset prepared for YOLO detection.
- [x] Annotation review performed and a consistent box convention defined.
- [x] Corrected Roboflow dataset exported and prepared locally.
- [x] Six polygon annotations converted to bounding boxes.
- [x] Corrected dataset format validation passed with zero issues.

### In progress

- [ ] Finish visual verification of the corrected annotations.
- [ ] Verify that original and corrected dataset splits are comparable before making a strict baseline comparison.
- [ ] Train a corrected-dataset YOLO26n model under controlled settings.
- [ ] Evaluate the selected detector on the held-out test split after model selection.

### Planned

- [ ] Improve detection performance and document error cases.
- [ ] Integrate ByteTrack for multi-object tracking.
- [ ] Integrate team classification with detected/tracked car crops, subject to compatible class labels and data.
- [ ] Generate annotated video output.
- [ ] Explore FastF1 telemetry retrieval and explicitly validate any video/telemetry synchronization.
- [ ] Build a FastAPI inference service.
- [ ] Build a Streamlit demonstration interface.
- [ ] Add Docker support and deployment instructions.
- [ ] Publish demonstration media and final reproducibility documentation.

## 8. Important interpretation and limitations

- The detector's current class label identifies an F1 car, not its team or driver.
- The initial detector metrics come from a five-epoch run and a validation set of only 20 images/instances; they are preliminary and may vary substantially.
- Lowering the prediction confidence threshold can increase the number of proposed detections, but it does not fix inaccurate boxes or missing ground-truth annotations.
- Structurally valid labels can still be visually inaccurate; manual image review remains necessary.
- A fair comparison between the original and corrected detector requires checking that image membership and split assignments are equivalent, and using the same model, training settings, and evaluation procedure.
- The corrected dataset and trained model artifacts currently exist in the local working environment; this progress note does not claim that those data or weights are stored in the Git repository.

## 9. Reproduction commands

Prepare the corrected dataset:

```powershell
python src/prepare_detection_dataset.py --input data/corrected --output data/processed/detection_corrected
```

Validate the converted labels:

```powershell
python src/validate_converted_dataset.py
```

The validation script expects the prepared dataset at `data/processed/detection_corrected`.

The next planned experiment is to train YOLO26n on:

```text
data/processed/detection_corrected/data.yaml
```

using a new run name so that the original baseline artifacts remain unchanged.
