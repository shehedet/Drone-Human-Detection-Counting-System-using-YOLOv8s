# Drone Human Detection & Counting — YOLOv8s on VisDrone

A computer vision pipeline for detecting humans and cars in drone/aerial images, built on the VisDrone2019 dataset. Includes detection, human counting, and object tracking.

---

## Overview

This project fine-tunes a YOLOv8s model on the VisDrone2019 detection dataset to:
- Detect humans (pedestrians) and cars in aerial images
- Count total humans per image
- Visualize detections with bounding boxes and count overlays
- Track objects across image sequences using ByteTrack

---

## Dataset

**VisDrone2019-DET** — drone-captured imagery with 10 object classes.

Download from Kaggle: https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset

After downloading, place the dataset in the project root:

```
VisDrone_Dataset/
├── VisDrone2019-DET-train/
│   ├── images/
│   └── labels/
├── VisDrone2019-DET-val/
│   ├── images/
│   └── labels/
├── VisDrone2019-DET-test-dev/
│   ├── images/
│   └── labels/
└── visdrone.yaml
```

**Dataset stats:**
- Train: 6,471 images
- Val: 548 images
- Test: 1,610 images
- Classes: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor

---

## Setup

```bash
pip install ultralytics opencv-python matplotlib pyyaml torch
```

Requires Python 3.8+ and a CUDA-capable GPU for training. Inference runs on CPU as well.

---

## Usage

All tasks are contained in a single notebook:

```
yolov8s_training_testing.ipynb
```

Run cells top to bottom. The notebook is organized into five sections matching the task structure:

- **Task 1** — Dataset EDA and visualization
- **Task 2** — Model training
- **Task 3** — Human & car detection with counting
- **Task 4** — Object tracking with ByteTrack
- **Task 5** — Evaluation and metrics

Before running inference/evaluation cells, update the path variables in Cell 11 to match your local YOLO run directory:

```python
BEST_PT_PATH  = r"runs\detect\...\weights\best.pt"
TRAIN_PROJECT = r"runs\detect\...\training"
```

---

## Training Details

| Setting | Value |
|---|---|
| Model | YOLOv8s (fine-tuned from COCO weights) |
| Epochs | 35 |
| Image size | 640 |
| Batch size | 4 |
| Optimizer | AdamW |
| Learning rate | 0.001 |
| Augmentation | Mosaic, HSV jitter, flips, scaling |
| Device | NVIDIA GeForce RTX 4050 Laptop GPU |
| Training time | ~1.74 hours |

---

## Results

### Overall Metrics (Validation Set)

| Metric | Value |
|---|---|
| Precision | 0.480 |
| Recall | 0.384 |
| mAP@0.50 | 0.366 |
| mAP@0.50:0.95 | 0.212 |
| Inference speed | 53.6 FPS |

### Per-Class AP@0.50

| Class | AP@0.50 |
|---|---|
| car | 0.775 |
| bus | 0.521 |
| motor | 0.423 |
| van | 0.421 |
| pedestrian | 0.404 |
| truck | 0.352 |
| people | 0.291 |
| tricycle | 0.236 |
| awning-tricycle | 0.128 |
| bicycle | 0.106 |

Car detection is strongest due to its consistent aerial appearance and large representation in the training set (144k instances). Bicycle and awning-tricycle are weakest due to class imbalance and visual ambiguity from above.

---

## Outputs

All generated outputs are saved under `yolov8s_outputs/`:

```
yolov8s_outputs/
├── task1_dataset_eda/
│   ├── class_distribution.png
│   ├── sample_images_raw.png
│   └── sample_images_with_gt_boxes.png
├── task2_training/
│   ├── training_curves.png
│   └── validation_sample_predictions.png
├── task3_detection/
│   ├── detection_demo_single.png
│   ├── detection_results_grid.png
│   └── detections/
├── task4_tracking/
│   ├── tracking_grid.png
│   └── bytetrack_output/
└── task5_evaluation/
    ├── per_class_ap50.png
    ├── confusion_matrix_normalized.png
    └── evaluation_report.txt
```

---

## Limitations

- Small object detection is challenging at 640px input resolution; higher resolution would improve recall
- The two human classes (pedestrian and people) are visually identical from above and are frequently confused with each other
- Tracking was demonstrated on sorted still images rather than a true video sequence
- Rare classes (bicycle, awning-tricycle) would benefit from oversampling or class-weighted loss

---

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [VisDrone Dataset](http://aiskyeye.com/)
