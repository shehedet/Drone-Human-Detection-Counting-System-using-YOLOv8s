# Vehicles and Human Detection & Counting using YOLOv8s trained on Drone Images

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

Kaggle dataset: https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset

**Dataset stats:**
- Train: 6,471 images
- Val: 548 images
- Test: 1,610 images
- Classes: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor

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

---

## Limitations

- Small object detection is challenging at 640px input resolution; higher resolution would improve recall
- The two human classes (pedestrian and people) are visually identical from above and are frequently confused with each other
- Tracking was demonstrated on sorted still images rather than a true video sequence
- Rare classes (bicycle, awning-tricycle) would benefit from oversampling or class-weighted loss
