# 🪖 Real-Time Helmet Detection — YOLOv5 / YOLOv8

> **A real-time object detection system** that identifies whether individuals on construction sites, roads, or industrial environments are wearing safety helmets — built by fine-tuning YOLOv5 and YOLOv8 on a custom annotated dataset, with live webcam inference.

---

## 📌 Project Overview

Helmet compliance is a critical safety requirement on construction sites, factory floors, and roads. Manual monitoring is impractical at scale. This project builds an **automated, real-time helmet detection system** using state-of-the-art YOLO (You Only Look Once) object detection models that can:

- Detect **`Helmet`** and **`No Helmet`** across multiple people in a single frame
- Run on **live webcam feeds** with real-time bounding box overlays
- Be fine-tuned on a **custom annotated helmet dataset**
- Scale to CCTV and video surveillance pipelines

---

## 🎯 Problem Statement

> *Given a live video frame or image, detect and localise all persons and determine whether each is wearing a safety helmet — in real time.*

**Real-world applications:**
- Construction site safety compliance monitoring
- Factory floor PPE (Personal Protective Equipment) enforcement
- Road traffic helmet violation detection
- Mining and industrial safety auditing

---

## 🏗️ System Architecture

```
Input (Image / Webcam Frame)
         │
         ▼
┌──────────────────────────────┐
│   Custom Dataset Preparation │
│   HelmetDataset/             │
│   YOLO format annotations    │
│   tosplit.py (train/val/test)│
└────────────┬─────────────────┘
             │
             ▼
┌──────────────────────────────┐
│   Model Fine-Tuning          │
│   YOLOv5s  (yolov5su.pt)    │
│   YOLOv8s  (yolov8s.pt)     │
│   data.yaml config           │
│   Train_images.py            │
└────────────┬─────────────────┘
             │  Trained weights: mask_yolov5.pt
             ▼
┌──────────────────────────────┐
│   Inference                  │
│   Train_wecam.py             │
│   Live webcam detection      │
│   Bounding boxes + labels    │
└──────────────────────────────┘
             │
             ▼
Output: Real-time annotated frames — Helmet ✅ / No Helmet ❌
```

---

## 🗂️ Project Structure

```
Helmet-Detection/
│
├── HelmetDataset/             # Custom dataset in YOLO annotation format
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
│
├── runs/detect/train/         # YOLOv5/v8 training outputs (weights, metrics, plots)
│
├── data.yaml                  # YOLO dataset config — class names, paths
├── tosplit.py                 # Script to split dataset into train/val/test
├── Train_images.py            # Fine-tune YOLO on custom helmet dataset
├── Train_wecam.py             # Real-time webcam helmet detection
│
├── mask_yolov5.pt             # Fine-tuned YOLOv5 model weights
├── yolov5su.pt                # YOLOv5 pretrained base weights
├── yolov8s.pt                 # YOLOv8 pretrained base weights
└── helmet-demo.mp4            # Demo video output
```

---

## 🔬 Technical Deep Dive

### Dataset Preparation

- Used a custom **HelmetDataset** with images annotated in **YOLO format** (`.txt` files with normalised bounding box coordinates per class).
- **Two object classes:**
  - `0` — `Helmet` (person wearing helmet)
  - `1` — `No Helmet` (person without helmet)
- `tosplit.py` — automated script to split the raw annotated dataset into `train / val / test` directories at the correct ratios.
- `data.yaml` — the YOLO dataset config file specifying class names, number of classes, and paths to train/val/test image directories.

### Model Selection — YOLOv5 vs YOLOv8

Two YOLO generations were used and compared:

| Model | Architecture | Speed | Accuracy | Notes |
|---|---|---|---|---|
| **YOLOv5s** | CSPNet backbone | Very fast | Good | Excellent edge/real-time baseline |
| **YOLOv8s** | Updated C2f backbone | Fast | Better mAP | Latest architecture, improved small object detection |

Both start from **pretrained ImageNet/COCO weights** — transfer learning allows the model to detect helmets accurately even with a relatively small custom dataset.

### Fine-Tuning (`Train_images.py`)

- Loaded pretrained YOLOv5s / YOLOv8s weights as the starting point.
- Fine-tuned on the custom `HelmetDataset` using `data.yaml` configuration.
- Training outputs (weights, loss curves, precision-recall curves, confusion matrix) are saved to `runs/detect/train/`.
- Best model checkpoint saved as `mask_yolov5.pt`.

### Real-Time Webcam Inference (`Train_wecam.py`)

- Captures live frames from webcam using OpenCV.
- Passes each frame through the fine-tuned YOLO model.
- Renders bounding boxes, class labels (`Helmet` / `No Helmet`), and confidence scores on each detected object in real time.

---

## 📊 Model Performance

Training metrics stored in `runs/detect/train/` include:

| Metric | Description |
|---|---|
| **mAP@0.5** | Mean Average Precision at IoU threshold 0.5 — primary detection metric |
| **mAP@0.5:0.95** | mAP averaged across IoU thresholds 0.5–0.95 — stricter evaluation |
| **Precision** | Of all predicted helmets, how many were correct |
| **Recall** | Of all actual helmets, how many were detected |
| **Confusion Matrix** | Per-class breakdown of correct and incorrect predictions |

> *YOLOv8s achieved higher mAP compared to YOLOv5s on the same dataset — particularly on partially occluded helmets (harder cases).*

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Object Detection | YOLOv5 (Ultralytics), YOLOv8 (Ultralytics) |
| Deep Learning | PyTorch |
| Real-Time Inference | OpenCV |
| Dataset Format | YOLO annotation format (`.txt` bounding boxes) |
| Config | YAML (`data.yaml`) |

---

## 🚀 How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Charu305/Helmet-Detection.git
cd Helmet-Detection

# 2. Install dependencies
pip install ultralytics opencv-python pyyaml

# 3. Prepare dataset split (if not already split)
python tosplit.py

# 4. Fine-tune on custom helmet dataset
python Train_images.py
# Trained weights will be saved to runs/detect/train/weights/best.pt

# 5. Run real-time webcam detection
python Train_wecam.py
```

---

## 📁 Dataset Format (`data.yaml`)

```yaml
train: HelmetDataset/images/train
val:   HelmetDataset/images/val
test:  HelmetDataset/images/test

nc: 2
names: ['Helmet', 'No Helmet']
```

Each image has a corresponding `.txt` annotation file:
```
# class  cx     cy     width  height  (all normalised 0–1)
0        0.512  0.334  0.198  0.312
1        0.731  0.289  0.154  0.278
```

---

## 💡 Key Learnings & Takeaways

- **YOLO is purpose-built for real-time detection** — its single-pass architecture processes the entire image in one forward pass, making it far faster than two-stage detectors (like Faster R-CNN) while remaining highly accurate.
- **Data quality beats data quantity** — correctly formatted YOLO annotations and a clean train/val split had more impact on final mAP than simply adding more images.
- **YOLOv8 improves on YOLOv5** — the updated C2f backbone and anchor-free head design improved detection of smaller and partially occluded helmets.
- **Transfer learning on custom domains works well** — starting from COCO-pretrained weights and fine-tuning on a domain-specific dataset (helmets) converges fast and requires significantly less data than training from scratch.
- **`runs/detect/train/` is your audit trail** — YOLO automatically saves loss curves, P-R curves, confusion matrices, and sample predictions — good practice for model evaluation and reporting.

---

## 👩‍💻 Author

**Charu** — Deep Learning & Computer Vision
🔗 [GitHub Profile](https://github.com/Charu305)

---

## 📄 License

This project is developed for educational and research purposes.
