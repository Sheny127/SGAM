# SGAM-DETR

Official PyTorch implementation accompanying the paper:

**SGAM-DETR for Unstructured Environmental Waste Detection**

SGAM-DETR is built upon **RT-DETR** and introduces a **Structure-Aware Graph Attention Module (SGAM)** to enhance feature representation for waste object detection in complex unstructured environments. The method leverages superpixel-guided structural priors extracted by a frozen SSN model and injects them into the detection backbone to improve robustness, especially for small objects, cluttered backgrounds, and ambiguous boundaries.

---

## Overview

This repository contains the implementation of the proposed **SGAM-DETR** framework described in the paper.  
The core idea is to combine:

- **RT-DETR** as the base detector
- **Frozen SSN** for superpixel-based structural feature extraction
- **Superpixel Graph Attention Module (SGAM)** for structure-aware feature enhancement

As illustrated in the paper, SGAM is inserted between the semantic output of the backbone and the efficient hybrid encoder, enabling the detector to better focus on object regions while suppressing background noise.

---

## Repository Structure

```text
.
├── lib/                     # auxiliary library files
├── LICENSE
├── README.md
├── best_model.pth           # pretrained SSN weights
├── generate_ssn_maps.py     # optional script for SSN-related processing
├── get_coco_metrics.py      # evaluation script for mAP and scale-wise metrics
├── modules.py               # SGAM / GAT-related modules
├── ssn.py                   # SSN-related implementation
├── ssn_handler.py           # wrapper for frozen SSN loading and feature extraction
├── ssn_model.py             # SSN model definition
└── train_sgam.py            # main training script
```

---

## Environment

This project is implemented in **Python** and **PyTorch**, and relies on the **Ultralytics** framework.

### Recommended setup

- Python 3.10+  
- PyTorch 2.x  
- CUDA-enabled GPU recommended  
- Ultralytics installed in a compatible environment  

Install the main dependency:

```bash
pip install ultralytics
```

You may also need common packages such as:

```bash
pip install torch torchvision numpy opencv-python matplotlib pyyaml
```

If additional dependencies are required by your local environment, please install them accordingly.

---

## Dataset Preparation

Before running training, please prepare your dataset **in YOLO detection format**, following the same organization style commonly used in the **Ultralytics** library.

### Expected dataset format

```text
dataset_root/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── dataset.yaml
```

### Label format

Each label file should be in standard **YOLO format**:

```text
class_id x_center y_center width height
```

where all coordinates are normalized to `[0, 1]`.

### YAML configuration

Create a dataset YAML file similar to the following:

```yaml
path: /path/to/dataset_root
train: images/train
val: images/val
test: images/test

names:
  0: class_0
  1: class_1
  2: class_2
```

Please make sure the path in `train_sgam.py` points to your dataset YAML file.

In the current script, the training call uses:

```python
data=r'D:\Lab\UAVVasteDataset\FloW_IMG\yolo_dataset\dataset.yaml'
```

You should modify this path to match your own local dataset configuration.

---

## SSN Pretrained Model

This repository uses a **frozen SSN superpixel segmentation network** to provide structural guidance.

The file:

```text
best_model.pth
```

is the pretrained SSN checkpoint used by the training script.

### How was `best_model.pth` obtained?

The SSN model was trained on the **BSD500** dataset.

For detailed instructions on training the SSN superpixel network, please refer to the original project:

**https://github.com/vvarga90/ssn-pytorch-optflow**

That repository provides the full procedure for SSN training and data preparation.  
If you want to retrain the SSN module yourself, please follow its documentation and replace `best_model.pth` accordingly.

---

## Training

To launch SGAM-DETR training, run:

```bash
python train_sgam.py
```

### Notes before training

1. Make sure your dataset is already organized in **YOLO format**.
2. Make sure the dataset YAML file is correctly filled in.
3. Make sure `best_model.pth` is present in the repository root (or update the path in the script).
4. Make sure the required Ultralytics environment is installed.
5. If needed, modify hyperparameters such as:
   - `epochs`
   - `batch`
   - `imgsz`
   - `device`
   - dataset path

### Important implementation note

The training script is built on top of **Ultralytics RT-DETR** and dynamically injects the SGAM components by registering hooks into the backbone. Specifically:

- `FrozenSSN` extracts structural features from the input image
- `SuperpixelGAT` enhances the final backbone feature using superpixel-aware graph attention
- the custom GAT module is added to the trainable model so its parameters are optimized during training

---

## Evaluation

To evaluate detection performance and compute COCO-style metrics, including object-size-specific performance (e.g., small / medium / large targets), run:

```bash
python get_coco_metrics.py
```

This script is used to assess **mAP** and related metrics across different object scales.

Please ensure that:
- prediction results are generated in the required format
- the corresponding annotation files are correctly specified
- dataset paths inside the script are updated to your environment

---

## Method Description

SGAM-DETR extends RT-DETR with a **Structure-Aware Graph Attention Module**.  
The main pipeline is:

1. Input image is forwarded into the RT-DETR backbone
2. In parallel, a frozen SSN extracts superpixel-based structural features
3. SGAM performs graph-based structural enhancement using the SSN features
4. The enhanced semantic representation is passed into the efficient hybrid encoder and the decoder head
5. Detection results are produced with improved attention to meaningful waste structures

This design is especially effective for:
- small litter objects
- heavily cluttered backgrounds
- irregular object boundaries
- complex UAV and outdoor waste detection scenarios

---
