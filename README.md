# SGAM-DETR: A Superpixel-Guided Graph Attention Network for Unstructured Environmental Waste Detection on Autonomous Systems

## 1. Title
**SGAM-DETR** (Superpixel-Guided Graph Attention Network with Real-Time Detection Transformer)

## 2. Description
This repository contains the official PyTorch implementation of **SGAM-DETR**. This framework introduces a Structure-Aware Graph Attention Module (SGAM) integrated into the RT-DETR architecture. By utilizing a differentiable Superpixel Sampling Network (SSN) pre-trained on the BSD500 dataset, SGAM-DETR decomposes complex, unstructured scenes into boundary-adherent superpixel regions. This non-Euclidean graph representation successfully isolates the intrinsic geometric and structural features of amorphous and micro-debris (such as plastic bags, bottles, and styrofoam) from dynamic, heterogeneous backgrounds (like water glint, sand, and riparian vegetation).

## 3. Dataset Information
The model is evaluated across a multi-platform composite benchmark consisting of four publicly available third-party datasets. Please download them from their respective repository owners:
*   **UAVVaste:** High-altitude UAV nadir perspective dataset for terrestrial litter. Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). [Zenodo Link](https://zenodo.org/records/8214061).
*   **Aerial Beach Waste Dataset:** High-altitude coastal UAV dataset focusing on granular backgrounds. Dedicated to the public domain under [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/). [Roboflow Universe Link](https://universe.roboflow.com/national-cheng-kung-university-wjot1/aerial-beach-waste-dataset-xpzsi).
*   **FloW-Img:** Low-altitude oblique USV dataset with severe aquatic reflections. Licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). [Orca-Tech Link](https://orca-tech.cn/datasets/FloW/FloW-Img).
*   **D-six (D_six):** Low-altitude oblique USV dataset with dense riparian camouflage. Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). [Zenodo Link](https://zenodo.org/records/15195086).

## 4. Code Information
The core implementation files and directories are organized as follows:
*   `train_sgam.py`: The main entry script to train the SGAM-DETR model on custom datasets.
*   `generate_ssn_maps.py`: Script to generate boundary-adherent superpixel maps using the pre-trained SSN prior.
*   `get_coco_metrics.py`: Evaluation script to compute class-agnostic MS COCO metrics ($AP_{0.5:0.95}$, $AP_S$, $AP_M$, $AP_L$) on the test splits.
*   `modules.py`: Implementation of core neural network modules, including the Superpixel-Guided Graph Attention Module.
*   `ssn.py`, `ssn_model.py`, & `ssn_handler.py`: Implementation and wrapper handlers of the Simple Superpixel Network (SSN).
*   `lib/`: Directory containing CUDA/C++ source code (such as `pair_wise_distance_cuda_source.py` and `pair_wise_distance.py`) for highly efficient pairwise distance calculations.
*   `best_model.pth`: Pre-trained weights for the SSN feature extractor.

## 5. Usage Instructions

### Data Preparation
1. Download the datasets from the links provided in Section 3.
2. Structure your dataset directory as follows:
   ```text
   /datasets
     /UAVVaste
     /BeachWaste
     /FloW_IMG
     /D_six
   ```

### Superpixel Feature Extraction
Generate the boundary-adherent superpixel spatial maps prior to training:
```bash
python generate_ssn_maps.py --data_path ./datasets/UAVVaste --output_path ./ssn_priors
```

### Training
Run the training pipeline with your configuration using:
```bash
python train_sgam.py --config configs/sgam_detr_l.yaml --epochs 100 --batch 16
```

### Evaluation
Compute standard MS COCO metrics on the test splits:
1. Open `get_coco_metrics.py` and configure your paths at the top of the file:
   ```python
   GT_FILE = 'path/to/test_ground_truth.json' 
   DT_FILE = 'path/to/predictions.json'
   TEST_IMG_DIR = 'path/to/test_images'
   ```
2. Run the evaluation script:
   ```bash
   python get_coco_metrics.py
   ```

## 6. Requirements
Ensure your workspace meets the following dependencies:
*   **Operating System:** Ubuntu 20.04 or Windows 10/11
*   **GPU:** NVIDIA GPU with CUDA support (CUDA 11.3+ recommended)
*   **Python:** >= 3.8
*   **PyTorch:** >= 1.10
*   **Libraries:** `torchvision`, `numpy`, `opencv-python`, `scikit-image`, `ultralytics`, `pycocotools`

Install all Python dependencies via:
```bash
pip install -r requirements.txt
```

## 7. Methodology
The SGAM-DETR computational pipeline involves:
1.  **Prior Extraction:** A parallel pre-trained SSN maps RGB inputs into boundary-adherent superpixels.
2.  **Euclidean to Non-Euclidean Aggregation:** Standard feature maps from the backbone are dynamically pooled into irregular graph node embeddings using a temperature-scaled affinity matrix.
3.  **Relational Reasoning:** A multi-head graph self-attention network models long-range dependencies across contiguous physical structures.
4.  **Graph Unpooling & Channel Concatenation:** Node features are unpooled back to pixel space and fused with residual backbone representations using channel-concatenated mixing convolutions before going into the hybrid encoder.

## 8. Citations
If you use this code, the pre-trained weights, or the methodology in your research, please cite our paper:
```bibtex
@article{yang2026sgam,
  title={SGAM-DETR: a superpixel-guided graph attention network for unstructured environmental waste detection on autonomous systems},
  author={Yang, [Your First Name/Last Name] and Chi, Haiyang and He, Banghua and Tang, Jun and Zhu, Shihua and Wen, Yadong},
  journal={PeerJ Computer Science},
  year={2026}
}
```

## 9. License & Contribution Guidelines
*   **License:** This repository is licensed under the Apache 2.0 License. See the `LICENSE` file for more details.
*   **Contributions:** We welcome contributions, bug reports, and pull requests to improve the efficiency and applicability of SGAM-DETR. Please open an issue first to discuss your proposed changes.
```
