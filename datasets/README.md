# SafeStep Datasets: Generation, Expansion, Validation, and Preparation Guide

This directory contains the two primary datasets for **SafeStep: AI-Enabled Smart Navigation System for the Visually Impaired**, along with the modular dataset generation pipeline in `dataset_generation/`.

## 1. Directory Structure

```text
datasets/
├── object_detection/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── data.yaml
│   └── metadata.csv
│
├── currency_detection/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── data.yaml
│   └── metadata.csv
│
└── dataset_generation/
    ├── generate_object_dataset.py
    ├── generate_currency_dataset.py
    ├── collect_images.py
    ├── annotate_dataset.py
    ├── split_dataset.py
    ├── augment_dataset.py
    └── validate_dataset.py
```

## 2. Dataset Specifications & Provenance Transparency

### Dataset A: Object and Obstacle Detection (`object_detection/`)
- **12 Navigation & Hazard Classes**:
  `0: Person`, `1: Car`, `2: Motorcycle`, `3: Bicycle`, `4: Bus`, `5: Truck`, `6: Pole`, `7: Wall`, `8: Chair`, `9: Table`, `10: Door`, `11: Other obstacles`.
- **Environmental Diversity**: Indoor (corridors, rooms, offices) and outdoor (roads, footpaths, public spaces, crowded scenes) under daylight, bright, low-light, night, and warm indoor lighting, with varied camera angles and partial occlusions.
- **Provenance Tracking**: Every image is logged in `metadata.csv` with `source_type` (`programmatic_synthetic`, `webcam_hybrid_scene`, or `webcam_capture`) and `is_synthetic` (`True`/`False`). Synthetic scenes are never misrepresented as genuine camera photographs.

### Dataset B: Indian Currency Detection (`currency_detection/`)
- **6 Denomination Classes**:
  `0: 10_rupees`, `1: 20_rupees`, `2: 50_rupees`, `3: 100_rupees`, `4: 200_rupees`, `5: 500_rupees`.
- **Visual Features**: Front and back sides, RBI aspect ratios and color palettes, denomination numerals, security strips, multiple orientations, partial folds, thumb/hand occlusions, and multi-note compositions across diverse backgrounds.
- **Specimen Watermark & Provenance**: All programmatically generated notes include a visible `SPECIMEN` label and are explicitly recorded with `source_type="programmatic_synthetic_specimen"` and `is_synthetic=True` in `metadata.csv`.

## 3. Zero-Leakage Pipeline Execution Order

To generate, split, augment, and validate both datasets from scratch:

```bash
# Step 1: Generate base images and YOLO annotations
python datasets/dataset_generation/generate_object_dataset.py --count 750
python datasets/dataset_generation/generate_currency_dataset.py --count 850

# Step 2 (Optional): Collect additional genuine webcam frames or import local photographs
python datasets/dataset_generation/collect_images.py --dataset object --webcam-frames 20
python datasets/dataset_generation/annotate_dataset.py --dataset-dir datasets/object_detection

# Step 3: Split into 70% train, 20% val, 10% test grouped by session_id BEFORE augmentation
python datasets/dataset_generation/split_dataset.py --dataset both

# Step 4: Apply Albumentations bounding-box-aware augmentations within each split
python datasets/dataset_generation/augment_dataset.py --dataset both --multiplier 2

# Step 5: Validate annotations, check zero session leakage, and generate quality charts
python datasets/dataset_generation/validate_dataset.py --dataset both
```

## 4. Expanding Beyond 5,000+ Images

To scale either dataset to 5,000+ images for extended GPU training runs:
```bash
python datasets/dataset_generation/generate_object_dataset.py --count 1800
python datasets/dataset_generation/split_dataset.py --dataset object
python datasets/dataset_generation/augment_dataset.py --dataset object --multiplier 2
python datasets/dataset_generation/validate_dataset.py --dataset object
```
