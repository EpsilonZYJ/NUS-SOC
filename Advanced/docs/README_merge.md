# YOLOv7 Dataset Merge Tool

This tool is used to merge multiple YOLOv7 format datasets, supporting automatic handling of duplicate classes and adding new classes.

## Features

- Automatically detect and merge identical classes
- Assign unique IDs for new classes
- Maintain YOLO format label files
- Generate merged configuration files
- Provide detailed merge reports

## Usage

### Method 1: Using Command Line Tool

```bash
python merge_datasets.py --datasets dataset1 dataset2 dataset3 --output merged_output
```

### Method 2: Using Simplified Script

```python
from simple_merge import merge_yolo_datasets

datasets = ["stand_fall_dataset", "violence3"]
output_dir = "merged_dataset"

merge_yolo_datasets(datasets, output_dir)
```

### Method 3: Using Complete Class

```python
from merge_datasets import YOLODatasetMerger

merger = YOLODatasetMerger("merged_output")
merger.merge_datasets(["dataset1", "dataset2"])
```

## Dataset Format Requirements

Each dataset must contain the following structure: