# Dataset

This folder contains the training and validation dataset for the garbage segmentation model.

## Source

**Dataset**: garbage-segmentation v2  
**Source**: [Roboflow Universe](https://universe.roboflow.com/projetannuel/garbage-segmentation-lt1hb)  
**License**: CC BY 4.0

## Directory Structure

```
data/
├── README.md
├── train/
│   ├── images/      # Training images (640×640)
│   └── labels/      # YOLOv8 annotation files
└── valid/
    ├── images/      # Validation images (640×640)
    └── labels/      # YOLOv8 annotation files
```

## Statistics

| Property | Value |
|----------|-------|
| Total Images | 481 |
| Training Set | ~385 images (~80%) |
| Validation Set | ~96 images (~20%) |
| Image Size | 640×640 |
| Annotation Format | YOLOv8 Segmentation |

## Classes (6 Categories)

| ID | Class | Description |
|----|-------|-------------|
| 0 | biological | Food waste, organic matter |
| 1 | cardboard | Boxes, packaging materials |
| 2 | glass | Bottles, jars, glass containers |
| 3 | metal | Cans, aluminum foils |
| 4 | paper | Documents, newspapers, magazines |
| 5 | plastic | Bottles, bags, plastic containers |

## Pre-processing Applied

- Auto-orientation of pixel data (EXIF-orientation stripping)
- Resize to 640×640 (Stretch)
- No augmentation applied

## Git LFS

> **Note**: Due to the size of image files, this folder should be tracked with Git LFS:
> ```bash
> git lfs install
> git lfs track "data/**/*.jpg"
> git lfs track "data/**/*.png"
> ```
