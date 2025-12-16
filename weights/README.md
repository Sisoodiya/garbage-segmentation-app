# Model Weights

This directory should contain the trained model weights.

## Download Instructions

The model weights are too large (~90MB) to be included in the Git repository.

### Option 1: Download from Release
Download `best.pt` from the GitHub Releases page and place it in this directory.

### Option 2: Train Your Own
See the notebooks directory for training scripts.

## Expected Files

| File | Description | Size |
|------|-------------|------|
| `best.pt` | YOLOv8 Segmentation Model | ~90MB |

## File Structure After Download

```
weights/
├── README.md
└── best.pt
```

## Alternative: Use Path Configuration

You can also specify a custom weights path when initializing the model:

```python
from models import GarbageSegmentor

segmentor = GarbageSegmentor(weights_path="/path/to/your/best.pt")
```
