# Notebooks

This directory contains Jupyter notebooks for training, evaluation, and experimentation.

## Recommended Notebooks

For training YOLOv8 segmentation models, refer to:
- [Roboflow Notebooks](https://github.com/roboflow/notebooks)
- [Ultralytics YOLOv8 Training Guide](https://docs.ultralytics.com/modes/train/)

## Example Training Script

```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-seg.yaml')  # build a new model from scratch
# or
model = YOLO('yolov8n-seg.pt')  # load a pretrained model

# Train the model
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='garbage_segmentation'
)
```
