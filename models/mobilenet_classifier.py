"""
MobileNetV2 Classification Model for Garbage Classification

This module provides a fine-tunable MobileNetV2 model for classifying
cropped garbage regions into 6 categories.
"""

import os
from pathlib import Path
from typing import Union, Optional, List, Tuple
import numpy as np

try:
    import torch
    import torch.nn as nn
    from torchvision import models, transforms
    from PIL import Image
except ImportError:
    raise ImportError("Please install torch and torchvision: pip install torch torchvision pillow")


class GarbageClassifier:
    """
    MobileNetV2-based classifier for garbage classification.
    
    This model classifies cropped garbage images into 6 categories:
    biological, cardboard, glass, metal, paper, plastic
    
    The model can be used standalone or in combination with YOLOv8 segmentor
    for a two-stage detection + classification pipeline.
    """
    
    # Class names for garbage classification
    CLASS_NAMES = ['biological', 'cardboard', 'glass', 'metal', 'paper', 'plastic']
    NUM_CLASSES = 6
    
    # Image preprocessing for MobileNetV2
    TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    def __init__(
        self,
        weights_path: Optional[Union[str, Path]] = None,
        pretrained: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize the GarbageClassifier.
        
        Args:
            weights_path: Path to fine-tuned model weights. If None, uses pretrained ImageNet weights.
            pretrained: Whether to load pretrained ImageNet weights (if weights_path is None)
            device: Device to run inference on ('cpu', 'cuda', 'mps', or None for auto)
        """
        self.weights_path = Path(weights_path) if weights_path else None
        self.pretrained = pretrained
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Build the model
        self.model = self._build_model()
        self.model.to(self.device)
        self.model.eval()
    
    def _build_model(self) -> nn.Module:
        """Build the MobileNetV2 model with custom classifier head."""
        # Load MobileNetV2 backbone
        if self.pretrained and self.weights_path is None:
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            model = models.mobilenet_v2(weights=None)
        
        # Replace the classifier head for 6 classes
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, self.NUM_CLASSES)
        )
        
        # Load fine-tuned weights if provided
        if self.weights_path and self.weights_path.exists():
            state_dict = torch.load(self.weights_path, map_location=self.device)
            model.load_state_dict(state_dict)
            print(f"Loaded weights from {self.weights_path}")
        
        return model
    
    def preprocess(self, image: Union[str, Path, np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess an image for the model.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            
        Returns:
            Preprocessed tensor ready for inference
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = Image.fromarray(image[..., ::-1] if image.dtype == np.uint8 else (image * 255).astype(np.uint8)[..., ::-1])
            else:
                image = Image.fromarray(image)
        
        return self.TRANSFORM(image).unsqueeze(0)
    
    @torch.no_grad()
    def classify(
        self,
        image: Union[str, Path, np.ndarray, Image.Image]
    ) -> Tuple[str, float, List[float]]:
        """
        Classify a single image.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            
        Returns:
            Tuple of (predicted_class, confidence, all_probabilities)
        """
        input_tensor = self.preprocess(image).to(self.device)
        
        outputs = self.model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        
        confidence, predicted_idx = torch.max(probabilities, dim=0)
        predicted_class = self.CLASS_NAMES[predicted_idx.item()]
        
        return predicted_class, confidence.item(), probabilities.cpu().numpy().tolist()
    
    @torch.no_grad()
    def classify_batch(
        self,
        images: List[Union[str, Path, np.ndarray, Image.Image]]
    ) -> List[Tuple[str, float, List[float]]]:
        """
        Classify multiple images in a batch.
        
        Args:
            images: List of input images
            
        Returns:
            List of (predicted_class, confidence, all_probabilities) tuples
        """
        if not images:
            return []
        
        # Preprocess all images
        batch_tensors = torch.cat([self.preprocess(img) for img in images], dim=0)
        batch_tensors = batch_tensors.to(self.device)
        
        outputs = self.model(batch_tensors)
        probabilities = torch.softmax(outputs, dim=1)
        
        results = []
        for i in range(len(images)):
            probs = probabilities[i]
            confidence, predicted_idx = torch.max(probs, dim=0)
            predicted_class = self.CLASS_NAMES[predicted_idx.item()]
            results.append((predicted_class, confidence.item(), probs.cpu().numpy().tolist()))
        
        return results
    
    def get_class_name(self, class_id: int) -> str:
        """Get the class name for a given class ID."""
        if 0 <= class_id < len(self.CLASS_NAMES):
            return self.CLASS_NAMES[class_id]
        return "unknown"
    
    def save_weights(self, path: Union[str, Path]):
        """Save the model weights to a file."""
        torch.save(self.model.state_dict(), path)
        print(f"Saved weights to {path}")


class GarbageClassifierTrainer:
    """
    Trainer class for fine-tuning MobileNetV2 on garbage classification.
    """
    
    def __init__(
        self,
        model: GarbageClassifier,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4
    ):
        """
        Initialize the trainer.
        
        Args:
            model: GarbageClassifier instance
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
        """
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Set up optimizer (only train classifier head initially)
        self.optimizer = torch.optim.AdamW(
            self.model.model.classifier.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=50,
            eta_min=1e-6
        )
    
    def train_epoch(self, dataloader) -> float:
        """Train for one epoch and return average loss."""
        self.model.model.train()
        total_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.model.device)
            labels = labels.to(self.model.device)
            
            self.optimizer.zero_grad()
            outputs = self.model.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        self.scheduler.step()
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader) -> Tuple[float, float]:
        """Evaluate and return (loss, accuracy)."""
        self.model.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.model.device)
                labels = labels.to(self.model.device)
                
                outputs = self.model.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        return total_loss / len(dataloader), accuracy
    
    def unfreeze_backbone(self, learning_rate: float = 1e-4):
        """Unfreeze the backbone for fine-tuning."""
        for param in self.model.model.features.parameters():
            param.requires_grad = True
        
        self.optimizer = torch.optim.AdamW(
            self.model.model.parameters(),
            lr=learning_rate,
            weight_decay=self.weight_decay
        )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"Classifying: {image_path}")
        
        classifier = GarbageClassifier(pretrained=True)
        predicted_class, confidence, probs = classifier.classify(image_path)
        
        print(f"\nPrediction: {predicted_class}")
        print(f"Confidence: {confidence:.2%}")
        print(f"\nAll probabilities:")
        for name, prob in zip(GarbageClassifier.CLASS_NAMES, probs):
            print(f"  {name}: {prob:.2%}")
    else:
        print("Usage: python mobilenet_classifier.py <image_path>")
        print("\nNote: This classifier works on cropped garbage images.")
        print("For best results, use with YOLOv8 segmentor to crop regions first.")
