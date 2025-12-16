"""
Unified Inference Pipeline

This module provides a combined pipeline for garbage detection
using YOLOv8 segmentation and optional MobileNetV2 classification.
"""

import os
from pathlib import Path
from typing import Union, Optional, List, Dict, Tuple
import numpy as np

try:
    import cv2
    from PIL import Image
except ImportError:
    raise ImportError("Please install opencv-python and pillow")

from .yolo_segmentation import GarbageSegmentor
from .mobilenet_classifier import GarbageClassifier
from .utils.visualization import draw_masks, draw_boxes, create_pie_chart, CLASS_NAMES


class GarbageDetectionPipeline:
    """
    Combined pipeline for garbage detection and classification.
    
    This pipeline uses:
    1. YOLOv8 for object segmentation (primary)
    2. Optional MobileNetV2 for refined classification of cropped regions
    
    By default, only YOLOv8 is used since it provides both segmentation
    and classification. MobileNetV2 can be enabled for a two-stage approach
    if needed.
    """
    
    def __init__(
        self,
        yolo_weights: Optional[str] = None,
        mobilenet_weights: Optional[str] = None,
        use_two_stage: bool = False,
        confidence_threshold: float = 0.25,
        device: Optional[str] = None
    ):
        """
        Initialize the detection pipeline.
        
        Args:
            yolo_weights: Path to YOLOv8 weights (uses default if None)
            mobilenet_weights: Path to MobileNetV2 weights (optional)
            use_two_stage: If True, use MobileNetV2 for secondary classification
            confidence_threshold: Minimum confidence for detections
            device: Device for inference ('cpu', 'cuda', 'mps', or None)
        """
        self.use_two_stage = use_two_stage
        self.confidence_threshold = confidence_threshold
        
        # Initialize YOLOv8 segmentor
        self.segmentor = GarbageSegmentor(
            weights_path=yolo_weights,
            confidence_threshold=confidence_threshold,
            device=device
        )
        
        # Initialize MobileNetV2 classifier if using two-stage approach
        self.classifier = None
        if use_two_stage:
            self.classifier = GarbageClassifier(
                weights_path=mobilenet_weights,
                device=device
            )
    
    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        return_crops: bool = False
    ) -> Dict:
        """
        Run full detection pipeline on an image.
        
        Args:
            image: Input image (path or numpy array)
            return_crops: If True, include cropped regions in results
            
        Returns:
            Dictionary containing:
                - image: Original image
                - annotated: Annotated image with detections
                - detections: List of detection info
                - summary: Detection summary statistics
                - pie_chart: Class distribution chart
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            original = cv2.imread(str(image))
            original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        else:
            original = image.copy()
        
        # Run YOLOv8 segmentation
        masks, boxes, class_ids, confidences = self.segmentor.segment(image)
        
        # Get annotated image from YOLO
        results = self.segmentor.predict(image)
        annotated = self.segmentor.visualize(results)
        if annotated is not None:
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        else:
            annotated = original.copy()
        
        # Build detections list
        detections = []
        crops = []
        
        for i, (mask, box, cls_id, conf) in enumerate(zip(masks, boxes, class_ids, confidences)):
            class_name = self.segmentor.get_class_name(cls_id)
            
            # Optionally refine classification with MobileNetV2
            if self.use_two_stage and self.classifier is not None:
                # Crop the detected region
                x1, y1, x2, y2 = map(int, box)
                crop = original[y1:y2, x1:x2]
                
                if crop.size > 0:
                    refined_class, refined_conf, probs = self.classifier.classify(crop)
                    
                    # Use refined classification if more confident
                    if refined_conf > conf:
                        class_name = refined_class
                        conf = refined_conf
                    
                    if return_crops:
                        crops.append(crop)
            
            detection = {
                "id": i,
                "class": class_name,
                "class_id": cls_id,
                "confidence": float(conf),
                "bbox": box.tolist() if hasattr(box, 'tolist') else list(box),
                "mask_area": int(mask.sum()) if mask is not None else 0
            }
            detections.append(detection)
        
        # Calculate summary
        class_counts = {name: 0 for name in CLASS_NAMES}
        for det in detections:
            if det["class"] in class_counts:
                class_counts[det["class"]] += 1
        
        summary = {
            "total_detections": len(detections),
            "class_counts": class_counts
        }
        
        # Create pie chart
        pie_chart = create_pie_chart(class_counts)
        
        result = {
            "image": original,
            "annotated": annotated,
            "detections": detections,
            "summary": summary,
            "pie_chart": pie_chart,
            "masks": masks,
            "boxes": boxes
        }
        
        if return_crops:
            result["crops"] = crops
        
        return result
    
    def detect_batch(
        self,
        images: List[Union[str, Path, np.ndarray]]
    ) -> List[Dict]:
        """
        Run detection on multiple images.
        
        Args:
            images: List of images to process
            
        Returns:
            List of detection results
        """
        return [self.detect(img) for img in images]
    
    def save_results(
        self,
        results: Dict,
        output_dir: Union[str, Path],
        prefix: str = "result"
    ) -> Dict[str, str]:
        """
        Save detection results to files.
        
        Args:
            results: Detection results from detect()
            output_dir: Directory to save results
            prefix: Filename prefix
            
        Returns:
            Dictionary with paths to saved files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        
        # Save annotated image
        annotated_path = output_dir / f"{prefix}_annotated.jpg"
        annotated_bgr = cv2.cvtColor(results["annotated"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(annotated_path), annotated_bgr)
        saved_files["annotated"] = str(annotated_path)
        
        # Save pie chart
        pie_path = output_dir / f"{prefix}_distribution.png"
        cv2.imwrite(str(pie_path), cv2.cvtColor(results["pie_chart"], cv2.COLOR_RGB2BGR))
        saved_files["pie_chart"] = str(pie_path)
        
        # Save summary as text
        summary_path = output_dir / f"{prefix}_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"Total Detections: {results['summary']['total_detections']}\n\n")
            f.write("Class Counts:\n")
            for cls, count in results['summary']['class_counts'].items():
                f.write(f"  {cls}: {count}\n")
            f.write("\nDetection Details:\n")
            for det in results['detections']:
                f.write(f"  [{det['id']}] {det['class']}: {det['confidence']:.1%}\n")
        saved_files["summary"] = str(summary_path)
        
        return saved_files


def run_inference(
    image_path: str,
    output_path: Optional[str] = None,
    confidence: float = 0.25,
    show: bool = False
) -> Dict:
    """
    Convenience function for quick inference.
    
    Args:
        image_path: Path to input image
        output_path: Optional path to save annotated result
        confidence: Confidence threshold
        show: Display results using matplotlib
        
    Returns:
        Detection results dictionary
    """
    pipeline = GarbageDetectionPipeline(confidence_threshold=confidence)
    results = pipeline.detect(image_path)
    
    if output_path:
        annotated_bgr = cv2.cvtColor(results["annotated"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, annotated_bgr)
        print(f"Saved annotated image to: {output_path}")
    
    if show:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(results["image"])
        axes[0].set_title("Original")
        axes[0].axis("off")
        
        axes[1].imshow(results["annotated"])
        axes[1].set_title("Detections")
        axes[1].axis("off")
        
        axes[2].imshow(results["pie_chart"])
        axes[2].set_title("Distribution")
        axes[2].axis("off")
        
        plt.tight_layout()
        plt.show()
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Garbage Detection Pipeline")
    parser.add_argument("--image", "-i", required=True, help="Input image path")
    parser.add_argument("--output", "-o", help="Output image path")
    parser.add_argument("--confidence", "-c", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--show", "-s", action="store_true", help="Show results")
    
    args = parser.parse_args()
    
    print(f"Processing: {args.image}")
    results = run_inference(
        args.image,
        output_path=args.output,
        confidence=args.confidence,
        show=args.show
    )
    
    print(f"\nDetection Summary:")
    print(f"  Total objects: {results['summary']['total_detections']}")
    for cls, count in results['summary']['class_counts'].items():
        if count > 0:
            print(f"  {cls}: {count}")
