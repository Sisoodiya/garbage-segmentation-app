"""
Gradio Web Interface for Garbage Detection

This module provides a user-friendly web interface for garbage
detection and classification using Gradio.
"""

import os
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

try:
    import gradio as gr
except ImportError:
    raise ImportError("Please install gradio: pip install gradio")

try:
    import cv2
    from PIL import Image
except ImportError:
    raise ImportError("Please install opencv-python and pillow")

from .yolo_segmentation import GarbageSegmentor
from .utils.visualization import create_pie_chart, CLASS_NAMES, CLASS_COLORS


# Global model instance (lazy loading)
_segmentor: Optional[GarbageSegmentor] = None


def get_segmentor() -> GarbageSegmentor:
    """Get or initialize the segmentor model."""
    global _segmentor
    if _segmentor is None:
        print("Loading YOLOv8 model...")
        _segmentor = GarbageSegmentor()
        print("Model loaded successfully!")
    return _segmentor


def process_image(
    image: np.ndarray,
    confidence_threshold: float = 0.25
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Process an image and return detection results.
    
    Args:
        image: Input image (RGB numpy array from Gradio)
        confidence_threshold: Detection confidence threshold
        
    Returns:
        Tuple of (annotated_image, pie_chart, summary_text)
    """
    if image is None:
        return None, None, "Please upload an image."
    
    # Get or initialize model
    segmentor = get_segmentor()
    segmentor.confidence_threshold = confidence_threshold
    
    # Run detection
    masks, boxes, class_ids, confidences = segmentor.segment(image)
    
    # Get annotated image
    results = segmentor.predict(image)
    annotated = segmentor.visualize(results)
    
    if annotated is not None:
        # Convert BGR to RGB for Gradio
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    else:
        annotated = image.copy()
    
    # Count classes
    class_counts = {name: 0 for name in CLASS_NAMES}
    detection_details = []
    
    for i, (cls_id, conf) in enumerate(zip(class_ids, confidences)):
        class_name = segmentor.get_class_name(cls_id)
        class_counts[class_name] += 1
        detection_details.append(f"• {class_name}: {conf:.1%}")
    
    # Create pie chart
    pie_chart = create_pie_chart(class_counts, title="Garbage Distribution")
    
    # Build summary text
    total = len(class_ids)
    summary_lines = [
        f"## Detection Summary",
        f"**Total Objects Detected:** {total}",
        "",
        "### Class Breakdown:"
    ]
    
    for cls, count in class_counts.items():
        if count > 0:
            percentage = (count / total * 100) if total > 0 else 0
            summary_lines.append(f"- **{cls.capitalize()}**: {count} ({percentage:.1f}%)")
    
    if detection_details:
        summary_lines.extend(["", "### Detection Confidence:"])
        summary_lines.extend(detection_details[:10])  # Limit to first 10
        if len(detection_details) > 10:
            summary_lines.append(f"... and {len(detection_details) - 10} more")
    
    summary_text = "\n".join(summary_lines)
    
    return annotated, pie_chart, summary_text


def create_interface() -> gr.Blocks:
    """Create the Gradio interface."""
    
    with gr.Blocks(title="Garbage Detection | AI Waste Classifier") as interface:
        # Header with inline styles
        gr.HTML("""
            <style>
                .main-title { text-align: center; color: #1a73e8; margin-bottom: 0.5rem; }
                .subtitle { text-align: center; color: #666; font-size: 1rem; margin-bottom: 1.5rem; }
            </style>
            <h1 class="main-title">🗑️ AI Garbage Segmentation System</h1>
            <p class="subtitle">Intelligent Waste Classification for Environmental Sustainability</p>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                input_image = gr.Image(
                    label="📷 Upload Image",
                    type="numpy",
                    height=400
                )
                
                confidence_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.95,
                    value=0.25,
                    step=0.05,
                    label="🎯 Confidence Threshold",
                    info="Higher values = fewer but more confident detections"
                )
                
                detect_btn = gr.Button(
                    "🔍 Detect Garbage",
                    variant="primary",
                    size="lg"
                )
                
                # Examples
                gr.Examples(
                    examples=[],  # Add example image paths here
                    inputs=input_image,
                    label="📌 Try these examples"
                )
            
            with gr.Column(scale=1):
                # Output section
                output_image = gr.Image(
                    label="🎨 Detection Results",
                    type="numpy",
                    height=400
                )
                
                with gr.Row():
                    pie_chart = gr.Image(
                        label="📊 Class Distribution",
                        type="numpy",
                        height=250
                    )
                
                summary_output = gr.Markdown(
                    label="📋 Summary"
                )
        
        # Class legend
        with gr.Accordion("📚 Garbage Classes", open=False):
            gr.HTML("""
                <div style="display: flex; flex-wrap: wrap; gap: 1rem; padding: 1rem;">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <div style="width: 20px; height: 20px; background-color: rgb(76, 175, 80); border-radius: 4px;"></div>
                        <span><strong>Biological</strong> - Food waste, organic matter</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <div style="width: 20px; height: 20px; background-color: rgb(121, 85, 72); border-radius: 4px;"></div>
                        <span><strong>Cardboard</strong> - Boxes, packaging</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <div style="width: 20px; height: 20px; background-color: rgb(33, 150, 243); border-radius: 4px;"></div>
                        <span><strong>Glass</strong> - Bottles, jars</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <div style="width: 20px; height: 20px; background-color: rgb(158, 158, 158); border-radius: 4px;"></div>
                        <span><strong>Metal</strong> - Cans, foils</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <div style="width: 20px; height: 20px; background-color: rgb(255, 193, 7); border-radius: 4px;"></div>
                        <span><strong>Paper</strong> - Documents, newspapers</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <div style="width: 20px; height: 20px; background-color: rgb(244, 67, 54); border-radius: 4px;"></div>
                        <span><strong>Plastic</strong> - Bottles, bags, containers</span>
                    </div>
                </div>
            """)
        
        # Footer
        gr.HTML("""
            <div style="text-align: center; margin-top: 2rem; padding: 1rem; color: #666; font-size: 0.9rem;">
                <p>🌱 Powered by YOLOv8 Segmentation | Built for Smart Waste Management</p>
                <p>Supporting UN SDG 11: Sustainable Cities and Communities</p>
            </div>
        """)
        
        # Event handlers
        detect_btn.click(
            fn=process_image,
            inputs=[input_image, confidence_slider],
            outputs=[output_image, pie_chart, summary_output]
        )
        
        # Auto-detect on image upload
        input_image.change(
            fn=process_image,
            inputs=[input_image, confidence_slider],
            outputs=[output_image, pie_chart, summary_output]
        )
    
    return interface


def launch_app(
    share: bool = False,
    server_port: int = 7860,
    server_name: str = "127.0.0.1"
):
    """
    Launch the Gradio web application.
    
    Args:
        share: Create a public shareable link
        server_port: Port to run the server on
        server_name: Server hostname
    """
    interface = create_interface()
    interface.launch(
        share=share,
        server_port=server_port,
        server_name=server_name,
        show_error=True
    )


# Module entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Garbage Detection Web Interface")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--host", default="127.0.0.1", help="Host address")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("🗑️  Garbage Segmentation System")
    print("=" * 50)
    print(f"Starting server at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    print("=" * 50)
    
    launch_app(
        share=args.share,
        server_port=args.port,
        server_name=args.host
    )
