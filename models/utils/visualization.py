"""
Visualization Utilities

This module provides utility functions for visualizing
detection and segmentation results.
"""

from typing import List, Tuple, Dict, Optional, Union
import numpy as np

try:
    import cv2
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    raise ImportError(
        "Please install required packages: pip install opencv-python pillow matplotlib"
    )


# Color palette for 6 garbage classes (RGB format)
CLASS_COLORS = {
    'biological': (76, 175, 80),    # Green
    'cardboard': (121, 85, 72),     # Brown
    'glass': (33, 150, 243),        # Blue
    'metal': (158, 158, 158),       # Gray
    'paper': (255, 193, 7),         # Amber
    'plastic': (244, 67, 54),       # Red
}

# Default class names
CLASS_NAMES = ['biological', 'cardboard', 'glass', 'metal', 'paper', 'plastic']


def get_class_color(class_name: str, format: str = 'rgb') -> Tuple[int, int, int]:
    """
    Get color for a class name.
    
    Args:
        class_name: Name of the class
        format: Color format ('rgb' or 'bgr')
        
    Returns:
        Color tuple
    """
    color = CLASS_COLORS.get(class_name, (128, 128, 128))
    if format.lower() == 'bgr':
        return (color[2], color[1], color[0])
    return color


def draw_masks(
    image: np.ndarray,
    masks: List[np.ndarray],
    class_ids: List[int],
    alpha: float = 0.5,
    class_names: List[str] = None
) -> np.ndarray:
    """
    Draw segmentation masks on an image.
    
    Args:
        image: Input image (RGB or BGR)
        masks: List of binary masks
        class_ids: List of class indices
        alpha: Transparency of masks
        class_names: Optional list of class names
        
    Returns:
        Image with masks overlaid
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    result = image.copy()
    
    for mask, class_id in zip(masks, class_ids):
        class_name = class_names[class_id] if class_id < len(class_names) else 'unknown'
        color = get_class_color(class_name)
        
        # Resize mask if needed
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask.astype(np.float32), (image.shape[1], image.shape[0]))
            mask = mask > 0.5
        
        # Create colored overlay
        overlay = result.copy()
        overlay[mask > 0] = color
        
        # Blend
        result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)
    
    return result


def draw_boxes(
    image: np.ndarray,
    boxes: List[np.ndarray],
    class_ids: List[int],
    confidences: List[float] = None,
    class_names: List[str] = None,
    line_width: int = 2,
    font_scale: float = 0.6,
    show_conf: bool = True
) -> np.ndarray:
    """
    Draw bounding boxes with labels on an image.
    
    Args:
        image: Input image (RGB or BGR)
        boxes: List of boxes as [x1, y1, x2, y2]
        class_ids: List of class indices
        confidences: Optional list of confidence scores
        class_names: Optional list of class names
        line_width: Thickness of box lines
        font_scale: Scale for text
        show_conf: Whether to show confidence scores
        
    Returns:
        Image with boxes drawn
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    if confidences is None:
        confidences = [None] * len(boxes)
    
    result = image.copy()
    
    for box, class_id, conf in zip(boxes, class_ids, confidences):
        class_name = class_names[class_id] if class_id < len(class_names) else 'unknown'
        color = get_class_color(class_name, format='bgr')  # OpenCV uses BGR
        
        # Draw box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, line_width)
        
        # Prepare label
        if show_conf and conf is not None:
            label = f"{class_name}: {conf:.1%}"
        else:
            label = class_name
        
        # Calculate label size
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        
        # Draw label background
        cv2.rectangle(
            result,
            (x1, y1 - label_h - baseline - 5),
            (x1 + label_w + 5, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            result,
            label,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    
    return result


def create_pie_chart(
    class_counts: Dict[str, int],
    title: str = "Garbage Distribution",
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None
) -> np.ndarray:
    """
    Create a pie chart showing class distribution.
    
    Args:
        class_counts: Dictionary mapping class names to counts
        title: Chart title
        figsize: Figure size in inches
        save_path: Optional path to save the chart
        
    Returns:
        Chart as numpy array (RGB)
    """
    # Filter out zero counts
    labels = []
    sizes = []
    colors = []
    
    for class_name in CLASS_NAMES:
        count = class_counts.get(class_name, 0)
        if count > 0:
            labels.append(f"{class_name}\n({count})")
            sizes.append(count)
            # Normalize color to 0-1 range for matplotlib
            color = tuple(c / 255 for c in get_class_color(class_name))
            colors.append(color)
    
    if not sizes:
        # Return empty chart if no detections
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No detections", ha='center', va='center', fontsize=16)
        ax.axis('off')
    else:
        fig, ax = plt.subplots(figsize=figsize)
        wedges, texts, autotexts = ax.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.75,
            labeldistance=1.1
        )
        
        # Style the text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Convert figure to numpy array
    fig.canvas.draw()
    # Use buffer_rgba() instead of deprecated tostring_rgb()
    buf = np.asarray(fig.canvas.buffer_rgba())
    chart_image = buf[:, :, :3]  # Remove alpha channel, keep RGB
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.close(fig)
    return chart_image


def create_legend(
    class_names: List[str] = None,
    figsize: Tuple[int, int] = (4, 3)
) -> np.ndarray:
    """
    Create a color legend for the classes.
    
    Args:
        class_names: List of class names
        figsize: Figure size
        
    Returns:
        Legend image as numpy array
    """
    if class_names is None:
        class_names = CLASS_NAMES
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    
    patches = []
    for class_name in class_names:
        color = tuple(c / 255 for c in get_class_color(class_name))
        patch = mpatches.Patch(color=color, label=class_name)
        patches.append(patch)
    
    ax.legend(handles=patches, loc='center', fontsize=12)
    
    fig.canvas.draw()
    # Use buffer_rgba() instead of deprecated tostring_rgb()
    buf = np.asarray(fig.canvas.buffer_rgba())
    legend_image = buf[:, :, :3]  # Remove alpha channel, keep RGB
    
    plt.close(fig)
    return legend_image


def create_comparison_image(
    original: np.ndarray,
    annotated: np.ndarray,
    title: str = "Before / After"
) -> np.ndarray:
    """
    Create side-by-side comparison of original and annotated images.
    
    Args:
        original: Original image
        annotated: Annotated image with detections
        title: Title for the comparison
        
    Returns:
        Combined comparison image
    """
    # Ensure same size
    if original.shape != annotated.shape:
        h = max(original.shape[0], annotated.shape[0])
        w = max(original.shape[1], annotated.shape[1])
        
        original_padded = np.zeros((h, w, 3), dtype=np.uint8)
        annotated_padded = np.zeros((h, w, 3), dtype=np.uint8)
        
        original_padded[:original.shape[0], :original.shape[1]] = original
        annotated_padded[:annotated.shape[0], :annotated.shape[1]] = annotated
        
        original = original_padded
        annotated = annotated_padded
    
    # Create separator line
    separator = np.ones((original.shape[0], 5, 3), dtype=np.uint8) * 128
    
    # Concatenate horizontally
    comparison = np.concatenate([original, separator, annotated], axis=1)
    
    return comparison
