"""
Image Preprocessing Utilities

This module provides utility functions for preprocessing images
before feeding them to the models.
"""

from pathlib import Path
from typing import Union, Tuple, Optional
import numpy as np

try:
    from PIL import Image
    import cv2
except ImportError:
    raise ImportError("Please install pillow and opencv-python: pip install pillow opencv-python")


def load_image(
    source: Union[str, Path, np.ndarray],
    convert_rgb: bool = True
) -> np.ndarray:
    """
    Load an image from various sources.
    
    Args:
        source: Image path, URL, or numpy array
        convert_rgb: Convert to RGB format (default True)
        
    Returns:
        Image as numpy array (RGB or BGR based on convert_rgb)
    """
    if isinstance(source, np.ndarray):
        image = source
    elif isinstance(source, (str, Path)):
        source = str(source)
        
        # Check if URL
        if source.startswith(('http://', 'https://')):
            import urllib.request
            resp = urllib.request.urlopen(source)
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(source)
            
        if image is None:
            raise ValueError(f"Could not load image from {source}")
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")
    
    if convert_rgb and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image


def resize_image(
    image: np.ndarray,
    size: Union[int, Tuple[int, int]] = 640,
    keep_aspect_ratio: bool = True,
    pad_color: Tuple[int, int, int] = (114, 114, 114)
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """
    Resize image to target size.
    
    Args:
        image: Input image as numpy array
        size: Target size (int for square, tuple for (width, height))
        keep_aspect_ratio: Whether to maintain aspect ratio with padding
        pad_color: Color for padding if keeping aspect ratio
        
    Returns:
        Tuple of (resized_image, scale_factors, padding)
    """
    if isinstance(size, int):
        target_size = (size, size)
    else:
        target_size = size
    
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    if keep_aspect_ratio:
        # Calculate scale factor
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Calculate padding
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        
        # Apply padding
        resized = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=pad_color
        )
        
        return resized, (scale, scale), (left, top)
    else:
        resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
        scale_x = target_w / w
        scale_y = target_h / h
        return resized, (scale_x, scale_y), (0, 0)


def normalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> np.ndarray:
    """
    Normalize image using ImageNet mean and std.
    
    Args:
        image: Input image (0-255 range)
        mean: Mean values for each channel
        std: Std values for each channel
        
    Returns:
        Normalized image as float32 array
    """
    image = image.astype(np.float32) / 255.0
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    
    normalized = (image - mean) / std
    return normalized


def crop_region(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    padding: float = 0.1
) -> np.ndarray:
    """
    Crop a region from an image with optional padding.
    
    Args:
        image: Input image
        bbox: Bounding box as (x1, y1, x2, y2)
        padding: Padding ratio to add around the crop
        
    Returns:
        Cropped image region
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Calculate padding
    box_w = x2 - x1
    box_h = y2 - y1
    pad_x = int(box_w * padding)
    pad_y = int(box_h * padding)
    
    # Apply padding with bounds checking
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    
    return image[y1:y2, x1:x2]


def apply_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Optional[Tuple[int, int, int]] = None,
    alpha: float = 0.5
) -> np.ndarray:
    """
    Apply a segmentation mask overlay to an image.
    
    Args:
        image: Input image (RGB)
        mask: Binary mask
        color: Optional color for the mask overlay
        alpha: Transparency for the overlay
        
    Returns:
        Image with mask overlay
    """
    if color is None:
        color = (0, 255, 0)  # Default green
    
    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color
    
    # Blend with original image
    result = image.copy()
    mask_indices = mask > 0
    result[mask_indices] = (
        (1 - alpha) * image[mask_indices] + alpha * colored_mask[mask_indices]
    ).astype(np.uint8)
    
    return result


def to_pil_image(image: np.ndarray) -> Image.Image:
    """Convert numpy array to PIL Image."""
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    return Image.fromarray(image)


def from_pil_image(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to numpy array."""
    return np.array(pil_image)


def enhance_image(
    image: np.ndarray,
    enhance_contrast: bool = True,
    sharpen: bool = True,
    denoise: bool = False
) -> np.ndarray:
    """
    Enhance image quality for better detection accuracy.
    
    Args:
        image: Input image (RGB or BGR)
        enhance_contrast: Apply CLAHE contrast enhancement
        sharpen: Apply sharpening filter
        denoise: Apply denoising (slower but cleaner)
        
    Returns:
        Enhanced image
    """
    result = image.copy()
    
    # Convert to LAB for contrast enhancement
    if enhance_contrast:
        # Convert to LAB color space
        lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge and convert back
        lab = cv2.merge([l, a, b])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Apply sharpening
    if sharpen:
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        result = cv2.filter2D(result, -1, kernel)
    
    # Apply denoising (optional, slower)
    if denoise:
        result = cv2.fastNlMeansDenoisingColored(result, None, 10, 10, 7, 21)
    
    return result


def auto_adjust_brightness(image: np.ndarray) -> np.ndarray:
    """
    Automatically adjust brightness based on image statistics.
    
    Args:
        image: Input image (RGB)
        
    Returns:
        Brightness-adjusted image
    """
    # Calculate mean brightness
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    mean_brightness = np.mean(gray)
    
    # Target brightness is around 127
    target = 127
    
    if mean_brightness < 100:  # Too dark
        factor = target / (mean_brightness + 1)
        factor = min(factor, 2.0)  # Cap at 2x
        result = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    elif mean_brightness > 180:  # Too bright
        factor = target / mean_brightness
        result = cv2.convertScaleAbs(image, alpha=factor, beta=0)
    else:
        result = image
    
    return result

