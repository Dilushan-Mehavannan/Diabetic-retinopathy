"""
utils/preprocess.py
Image preprocessing pipeline for Diabetic Retinopathy classification.
Applies CLAHE, Gaussian blur, and normalization before feeding into EfficientNet-B0.
"""

import cv2
import numpy as np


IMG_SIZE = 224  # EfficientNet-B0 default input size


def apply_clahe(image: np.ndarray) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    on the green channel of the retinal image to enhance lesion visibility.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return enhanced


def remove_black_border(image: np.ndarray, threshold: int = 7) -> np.ndarray:
    """
    Crop or mask the black circular border typical in fundus images.
    Returns the image with border pixels set to zero.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    result = cv2.bitwise_and(image, image, mask=mask)
    return result


def subtract_local_mean(image: np.ndarray, sigma: int = 30) -> np.ndarray:
    """
    Subtract local average colour to reduce vignetting / uneven illumination.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    result = cv2.addWeighted(image, 4, blurred, -4, 128)
    return np.clip(result, 0, 255).astype(np.uint8)


def preprocess_image(
    image: np.ndarray,
    img_size: int = IMG_SIZE,
    apply_border_removal: bool = True,
    apply_local_mean: bool = True,
    apply_clahe_flag: bool = True,
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single retinal fundus image.

    Steps:
        1. Resize to (img_size x img_size)
        2. (Optional) Remove black border
        3. (Optional) Subtract local mean
        4. (Optional) CLAHE enhancement
        5. Normalize to [0, 1]
        6. Add batch dimension → (1, img_size, img_size, 3)

    Args:
        image:                 np.ndarray RGB image, any size.
        img_size:              Target spatial dimension (default 224).
        apply_border_removal:  Whether to mask the circular border.
        apply_local_mean:      Whether to apply local mean subtraction.
        apply_clahe_flag:      Whether to apply CLAHE.

    Returns:
        np.ndarray of shape (1, img_size, img_size, 3), dtype float32.
    """
    # 1. Resize
    img = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)

    # 2. Border removal
    if apply_border_removal:
        img = remove_black_border(img)

    # 3. Local mean subtraction
    if apply_local_mean:
        img = subtract_local_mean(img)

    # 4. CLAHE
    if apply_clahe_flag:
        img = apply_clahe(img)

    # 5. Normalize
    img = img.astype(np.float32) / 255.0

    # 6. Batch dim
    img = np.expand_dims(img, axis=0)  # (1, H, W, 3)

    return img


def preprocess_batch(images: list, img_size: int = IMG_SIZE) -> np.ndarray:
    """
    Preprocess a list of images into a batch array.

    Args:
        images:   List of np.ndarray RGB images.
        img_size: Target spatial dimension.

    Returns:
        np.ndarray of shape (N, img_size, img_size, 3), dtype float32.
    """
    batch = [preprocess_image(img, img_size)[0] for img in images]
    return np.array(batch, dtype=np.float32)
