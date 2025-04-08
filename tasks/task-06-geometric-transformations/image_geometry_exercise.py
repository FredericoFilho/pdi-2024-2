# image_geometry_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `apply_geometric_transformations(img)` that receives a grayscale image
represented as a NumPy array (2D array) and returns a dictionary with the following transformations:

1. Translated image (shift right and down)
2. Rotated image (90 degrees clockwise)
3. Horizontally stretched image (scale width by 1.5)
4. Horizontally mirrored image (flip along vertical axis)
5. Barrel distorted image (simple distortion using a radial function)

You must use only NumPy to implement these transformations. Do NOT use OpenCV, PIL, skimage or similar libraries.

Function signature:
    def apply_geometric_transformations(img: np.ndarray) -> dict:

The return value should be like:
{
    "translated": np.ndarray,
    "rotated": np.ndarray,
    "stretched": np.ndarray,
    "mirrored": np.ndarray,
    "distorted": np.ndarray
}
"""

import numpy as np

def apply_geometric_transformations(img: np.ndarray) -> dict:

    h, w = img.shape

    # 1. Translated image
    translated = np.zeros_like(img)
    shift_y, shift_x = 10, 10
    translated[shift_y:, shift_x:] = img[:h - shift_y, :w - shift_x]

    # 2. Rotated image
    rotated = np.rot90(img, k=-1)

    # 3. Horizontally stretched image
    new_w = int(w * 1.5)
    stretched = np.zeros((h, new_w), dtype=img.dtype)
    x_coords = (np.arange(new_w) / 1.5).astype(int)
    x_coords = np.clip(x_coords, 0, w - 1)
    stretched[:, :] = img[:, x_coords]

    # 4. Horizontally mirrored image
    mirrored = img[:, ::-1]

    # 5. Barrel distorted image
    distorted = np.zeros_like(img)
    cy, cx = h / 2, w / 2
    y_indices, x_indices = np.indices((h, w))
    x_norm = (x_indices - cx) / cx
    y_norm = (y_indices - cy) / cy
    r = np.sqrt(x_norm ** 2 + y_norm ** 2)
    k = 0.3
    x_distorted = x_norm * (1 + k * r ** 2)
    y_distorted = y_norm * (1 + k * r ** 2)
    x_src = (x_distorted * cx + cx).astype(int)
    y_src = (y_distorted * cy + cy).astype(int)
    valid = (x_src >= 0) & (x_src < w) & (y_src >= 0) & (y_src < h)
    distorted[y_indices[valid], x_indices[valid]] = img[y_src[valid], x_src[valid]]

    return {
        "translated": translated,
        "rotated": rotated,
        "stretched": stretched,
        "mirrored": mirrored,
        "distorted": distorted
    }
