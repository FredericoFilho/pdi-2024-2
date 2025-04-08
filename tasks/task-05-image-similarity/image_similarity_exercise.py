# image_similarity_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `compare_images(i1, i2)` that receives two grayscale images
represented as NumPy arrays (2D arrays of shape (H, W)) and returns a dictionary with the following metrics:

1. Mean Squared Error (MSE)
2. Peak Signal-to-Noise Ratio (PSNR)
3. Structural Similarity Index (SSIM) - simplified version without using external libraries
4. Normalized Pearson Correlation Coefficient (NPCC)

You must implement these functions yourself using only NumPy (no OpenCV, skimage, etc).

Each function should be implemented as a helper function and called inside `compare_images(i1, i2)`.

Function signature:
    def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

The return value should be like:
{
    "mse": float,
    "psnr": float,
    "ssim": float,
    "npcc": float
}

Assume that i1 and i2 are normalized grayscale images (values between 0 and 1).
"""

import numpy as np

def compare_images(i1: np.ndarray, i2: np.ndarray) -> dict:

    if i1.shape != i2.shape:
        raise ValueError("Input images must have the same dimensions")
    
    mse_value = mse(i1, i2)
    psnr_value = psnr(i1, i2, max_pixel=1.0)
    ssim_value = ssim(i1, i2)
    npcc_value = npcc(i1, i2)

    return {
        "mse": mse_value,
        "psnr": psnr_value,
        "ssim": ssim_value,
        "npcc": npcc_value
    }


def mse(i1: np.ndarray, i2: np.ndarray) -> float:

    return np.mean((i1 - i2) ** 2)


def psnr(i1: np.ndarray, i2: np.ndarray, max_pixel: float = 1.0) -> float:

    mse_value = mse(i1, i2)

    if mse_value == 0:
        return float('inf')
    
    return 10 * np.log10((max_pixel ** 2) / mse_value)


def ssim(i1: np.ndarray, i2: np.ndarray, k1: float = 0.01, k2: float = 0.03, L: float = 1.0) -> float:

    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2

    mu_x = np.mean(i1)
    mu_y = np.mean(i2)

    sigma_x = np.var(i1)
    sigma_y = np.var(i2)
    sigma_xy = np.cov(i1.flatten(), i2.flatten())[0, 1]

    luminance = (2 * mu_x * mu_y + c1) / (mu_x**2 + mu_y**2 + c1)
    contrast = (2 * sigma_x * sigma_y + c2) / (sigma_x**2 + sigma_y**2 + c2)
    structure = (sigma_xy + c2/2) / (np.sqrt(sigma_x * sigma_y) + c2/2)

    return luminance * contrast * structure


def npcc(i1: np.ndarray, i2: np.ndarray) -> float:

    i1_flat = i1.flatten()
    i2_flat = i2.flatten()

    mean1 = np.mean(i1_flat)
    mean2 = np.mean(i2_flat)

    numerator = np.sum((i1_flat - mean1) * (i2_flat - mean2))
    denominator1 = np.sqrt(np.sum((i1_flat - mean1) ** 2))
    denominator2 = np.sqrt(np.sum((i2_flat - mean2) ** 2))

    if denominator1 == 0 or denominator2 == 0:
        return 0.0
    
    return numerator / (denominator1 * denominator2)