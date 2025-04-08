# histogram_matching_exercise.py
# STUDENT'S EXERCISE FILE

"""
Exercise:
Implement a function `match_histograms_rgb(source_img, reference_img)` that receives two RGB images
(as NumPy arrays with shape (H, W, 3)) and returns a new image where the histogram of each RGB channel 
from the source image is matched to the corresponding histogram of the reference image.

Your task:
- Read two RGB images: source and reference (they will be provided externally).
- Match the histograms of the source image to the reference image using all RGB channels.
- Return the matched image as a NumPy array (uint8)

Function signature:
    def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray

Return:
    - matched_img: NumPy array of the result image

Notes:
- Do NOT save or display the image in this function.
- Do NOT use OpenCV to apply the histogram match (only for loading images, if needed externally).
- You can assume the input images are already loaded and in RGB format (not BGR).
"""

import cv2 as cv
import numpy as np
import skimage as ski

def match_histograms_rgb(source_img: np.ndarray, reference_img: np.ndarray) -> np.ndarray:
    matched_channels = []
    for channel in range(3):
        source_channel = source_img[:, :, channel]
        reference_channel = reference_img[:, :, channel]
        
        hist_source, bins_source = np.histogram(source_channel.flatten(), bins=256, range=(0, 256), density=True)
        hist_reference, bins_reference = np.histogram(reference_channel.flatten(), bins=256, range=(0, 256), density=True)
        
        cdf_source = hist_source.cumsum()
        cdf_reference = hist_reference.cumsum()
        
        cdf_source_normalized = cdf_source / cdf_source.max()
        cdf_reference_normalized = cdf_reference / cdf_reference.max()
        
        mapping = np.zeros(256, dtype=np.uint8)
        for s_val in range(256):
            diff = np.abs(cdf_reference_normalized - cdf_source_normalized[s_val])
            mapping[s_val] = np.argmin(diff)
        
        matched_channel = mapping[source_channel]
        matched_channels.append(matched_channel)
    
    matched_img = np.stack(matched_channels, axis=2).astype(np.uint8)
    return matched_img