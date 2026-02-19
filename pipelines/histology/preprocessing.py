import numpy as np
import cv2

from core.intensity import stain_normalize, normalize_minmax


def detect_tissue(img, threshold=220):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    tissue_mask = (gray < threshold).astype(np.uint8)
    return tissue_mask


def get_tissue_ratio(img, threshold=220):
    mask = detect_tissue(img, threshold)
    return np.sum(mask) / mask.size


def normalize_stain(img, method='macenko'):
    return stain_normalize(img, method=method)
