import numpy as np

from core.intensity import clahe_enhance, normalize_minmax
from core.filtering import bilateral_filter, gaussian_filter


def extract_green_channel(img):
    if len(img.shape) == 3 and img.shape[2] >= 3:
        return img[:, :, 1].astype(np.float32)
    return img.astype(np.float32)


def enhance_vessels(img, clip_limit=3.0):
    enhanced = clahe_enhance(img, clip_limit=clip_limit, grid_size=(8, 8))
    return enhanced


def preprocess_fundus(img):
    green = extract_green_channel(img)
    enhanced = enhance_vessels(green)
    filtered = bilateral_filter(enhanced, d=0, sigma_color=2.0, sigma_space=5.0)
    return filtered


def preprocess_oct(img):
    if len(img.shape) == 3:
        gray = np.mean(img, axis=2).astype(np.float32)
    else:
        gray = img.astype(np.float32)
    normalized = normalize_minmax(gray, 0, 255)
    filtered = bilateral_filter(normalized, d=0, sigma_color=3.0, sigma_space=5.0)
    return filtered


def preprocess_slit_lamp(img):
    if len(img.shape) == 3:
        gray = np.mean(img, axis=2).astype(np.float32)
    else:
        gray = img.astype(np.float32)
    enhanced = clahe_enhance(gray, clip_limit=2.0, grid_size=(16, 16))
    filtered = gaussian_filter(enhanced, ksize=3)
    return filtered
