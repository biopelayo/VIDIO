import numpy as np
import cv2

from core.edges import laplacian_edges, canny_edges
from core.morphology import morph_open, morph_close, morph_dilate, fill_holes, remove_small_objects
from core.contours import find_contours


def segment_vessels(img, method='laplacian'):
    if method == 'laplacian':
        edge_map, stats = laplacian_edges(img)
        threshold = stats['std'] * 0.5
        vessel_mask = (np.abs(edge_map) > threshold).astype(np.uint8)
    else:
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        vessel_mask = canny_edges(img_uint8, low=30, high=100)

    vessel_mask = morph_close(vessel_mask, kernel_size=3, iterations=2)
    vessel_mask = remove_small_objects(vessel_mask, min_area=50)
    return vessel_mask


def detect_optic_disc(img, min_radius=30, max_radius=120):
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    circles = cv2.HoughCircles(
        img_uint8,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=img.shape[0] // 4,
        param1=100,
        param2=40,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is not None:
        circles = np.round(circles[0, :]).astype(int)
        best = circles[0]
        return {'center': (int(best[0]), int(best[1])), 'radius': int(best[2])}
    return None


def detect_macula(img, optic_disc=None):
    h, w = img.shape[:2]
    if optic_disc:
        cx, cy = optic_disc['center']
        r = optic_disc['radius']
        search_x = cx + int(2.5 * r) if cx < w // 2 else cx - int(2.5 * r)
        search_y = cy
    else:
        search_x = w // 2
        search_y = h // 2

    region_size = min(h, w) // 8
    x1 = max(0, search_x - region_size)
    y1 = max(0, search_y - region_size)
    x2 = min(w, search_x + region_size)
    y2 = min(h, search_y + region_size)

    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    min_loc = np.unravel_index(np.argmin(roi), roi.shape)
    return {
        'center': (x1 + min_loc[1], y1 + min_loc[0]),
        'region': (x1, y1, x2, y2),
    }


def segment_lesions(img):
    edge_map, stats = laplacian_edges(img)
    std = stats['std']

    anomaly_mask = np.logical_or(edge_map <= -2 * std, edge_map >= 2 * std).astype(np.uint8)

    anomaly_mask = morph_close(anomaly_mask, kernel_size=5, iterations=3)
    anomaly_mask = morph_open(anomaly_mask, kernel_size=3, iterations=1)
    anomaly_mask = fill_holes(anomaly_mask)
    anomaly_mask = remove_small_objects(anomaly_mask, min_area=100)

    regions = find_contours(anomaly_mask, min_area=100)
    return regions
