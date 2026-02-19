"""
VIDIO Core - Shape Detection & Features
=========================================
Detect geometric shapes from contours, classify them, and extract
comprehensive shape feature vectors.
"""

import logging

import numpy as np
import cv2

logger = logging.getLogger(__name__)


def shape_features(contour: np.ndarray) -> dict:
    """Compute a comprehensive set of shape features for a contour.

    Parameters
    ----------
    contour : np.ndarray
        Contour points as returned by cv2.findContours.

    Returns
    -------
    dict
        Dictionary with keys: area, perimeter, circularity, solidity,
        convexity, elongation, aspect_ratio, extent, equivalent_diameter,
        num_vertices (from polygon approximation), bounding_rect,
        min_enclosing_circle_radius, centroid, angle, hu_moments.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)

    # Convex hull
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, closed=True)

    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / float(h) if h > 0 else 1.0
    extent = area / float(w * h) if (w * h) > 0 else 0.0

    # Circularity
    circularity = (4.0 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

    # Solidity
    solidity = area / hull_area if hull_area > 0 else 0.0

    # Convexity
    convexity = hull_perimeter / perimeter if perimeter > 0 else 0.0

    # Equivalent diameter
    equivalent_diameter = np.sqrt(4 * area / np.pi) if area > 0 else 0.0

    # Minimum enclosing circle
    (mc_x, mc_y), mc_radius = cv2.minEnclosingCircle(contour)

    # Polygon approximation
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    num_vertices = len(approx)

    # Ellipse fitting for elongation and angle
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        center = ellipse[0]
        axes = ellipse[1]
        angle = ellipse[2]
        minor_axis = min(axes)
        major_axis = max(axes)
        elongation = minor_axis / major_axis if major_axis > 0 else 1.0
    else:
        rect = cv2.minAreaRect(contour)
        center = rect[0]
        rw, rh = rect[1]
        angle = rect[2]
        minor_axis = min(rw, rh)
        major_axis = max(rw, rh)
        elongation = minor_axis / major_axis if major_axis > 0 else 1.0

    # Centroid
    M = cv2.moments(contour)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx, cy = center

    # Hu moments (log-transformed, 7 values)
    hu = cv2.HuMoments(M).flatten()
    # Log-transform for scale invariance (handle zeros)
    hu_log = np.array([
        -np.sign(h) * np.log10(abs(h)) if abs(h) > 1e-30 else 0.0
        for h in hu
    ])

    features = {
        "area": float(area),
        "perimeter": float(perimeter),
        "circularity": float(circularity),
        "solidity": float(solidity),
        "convexity": float(convexity),
        "elongation": float(elongation),
        "aspect_ratio": float(aspect_ratio),
        "extent": float(extent),
        "equivalent_diameter": float(equivalent_diameter),
        "num_vertices": int(num_vertices),
        "bounding_rect": (int(x), int(y), int(w), int(h)),
        "min_enclosing_circle_radius": float(mc_radius),
        "centroid": (float(cx), float(cy)),
        "angle": float(angle),
        "hu_moments": hu_log.tolist(),
    }
    return features


def detect_shape(contour: np.ndarray) -> str:
    """Classify a contour as a geometric shape.

    Uses polygon approximation and circularity to determine the shape.

    Parameters
    ----------
    contour : np.ndarray
        Contour points.

    Returns
    -------
    str
        One of: ``"triangle"``, ``"rectangle"``, ``"square"``,
        ``"pentagon"``, ``"circle"``, ``"irregular"``.
    """
    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter < 1e-6:
        return "irregular"

    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    num_vertices = len(approx)

    # Check circularity first
    area = cv2.contourArea(contour)
    circularity = (4.0 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0.0

    if circularity > 0.85 and num_vertices > 6:
        return "circle"

    if num_vertices == 3:
        return "triangle"

    if num_vertices == 4:
        # Distinguish square from rectangle using aspect ratio
        x, y, w, h = cv2.boundingRect(approx)
        aspect = float(w) / float(h) if h > 0 else 1.0
        if 0.85 <= aspect <= 1.15:
            return "square"
        return "rectangle"

    if num_vertices == 5:
        return "pentagon"

    # Check convexity for irregular shapes
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    if solidity < 0.7:
        return "irregular"

    # High vertex count but not circular
    if num_vertices > 6:
        if circularity > 0.7:
            return "circle"
        return "irregular"

    return "irregular"


def is_circular(contour: np.ndarray, threshold: float = 0.85) -> bool:
    """Test whether a contour is approximately circular.

    Parameters
    ----------
    contour : np.ndarray
        Contour points.
    threshold : float
        Circularity threshold in [0, 1]. Values above this are circular.

    Returns
    -------
    bool
        True if the contour circularity exceeds the threshold.
    """
    perimeter = cv2.arcLength(contour, closed=True)
    if perimeter < 1e-6:
        return False
    area = cv2.contourArea(contour)
    circularity = (4.0 * np.pi * area) / (perimeter ** 2)
    result = circularity >= threshold
    logger.debug("is_circular: circularity=%.3f threshold=%.3f -> %s", circularity, threshold, result)
    return result


def is_irregular(contour: np.ndarray, convexity_threshold: float = 0.7) -> bool:
    """Test whether a contour is irregular (non-convex).

    A contour is considered irregular if its solidity (ratio of area to
    convex hull area) is below the given threshold.

    Parameters
    ----------
    contour : np.ndarray
        Contour points.
    convexity_threshold : float
        Solidity threshold. Values below this are irregular.

    Returns
    -------
    bool
        True if the contour is irregular.
    """
    area = cv2.contourArea(contour)
    if area < 1e-6:
        return True
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0
    result = solidity < convexity_threshold
    logger.debug("is_irregular: solidity=%.3f threshold=%.3f -> %s", solidity, convexity_threshold, result)
    return result
