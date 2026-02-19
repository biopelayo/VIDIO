import numpy as np

from core.clustering import kmeans_segment
from core.statistics import roi_stats
from core.classification import classify_severity, build_finding


def analyze_retinal_regions(img, regions, global_stats=None):
    if global_stats is None:
        global_stats = roi_stats(img)

    findings = []
    global_mean = global_stats.get('mean', 0)
    global_std = global_stats.get('std', 1)

    for region in regions:
        stats_data = region.get('stats', {})
        x = stats_data.get('x', 0)
        y = stats_data.get('y', 0)
        w = stats_data.get('w', 0)
        h = stats_data.get('h', 0)

        roi = img[max(0, y):y + h, max(0, x):x + w]
        if roi.size == 0:
            continue

        region_stat = roi_stats(roi)
        mean_val = region_stat.get('mean', 0)
        gradient = region_stat.get('gradient', 0)

        z_score = abs(mean_val - global_mean) / global_std if global_std > 0 else 0

        if z_score < 1.5:
            continue

        confidence = min(1.0, z_score / 5.0)
        severity = classify_severity(gradient, confidence)

        finding = build_finding(
            finding_type='RETINAL_ANOMALY',
            severity=severity,
            confidence=confidence,
            statistics=region_stat,
            geometric_properties=stats_data,
            location={'x': x, 'y': y, 'w': w, 'h': h},
        )
        findings.append(finding)

    return findings


def detect_bright_lesions(img, threshold_factor=2.0):
    stats = roi_stats(img)
    threshold = stats['mean'] + threshold_factor * stats['std']
    bright_mask = (img > threshold).astype(np.uint8)
    return bright_mask


def detect_dark_lesions(img, threshold_factor=2.0):
    stats = roi_stats(img)
    threshold = stats['mean'] - threshold_factor * stats['std']
    dark_mask = (img < threshold).astype(np.uint8)
    return dark_mask
