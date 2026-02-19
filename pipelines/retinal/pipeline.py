import logging
import numpy as np

from pipelines.base import BasePipeline
from core.filtering import bilateral_filter
from core.intensity import clahe_enhance, normalize_minmax
from core.edges import laplacian_edges
from core.morphology import morph_open, morph_close, remove_small_objects
from core.contours import find_contours, contour_stats
from core.clustering import adaptive_kmeans, kmeans_segment
from core.statistics import roi_stats
from core.classification import classify_severity, build_finding


class RetinalPipeline(BasePipeline):
    """
    Retinal image analysis pipeline.

    Processes fundus photography, OCT, and slit-lamp images for:
    - Vessel segmentation and analysis
    - Optic disc and macula detection
    - Lesion detection (drusen, hemorrhages, exudates)
    - Amyloid plaque candidate detection in crystalline lens images
    """

    def preprocess(self, img, image_record):
        if img is None:
            return img

        if isinstance(img, tuple):
            img = img[0]

        if len(img.shape) == 3 and img.shape[2] >= 3:
            green = img[:, :, 1].astype(np.float32)
        else:
            green = img.astype(np.float32) if len(img.shape) == 2 else img[:, :, 0].astype(np.float32)

        enhanced = clahe_enhance(green, clip_limit=3.0, grid_size=(8, 8))
        filtered = bilateral_filter(enhanced, d=0, sigma_color=2.0, sigma_space=5.0)

        return filtered

    def segment(self, img, image_record):
        if img is None:
            return []

        edge_map, edge_stats = laplacian_edges(img)

        std = edge_stats['std']
        smooth_mask = np.logical_and(edge_map > -std, edge_map < std).astype(np.uint8)

        cleaned = morph_open(smooth_mask, kernel_size=3, iterations=3)
        cleaned = morph_close(cleaned, kernel_size=3, iterations=2)
        cleaned = remove_small_objects(cleaned, min_area=200)

        regions = find_contours(cleaned, min_area=100)
        return regions

    def calculate_stats(self, img, regions):
        if not regions:
            return []

        stats_list = []
        for region in regions:
            x, y, w, h = (region['stats']['x'], region['stats']['y'],
                          region['stats']['w'], region['stats']['h'])

            x, y = max(0, x), max(0, y)
            roi = img[y:y + h, x:x + w]

            if roi.size == 0:
                stats_list.append({})
                continue

            stats = roi_stats(roi)
            stats['geometric'] = region['stats']
            stats_list.append(stats)

        return stats_list

    def detect_anomalies(self, img, regions, stats, image_record):
        findings = []

        if not regions or not stats:
            return findings

        global_stats = roi_stats(img) if img is not None else {}
        global_mean = global_stats.get('mean', 0)
        global_std = global_stats.get('std', 1)

        for region, region_stats in zip(regions, stats):
            if not region_stats:
                continue

            mean_val = region_stats.get('mean', 0)
            gradient = region_stats.get('gradient', 0)
            geometric = region_stats.get('geometric', {})

            z_score = abs(mean_val - global_mean) / global_std if global_std > 0 else 0

            if z_score < 1.5 and gradient < 20:
                continue

            solidity = geometric.get('solidity', 1.0)
            elongation = geometric.get('elongation', 1.0)
            area = geometric.get('area', 0)

            confidence = min(1.0, z_score / 5.0)
            severity = classify_severity(gradient, confidence)

            if solidity > 0.8 and elongation < 1.5:
                finding_type = 'LESION'
                disease = 'RETINAL_LESION'
            elif elongation > 3.0:
                finding_type = 'VESSEL_ANOMALY'
                disease = 'VASCULAR'
            else:
                finding_type = 'REGION_ANOMALY'
                disease = 'UNCLASSIFIED'

            finding = build_finding(
                finding_type=finding_type,
                severity=severity,
                confidence=confidence,
                statistics={
                    'mean': region_stats.get('mean'),
                    'std': region_stats.get('std'),
                    'min': region_stats.get('min'),
                    'max': region_stats.get('max'),
                    'gradient': gradient,
                    'z_score': z_score,
                },
                geometric_properties={
                    'area': area,
                    'solidity': solidity,
                    'elongation': elongation,
                    'convexity': geometric.get('convexity', 1.0),
                    'circularity': geometric.get('circularity', 0),
                },
                location={
                    'x': geometric.get('x', 0),
                    'y': geometric.get('y', 0),
                    'w': geometric.get('w', 0),
                    'h': geometric.get('h', 0),
                },
            )
            finding['disease_category'] = disease
            findings.append(finding)

        return findings
