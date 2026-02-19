import logging
import numpy as np

from pipelines.base import BasePipeline
from core.statistics import roi_stats
from core.classification import classify_severity, build_finding


class HistologyPipeline(BasePipeline):
    """
    Histopathology whole-slide image analysis pipeline.

    Processes SVS/TIFF whole-slide images for:
    - Tissue detection and stain normalization
    - Tile-based processing for gigapixel images
    - Tumor region detection and classification
    - Cancer grading
    """

    def __init__(self, process_id, study_id, parameters=None):
        super().__init__(process_id, study_id, parameters)
        self.tile_size = self.parameters.get('tile_size', 256)
        self.tissue_threshold = self.parameters.get('tissue_threshold', 0.5)

    def load_image(self, image_record):
        from core.image_io import read_image
        result = read_image(image_record['storage_path'])
        if isinstance(result, tuple):
            return result[0]
        return result

    def preprocess(self, img, image_record):
        if img is None:
            return img

        from core.intensity import stain_normalize
        if len(img.shape) == 3 and img.shape[2] == 3:
            try:
                img = stain_normalize(img)
            except Exception as ex:
                self.log.warning(f'Stain normalization failed: {ex}')

        return img

    def segment(self, img, image_record):
        if img is None:
            return []

        tiles = self._extract_tiles(img)
        return tiles

    def _extract_tiles(self, img):
        h, w = img.shape[:2]
        tiles = []
        for y in range(0, h - self.tile_size + 1, self.tile_size):
            for x in range(0, w - self.tile_size + 1, self.tile_size):
                tile = img[y:y + self.tile_size, x:x + self.tile_size]

                if len(tile.shape) == 3:
                    gray = np.mean(tile, axis=2)
                else:
                    gray = tile

                tissue_ratio = np.sum(gray < 220) / gray.size
                if tissue_ratio >= self.tissue_threshold:
                    tiles.append({
                        'tile': tile,
                        'x': x, 'y': y,
                        'w': self.tile_size, 'h': self.tile_size,
                        'tissue_ratio': tissue_ratio,
                    })

        self.log.info(f'Extracted {len(tiles)} tiles from {h}x{w} image')
        return tiles

    def calculate_stats(self, img, regions):
        stats_list = []
        for tile_info in regions:
            tile = tile_info['tile']
            if len(tile.shape) == 3:
                gray = np.mean(tile, axis=2).astype(np.float32)
            else:
                gray = tile.astype(np.float32)
            stats = roi_stats(gray)
            stats['tile_x'] = tile_info['x']
            stats['tile_y'] = tile_info['y']
            stats['tissue_ratio'] = tile_info['tissue_ratio']
            stats_list.append(stats)
        return stats_list

    def detect_anomalies(self, img, regions, stats, image_record):
        if not stats:
            return []

        all_means = [s['mean'] for s in stats if 'mean' in s]
        if not all_means:
            return []
        global_mean = np.mean(all_means)
        global_std = np.std(all_means) if len(all_means) > 1 else 1.0

        findings = []
        for tile_info, tile_stats in zip(regions, stats):
            mean_val = tile_stats.get('mean', 0)
            z_score = abs(mean_val - global_mean) / global_std if global_std > 0 else 0

            if z_score < 2.0:
                continue

            gradient = tile_stats.get('gradient', 0)
            confidence = min(1.0, z_score / 5.0)
            severity = classify_severity(gradient, confidence)

            finding = build_finding(
                finding_type='TISSUE_ANOMALY',
                severity=severity,
                confidence=confidence,
                statistics=tile_stats,
                geometric_properties={
                    'area': tile_info['w'] * tile_info['h'],
                    'tissue_ratio': tile_info['tissue_ratio'],
                },
                location={
                    'x': tile_info['x'],
                    'y': tile_info['y'],
                    'w': tile_info['w'],
                    'h': tile_info['h'],
                },
            )
            finding['disease_category'] = 'HISTOLOGY'
            findings.append(finding)

        return findings
