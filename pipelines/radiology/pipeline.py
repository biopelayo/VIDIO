import logging
import numpy as np

from pipelines.base import BasePipeline
from core.statistics import roi_stats
from core.classification import classify_severity, build_finding


class RadiologyPipeline(BasePipeline):
    """
    Radiology image analysis pipeline.

    Processes DICOM CT/MRI volumes for:
    - Brain MRI: atrophy measurement, lesion detection
    - CT: tumor segmentation, volumetric analysis
    - Skull stripping, bias correction, registration
    """

    def load_image(self, image_record):
        fmt = image_record.get('file_format', '').upper()
        path = image_record['storage_path']

        if fmt == 'NIFTI' or path.endswith(('.nii', '.nii.gz')):
            from core.image_io import read_nifti
            return read_nifti(path)
        elif fmt == 'DICOM':
            from core.image_io import read_dicom
            return read_dicom(path)
        else:
            from core.image_io import read_image
            return read_image(path)

    def preprocess(self, img, image_record):
        if img is None:
            return img

        if isinstance(img, tuple):
            img_data, metadata = img
        else:
            img_data = img
            metadata = {}

        img_data = img_data.astype(np.float32)

        modality = image_record.get('info', {}).get('modality', '')
        if modality == 'CT':
            from core.intensity import hu_windowing
            window_center = image_record.get('info', {}).get('window_center', 40)
            window_width = image_record.get('info', {}).get('window_width', 400)
            img_data = hu_windowing(img_data, window_center, window_width).astype(np.float32)
        else:
            from core.intensity import normalize_minmax
            img_data = normalize_minmax(img_data, 0, 255)

        return img_data

    def segment(self, img, image_record):
        if img is None:
            return []

        if len(img.shape) == 3:
            return self._segment_3d(img)
        else:
            return self._segment_2d(img)

    def _segment_2d(self, img):
        from core.edges import laplacian_edges
        from core.morphology import morph_close, remove_small_objects
        from core.contours import find_contours

        edge_map, stats = laplacian_edges(img.astype(np.float32))
        threshold = stats['std'] * 1.5
        anomaly_mask = (np.abs(edge_map) > threshold).astype(np.uint8)
        anomaly_mask = morph_close(anomaly_mask, kernel_size=5, iterations=3)
        anomaly_mask = remove_small_objects(anomaly_mask, min_area=200)
        regions = find_contours(anomaly_mask, min_area=100)
        return regions

    def _segment_3d(self, volume):
        regions = []
        n_slices = volume.shape[2] if volume.ndim == 3 else volume.shape[0]

        for z in range(n_slices):
            if volume.ndim == 3 and volume.shape[2] < volume.shape[0]:
                slc = volume[:, :, z]
            else:
                slc = volume[z]

            slice_regions = self._segment_2d(slc)
            for r in slice_regions:
                r['stats']['z'] = z
            regions.extend(slice_regions)

        return regions

    def calculate_stats(self, img, regions):
        stats_list = []
        for region in regions:
            s = region.get('stats', {})
            x, y, w, h = s.get('x', 0), s.get('y', 0), s.get('w', 0), s.get('h', 0)
            z = s.get('z', None)

            if z is not None and len(img.shape) == 3:
                if img.shape[2] < img.shape[0]:
                    slc = img[:, :, z]
                else:
                    slc = img[z]
            else:
                slc = img

            roi = slc[max(0, y):y + h, max(0, x):x + w]
            if roi.size == 0:
                stats_list.append({})
                continue

            stats = roi_stats(roi.astype(np.float32))
            stats['geometric'] = s
            stats_list.append(stats)

        return stats_list

    def detect_anomalies(self, img, regions, stats, image_record):
        findings = []
        if not regions or not stats:
            return findings

        for region, region_stats in zip(regions, stats):
            if not region_stats:
                continue

            gradient = region_stats.get('gradient', 0)
            std_val = region_stats.get('std', 0)
            geometric = region_stats.get('geometric', {})

            if gradient < 10 and std_val < 5:
                continue

            confidence = min(1.0, gradient / 100.0)
            severity = classify_severity(gradient, confidence)

            finding = build_finding(
                finding_type='RADIOLOGICAL_ANOMALY',
                severity=severity,
                confidence=confidence,
                statistics=region_stats,
                geometric_properties={
                    'area': geometric.get('area', 0),
                    'solidity': geometric.get('solidity', 0),
                    'elongation': geometric.get('elongation', 0),
                },
                location={
                    'x': geometric.get('x', 0),
                    'y': geometric.get('y', 0),
                    'z': geometric.get('z'),
                    'w': geometric.get('w', 0),
                    'h': geometric.get('h', 0),
                },
            )
            finding['disease_category'] = 'RADIOLOGY'
            findings.append(finding)

        return findings
