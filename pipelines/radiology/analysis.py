import numpy as np

from core.statistics import roi_stats
from core.classification import classify_severity, build_finding


def analyze_brain_atrophy(volume, brain_mask=None):
    if brain_mask is not None:
        brain_volume = np.sum(brain_mask > 0)
        total_volume = brain_mask.size
        brain_ratio = brain_volume / total_volume
    else:
        threshold = np.mean(volume) * 0.3
        brain_volume = np.sum(volume > threshold)
        total_volume = volume.size
        brain_ratio = brain_volume / total_volume

    return {
        'brain_volume_voxels': int(brain_volume),
        'total_volume_voxels': int(total_volume),
        'brain_parenchymal_fraction': float(brain_ratio),
    }


def analyze_lesion(volume, lesion_mask, voxel_spacing=(1.0, 1.0, 1.0)):
    from pipelines.radiology.segmentation import measure_volume

    vol_info = measure_volume(lesion_mask, voxel_spacing)

    lesion_values = volume[lesion_mask > 0]
    if lesion_values.size == 0:
        return vol_info

    stats = {
        'mean_intensity': float(np.mean(lesion_values)),
        'std_intensity': float(np.std(lesion_values)),
        'min_intensity': float(np.min(lesion_values)),
        'max_intensity': float(np.max(lesion_values)),
    }
    vol_info.update(stats)
    return vol_info
