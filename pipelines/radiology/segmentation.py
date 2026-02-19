import numpy as np

from core.clustering import kmeans_segment
from core.morphology import morph_close, morph_open, fill_holes, remove_small_objects
from core.contours import find_contours


def segment_brain_regions(slice_2d, k=4):
    segmented, labels, centers = kmeans_segment(slice_2d.astype(np.float32), k=k)
    return segmented, labels, centers


def segment_lesions_3d(volume, threshold_factor=2.0):
    mean_val = np.mean(volume)
    std_val = np.std(volume)
    threshold = mean_val + threshold_factor * std_val

    lesion_mask = (volume > threshold).astype(np.uint8)

    for z in range(lesion_mask.shape[0]):
        slc = lesion_mask[z]
        slc = morph_close(slc, kernel_size=3, iterations=2)
        slc = fill_holes(slc)
        slc = remove_small_objects(slc, min_area=50)
        lesion_mask[z] = slc

    return lesion_mask


def measure_volume(mask, voxel_spacing=(1.0, 1.0, 1.0)):
    voxel_volume_mm3 = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]
    n_voxels = np.sum(mask > 0)
    volume_mm3 = n_voxels * voxel_volume_mm3
    volume_cm3 = volume_mm3 / 1000.0
    return {
        'n_voxels': int(n_voxels),
        'volume_mm3': float(volume_mm3),
        'volume_cm3': float(volume_cm3),
    }
