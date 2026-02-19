import numpy as np

from core.edges import laplacian_edges
from core.morphology import morph_close, morph_open, remove_small_objects
from core.contours import find_contours
from core.clustering import kmeans_segment


def segment_tumor_regions(img):
    if len(img.shape) == 3:
        gray = np.mean(img, axis=2).astype(np.float32)
    else:
        gray = img.astype(np.float32)

    segmented, labels, centers = kmeans_segment(gray, k=3)

    sorted_centers = np.sort(centers.flatten())
    darkest_cluster = sorted_centers[0]

    tumor_mask = (segmented.astype(np.float32) == darkest_cluster).astype(np.uint8)
    tumor_mask = morph_close(tumor_mask, kernel_size=5, iterations=3)
    tumor_mask = morph_open(tumor_mask, kernel_size=3, iterations=1)
    tumor_mask = remove_small_objects(tumor_mask, min_area=500)

    regions = find_contours(tumor_mask, min_area=200)
    return regions
