"""
VIDIO Core - Clustering & Segmentation
========================================
K-Means (with automatic K selection), Mean Shift, and HDBSCAN clustering
for image segmentation and feature-space analysis.
"""

import logging

import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Optional: HDBSCAN
try:
    import hdbscan as _hdbscan
    _HAS_HDBSCAN = True
except ImportError:
    _HAS_HDBSCAN = False
    logger.debug("hdbscan not available; hdbscan_cluster will raise ImportError.")

# Optional: sklearn for silhouette score and MeanShift
try:
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import MeanShift, estimate_bandwidth
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    logger.debug("scikit-learn not available; some clustering functions may be limited.")


def adaptive_kmeans(data: np.ndarray, k_range: tuple = (2, 10),
                    criteria_eps: float = 0.1, max_iter: int = 20) -> tuple:
    """Run K-Means for a range of K values and auto-select the best K
    using the silhouette score.

    Parameters
    ----------
    data : np.ndarray
        Feature matrix of shape (n_samples, n_features), float32.
    k_range : tuple
        (k_min, k_max) inclusive range of K values to try.
    criteria_eps : float
        Epsilon for the OpenCV K-Means termination criteria.
    max_iter : int
        Maximum iterations for each K-Means run.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        (labels, centers, best_k)
    """
    if not _HAS_SKLEARN:
        raise ImportError("scikit-learn is required for silhouette-based K selection. "
                          "Install with: pip install scikit-learn")

    data = np.ascontiguousarray(data.astype(np.float32))
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    k_min, k_max = k_range
    k_min = max(2, k_min)
    k_max = min(k_max, data.shape[0] - 1)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, criteria_eps)

    best_score = -1.0
    best_k = k_min
    best_labels = None
    best_centers = None

    for k in range(k_min, k_max + 1):
        _, labels, centers = cv2.kmeans(
            data, k, None, criteria, attempts=3, flags=cv2.KMEANS_PP_CENTERS
        )
        labels_flat = labels.flatten()

        # Silhouette score needs at least 2 unique labels
        n_unique = len(np.unique(labels_flat))
        if n_unique < 2:
            continue

        # Subsample for speed if large dataset
        if data.shape[0] > 10000:
            idx = np.random.choice(data.shape[0], 10000, replace=False)
            score = silhouette_score(data[idx], labels_flat[idx])
        else:
            score = silhouette_score(data, labels_flat)

        logger.debug("adaptive_kmeans: k=%d  silhouette=%.4f", k, score)
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels_flat
            best_centers = centers

    logger.info("adaptive_kmeans: best_k=%d  silhouette=%.4f", best_k, best_score)
    return best_labels, best_centers, best_k


def kmeans_segment(img: np.ndarray, k: int = 3) -> tuple:
    """Segment an image into k colour clusters using K-Means.

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR uint8).
    k : int
        Number of clusters.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (segmented_image, labels_2d, centers)
        - segmented_image: image recoloured by cluster centres (uint8)
        - labels_2d: cluster label for each pixel (shape = img height x width)
        - centers: cluster centres (k x channels, uint8)
    """
    h, w = img.shape[:2]
    if img.ndim == 3:
        data = img.reshape(-1, img.shape[2]).astype(np.float32)
    else:
        data = img.reshape(-1, 1).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    _, labels, centers = cv2.kmeans(
        data, k, None, criteria, attempts=5, flags=cv2.KMEANS_PP_CENTERS
    )

    centers = centers.astype(np.uint8)
    labels_flat = labels.flatten()
    segmented = centers[labels_flat].reshape(img.shape)
    labels_2d = labels_flat.reshape(h, w)

    logger.info("kmeans_segment: k=%d  image_shape=%s", k, img.shape)
    return segmented, labels_2d, centers


def mean_shift_segment(img: np.ndarray, bandwidth: float = None) -> tuple:
    """Segment image features using Mean Shift clustering.

    Parameters
    ----------
    img : np.ndarray
        Input image (BGR uint8).
    bandwidth : float, optional
        Kernel bandwidth. None = auto-estimate.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (labels_2d, centers)
    """
    if not _HAS_SKLEARN:
        raise ImportError("scikit-learn is required for Mean Shift clustering. "
                          "Install with: pip install scikit-learn")

    h, w = img.shape[:2]
    if img.ndim == 3:
        data = img.reshape(-1, img.shape[2]).astype(np.float64)
    else:
        data = img.reshape(-1, 1).astype(np.float64)

    # Subsample for bandwidth estimation if large
    if bandwidth is None:
        n_samples = min(data.shape[0], 5000)
        idx = np.random.choice(data.shape[0], n_samples, replace=False)
        bandwidth = estimate_bandwidth(data[idx], quantile=0.2)
        if bandwidth < 1e-3:
            bandwidth = 1.0
            logger.warning("mean_shift_segment: estimated bandwidth too small, using 1.0")

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(data)

    labels_2d = ms.labels_.reshape(h, w)
    centers = ms.cluster_centers_

    logger.info("mean_shift_segment: bandwidth=%.2f  n_clusters=%d", bandwidth, len(centers))
    return labels_2d, centers


def hdbscan_cluster(data: np.ndarray, min_cluster_size: int = 50,
                    min_samples: int = 5) -> tuple:
    """Cluster data using HDBSCAN (density-based, no need to specify K).

    Parameters
    ----------
    data : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    min_cluster_size : int
        Minimum number of samples in a cluster.
    min_samples : int
        Number of samples in a neighbourhood for a point to be core.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (labels, probabilities)
        Labels of -1 indicate noise points.
    """
    if not _HAS_HDBSCAN:
        raise ImportError("hdbscan is required for HDBSCAN clustering. "
                          "Install with: pip install hdbscan")

    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    clusterer = _hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        gen_min_span_tree=False,
    )
    clusterer.fit(data)

    labels = clusterer.labels_
    probabilities = clusterer.probabilities_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    logger.info("hdbscan_cluster: n_clusters=%d  n_noise=%d", n_clusters, n_noise)
    return labels, probabilities
