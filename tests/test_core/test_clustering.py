"""
Tests for :mod:`core.clustering`
=================================

Unit tests for adaptive K-Means, colour K-Means segmentation,
Mean Shift, and HDBSCAN clustering.
"""

import numpy as np
import pytest

from core.clustering import adaptive_kmeans, kmeans_segment


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

@pytest.fixture
def two_cluster_data():
    """Generate clearly separated 2-cluster data in 2D."""
    rng = np.random.default_rng(42)
    c1 = rng.normal(loc=[10, 10], scale=1.0, size=(100, 2))
    c2 = rng.normal(loc=[50, 50], scale=1.0, size=(100, 2))
    return np.vstack([c1, c2]).astype(np.float32)


@pytest.fixture
def three_cluster_data():
    """Generate clearly separated 3-cluster data in 2D."""
    rng = np.random.default_rng(42)
    c1 = rng.normal(loc=[10, 10], scale=1.0, size=(80, 2))
    c2 = rng.normal(loc=[50, 10], scale=1.0, size=(80, 2))
    c3 = rng.normal(loc=[30, 50], scale=1.0, size=(80, 2))
    return np.vstack([c1, c2, c3]).astype(np.float32)


@pytest.fixture
def gray_image():
    """Create a synthetic 60x60 grayscale image with 3 distinct bands."""
    img = np.zeros((60, 60), dtype=np.uint8)
    img[0:20, :] = 50     # dark band
    img[20:40, :] = 128   # medium band
    img[40:60, :] = 220   # bright band
    return img


@pytest.fixture
def color_image():
    """Create a synthetic 60x60 BGR image with distinct regions."""
    img = np.zeros((60, 60, 3), dtype=np.uint8)
    img[0:30, 0:30] = [255, 0, 0]     # blue region
    img[0:30, 30:60] = [0, 255, 0]    # green region
    img[30:60, :] = [0, 0, 255]       # red region
    return img


# ------------------------------------------------------------------ #
#  adaptive_kmeans                                                     #
# ------------------------------------------------------------------ #

class TestAdaptiveKmeans:
    """Tests for adaptive_kmeans()."""

    def test_finds_two_clusters(self, two_cluster_data):
        labels, centers, best_k = adaptive_kmeans(two_cluster_data, k_range=(2, 5))
        assert best_k == 2
        assert len(np.unique(labels)) == 2

    def test_finds_three_clusters(self, three_cluster_data):
        labels, centers, best_k = adaptive_kmeans(three_cluster_data, k_range=(2, 6))
        assert best_k == 3
        assert len(np.unique(labels)) == 3

    def test_labels_shape(self, two_cluster_data):
        labels, centers, best_k = adaptive_kmeans(two_cluster_data)
        assert labels.shape[0] == two_cluster_data.shape[0]

    def test_centers_shape(self, two_cluster_data):
        labels, centers, best_k = adaptive_kmeans(two_cluster_data)
        assert centers.shape[0] == best_k
        assert centers.shape[1] == two_cluster_data.shape[1]

    def test_1d_input(self):
        """1D input should be reshaped internally to (N, 1)."""
        rng = np.random.default_rng(42)
        data = np.concatenate([
            rng.normal(10, 1, 100),
            rng.normal(50, 1, 100),
        ]).astype(np.float32)
        labels, centers, best_k = adaptive_kmeans(data, k_range=(2, 4))
        assert best_k == 2


# ------------------------------------------------------------------ #
#  kmeans_segment                                                      #
# ------------------------------------------------------------------ #

class TestKmeansSegment:
    """Tests for kmeans_segment()."""

    def test_grayscale_output_shape(self, gray_image):
        segmented, labels, centers = kmeans_segment(gray_image, k=3)
        assert segmented.shape == gray_image.shape
        assert labels.shape == (60, 60)

    def test_correct_number_of_clusters(self, gray_image):
        segmented, labels, centers = kmeans_segment(gray_image, k=3)
        assert centers.shape[0] == 3
        assert len(np.unique(labels)) == 3

    def test_colour_output_shape(self, color_image):
        segmented, labels, centers = kmeans_segment(color_image, k=3)
        assert segmented.shape == color_image.shape
        assert labels.shape == (60, 60)

    def test_segmented_dtype(self, gray_image):
        segmented, _, _ = kmeans_segment(gray_image, k=2)
        assert segmented.dtype == np.uint8


# ------------------------------------------------------------------ #
#  Optional: Mean Shift and HDBSCAN                                    #
# ------------------------------------------------------------------ #

class TestMeanShiftSegment:
    """Tests for mean_shift_segment() — requires scikit-learn."""

    def test_import_available(self):
        """Verify scikit-learn is importable for Mean Shift."""
        try:
            from core.clustering import mean_shift_segment
        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_basic_segmentation(self, gray_image):
        try:
            from core.clustering import mean_shift_segment
        except ImportError:
            pytest.skip("scikit-learn not available")
        labels, centers = mean_shift_segment(gray_image)
        assert labels.shape == (60, 60)
        assert len(centers) >= 1


class TestHDBSCAN:
    """Tests for hdbscan_cluster() — requires hdbscan."""

    def test_import_available(self):
        try:
            from core.clustering import hdbscan_cluster
        except ImportError:
            pytest.skip("hdbscan not available")

    def test_basic_clustering(self, two_cluster_data):
        try:
            from core.clustering import hdbscan_cluster
        except ImportError:
            pytest.skip("hdbscan not available")
        labels, probs = hdbscan_cluster(two_cluster_data, min_cluster_size=20)
        assert labels.shape[0] == two_cluster_data.shape[0]
        assert probs.shape[0] == two_cluster_data.shape[0]
        # Should find 2 clusters (labels 0 and 1), possibly with noise (-1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        assert n_clusters >= 1
