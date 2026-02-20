"""
Tests for :mod:`core.statistics`
=================================

Unit tests for ROI statistics, region statistics, and z-score
region comparison.
"""

import numpy as np
import pytest

from core.statistics import roi_stats, region_stats, compare_regions


@pytest.fixture
def uniform_image():
    """A completely uniform grayscale image (value = 100)."""
    return np.full((50, 50), 100.0, dtype=np.float64)


@pytest.fixture
def gradient_image():
    """A horizontal gradient from 0 to 255."""
    return np.tile(np.linspace(0, 255, 100), (100, 1)).astype(np.float64)


@pytest.fixture
def noisy_image():
    """An image with known mean and std."""
    rng = np.random.default_rng(42)
    return rng.normal(loc=128, scale=30, size=(100, 100)).astype(np.float64)


class TestRoiStats:
    def test_returns_all_keys(self, gradient_image):
        stats = roi_stats(gradient_image)
        expected_keys = {'min', 'max', 'mean', 'std', 'median', 'mode',
                         'gradient', 'skewness', 'kurtosis'}
        assert expected_keys.issubset(stats.keys())

    def test_uniform_zero_std(self, uniform_image):
        stats = roi_stats(uniform_image)
        assert abs(stats['std']) < 1e-6
        assert abs(stats['mean'] - 100.0) < 1e-6

    def test_gradient_range(self, gradient_image):
        stats = roi_stats(gradient_image)
        assert stats['min'] >= 0
        assert stats['max'] <= 255
        assert stats['gradient'] > 0  # max - mean > 0

    def test_noisy_stats(self, noisy_image):
        stats = roi_stats(noisy_image)
        assert abs(stats['mean'] - 128) < 5  # should be near 128
        assert abs(stats['std'] - 30) < 5    # should be near 30

    def test_with_mask(self, gradient_image):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:50, 0:50] = 255  # top-left quadrant only
        stats = roi_stats(gradient_image, mask=mask)
        # Top-left has lower values
        assert stats['mean'] < 128

    def test_empty_roi_returns_zeros(self):
        img = np.zeros((10, 10), dtype=np.float64)
        mask = np.zeros((10, 10), dtype=np.uint8)
        stats = roi_stats(img, mask=mask)
        assert stats['mean'] == 0.0
        assert stats['std'] == 0.0

    def test_bgr_input(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        img[:, :, 0] = 100  # blue=100
        img[:, :, 1] = 150  # green=150
        img[:, :, 2] = 200  # red=200
        stats = roi_stats(img)
        # Should convert to grayscale internally
        assert 'mean' in stats


class TestRegionStats:
    def test_returns_list(self, gradient_image):
        masks = [
            np.ones((100, 100), dtype=np.uint8) * 255,
        ]
        result = region_stats(gradient_image, masks)
        assert isinstance(result, list)
        assert len(result) == 1
        assert 'region_index' in result[0]

    def test_multiple_regions(self, gradient_image):
        masks = []
        for i in range(3):
            m = np.zeros((100, 100), dtype=np.uint8)
            m[i*30:(i+1)*30, :] = 255
            masks.append(m)
        result = region_stats(gradient_image, masks)
        assert len(result) == 3


class TestCompareRegions:
    def test_returns_z_scores(self):
        stats_list = [
            {'mean': 10, 'std': 2, 'gradient': 5, 'min': 0, 'max': 20,
             'median': 10, 'mode': 10, 'skewness': 0.1, 'kurtosis': 0.2,
             'region_index': 0},
            {'mean': 50, 'std': 5, 'gradient': 20, 'min': 10, 'max': 90,
             'median': 50, 'mode': 50, 'skewness': 0.5, 'kurtosis': 1.0,
             'region_index': 1},
        ]
        z_scores = compare_regions(stats_list)
        assert len(z_scores) == 2
        assert 'mean' in z_scores[0]
        # Z-scores should be opposite signs for the two regions
        assert z_scores[0]['mean'] * z_scores[1]['mean'] <= 0
