"""
Tests for :mod:`core.intensity`
================================

Unit tests for intensity normalization, CLAHE, HU windowing,
stain normalization, and auto brightness/contrast.
"""

import numpy as np
import cv2
import pytest

from core.intensity import (
    normalize_minmax, normalize_zscore, clahe_enhance,
    hu_windowing, auto_brightness_contrast,
)


class TestNormalizeMinmax:
    def test_default_range(self):
        img = np.array([0, 50, 100, 200, 255], dtype=np.uint8)
        result = normalize_minmax(img)
        assert abs(result.min() - 0.0) < 1e-10
        assert abs(result.max() - 1.0) < 1e-10

    def test_custom_range(self):
        img = np.array([0, 128, 255], dtype=np.uint8)
        result = normalize_minmax(img, 0, 255)
        assert abs(result.min() - 0.0) < 1e-10
        assert abs(result.max() - 255.0) < 1e-10

    def test_constant_image(self):
        img = np.full((10, 10), 50, dtype=np.uint8)
        result = normalize_minmax(img)
        assert np.all(result == 0.0)

    def test_output_dtype(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        result = normalize_minmax(img)
        assert result.dtype == np.float64


class TestNormalizeZscore:
    def test_zero_mean(self):
        rng = np.random.default_rng(42)
        img = rng.normal(100, 30, (50, 50))
        result = normalize_zscore(img)
        assert abs(result.mean()) < 0.01

    def test_unit_variance(self):
        rng = np.random.default_rng(42)
        img = rng.normal(100, 30, (50, 50))
        result = normalize_zscore(img)
        assert abs(result.std() - 1.0) < 0.01

    def test_constant_image(self):
        img = np.full((10, 10), 50.0)
        result = normalize_zscore(img)
        assert np.all(result == 0.0)


class TestClaheEnhance:
    def test_grayscale_uint8(self):
        img = np.random.default_rng(42).integers(0, 256, (100, 100), dtype=np.uint8)
        result = clahe_enhance(img)
        assert result.shape == (100, 100)
        assert result.dtype == np.uint8

    def test_grayscale_float(self):
        img = np.random.default_rng(42).uniform(50, 200, (100, 100)).astype(np.float32)
        result = clahe_enhance(img)
        assert result.shape == (100, 100)

    def test_colour_bgr(self):
        img = np.random.default_rng(42).integers(0, 256, (100, 100, 3), dtype=np.uint8)
        result = clahe_enhance(img)
        assert result.shape == (100, 100, 3)

    def test_increases_contrast(self):
        # Low-contrast image
        img = np.full((100, 100), 128, dtype=np.uint8)
        img[20:80, 20:80] = 130  # very subtle difference
        result = clahe_enhance(img, clip_limit=4.0)
        # CLAHE should increase the range
        assert result.max() >= img.max()


class TestHuWindowing:
    def test_soft_tissue_window(self):
        # CT image in HU: -1000 (air) to +3000 (dense bone)
        img = np.array([-1000, -200, 0, 40, 200, 1000, 3000], dtype=np.float64)
        result = hu_windowing(img, window_center=40, window_width=400)
        assert result.dtype == np.uint8
        assert result.min() == 0
        assert result.max() == 255

    def test_clipping(self):
        img = np.array([-500, 40, 500], dtype=np.float64)
        result = hu_windowing(img, window_center=40, window_width=400)
        # -500 < 40-200=-160 → clipped to lower; 500 > 40+200=240 → clipped to upper
        assert result[0] == 0
        assert result[2] == 255

    def test_brain_window(self):
        img = np.linspace(-50, 100, 100).astype(np.float64)
        result = hu_windowing(img, window_center=35, window_width=80)
        assert result.dtype == np.uint8


class TestAutoBrightnessContrast:
    def test_output_uint8(self):
        img = np.random.default_rng(42).integers(50, 200, (100, 100), dtype=np.uint8)
        result = auto_brightness_contrast(img)
        assert result.dtype == np.uint8

    def test_stretches_range(self):
        # Narrow range image
        img = np.random.default_rng(42).integers(100, 150, (100, 100), dtype=np.uint8)
        result = auto_brightness_contrast(img, clip_percent=1.0)
        # Output should use more of the 0-255 range
        assert (int(result.max()) - int(result.min())) >= (int(img.max()) - int(img.min()))

    def test_colour_input(self):
        img = np.random.default_rng(42).integers(50, 200, (100, 100, 3), dtype=np.uint8)
        result = auto_brightness_contrast(img)
        assert result.shape == (100, 100, 3)
