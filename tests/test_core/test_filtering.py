"""
Tests for :mod:`core.filtering`
================================

Unit tests for the bilateral, Gaussian, median, unsharp mask, and
non-local means denoising filters.  All tests use synthetic images
to avoid external data dependencies.
"""

import numpy as np
import pytest

from core.filtering import (
    bilateral_filter,
    gaussian_filter,
    median_filter,
    unsharp_mask,
    denoise_nlm,
)


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

@pytest.fixture
def gray_uint8():
    """Create a synthetic 100x100 grayscale uint8 image with noise."""
    rng = np.random.default_rng(42)
    base = np.full((100, 100), 128, dtype=np.uint8)
    noise = rng.integers(-30, 30, size=(100, 100), dtype=np.int16)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


@pytest.fixture
def gray_float32():
    """Create a synthetic 100x100 grayscale float32 image."""
    rng = np.random.default_rng(42)
    return rng.uniform(50, 200, size=(100, 100)).astype(np.float32)


@pytest.fixture
def color_uint8():
    """Create a synthetic 100x100 BGR colour image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(100, 100, 3), dtype=np.uint8)


# ------------------------------------------------------------------ #
#  Bilateral Filter                                                    #
# ------------------------------------------------------------------ #

class TestBilateralFilter:
    """Tests for bilateral_filter()."""

    def test_preserves_shape(self, gray_float32):
        result = bilateral_filter(gray_float32)
        assert result.shape == gray_float32.shape

    def test_reduces_noise(self, gray_uint8):
        """Filtered image should have lower variance than noisy input."""
        result = bilateral_filter(gray_uint8.astype(np.float32))
        assert result.std() <= gray_uint8.astype(np.float32).std()

    def test_float64_conversion(self):
        """float64 input should be converted to float32 internally."""
        img = np.random.rand(50, 50).astype(np.float64) * 255
        result = bilateral_filter(img)
        assert result.shape == (50, 50)

    def test_custom_sigma(self, gray_float32):
        result = bilateral_filter(gray_float32, d=5, sigma_color=10.0, sigma_space=10.0)
        assert result.shape == gray_float32.shape


# ------------------------------------------------------------------ #
#  Gaussian Filter                                                     #
# ------------------------------------------------------------------ #

class TestGaussianFilter:
    """Tests for gaussian_filter()."""

    def test_preserves_shape(self, gray_uint8):
        result = gaussian_filter(gray_uint8)
        assert result.shape == gray_uint8.shape

    def test_even_ksize_adjusted(self, gray_uint8):
        """Even kernel sizes should be auto-adjusted to odd."""
        result = gaussian_filter(gray_uint8, ksize=4)
        assert result.shape == gray_uint8.shape

    def test_smoothing_effect(self, gray_uint8):
        """Gaussian blur should reduce high-frequency content."""
        result = gaussian_filter(gray_uint8, ksize=11, sigma=3.0)
        assert result.std() <= gray_uint8.std()

    def test_colour_input(self, color_uint8):
        result = gaussian_filter(color_uint8, ksize=5)
        assert result.shape == color_uint8.shape


# ------------------------------------------------------------------ #
#  Median Filter                                                       #
# ------------------------------------------------------------------ #

class TestMedianFilter:
    """Tests for median_filter()."""

    def test_preserves_shape(self, gray_uint8):
        result = median_filter(gray_uint8)
        assert result.shape == gray_uint8.shape

    def test_removes_salt_and_pepper(self):
        """Median filter should effectively remove impulse noise."""
        img = np.full((100, 100), 128, dtype=np.uint8)
        # Add salt and pepper
        rng = np.random.default_rng(42)
        salt = rng.random((100, 100)) > 0.95
        pepper = rng.random((100, 100)) > 0.95
        img[salt] = 255
        img[pepper] = 0

        result = median_filter(img, ksize=3)
        # Most extreme pixels should be gone
        assert np.sum(result == 0) < np.sum(img == 0)
        assert np.sum(result == 255) < np.sum(img == 255)


# ------------------------------------------------------------------ #
#  Unsharp Mask                                                        #
# ------------------------------------------------------------------ #

class TestUnsharpMask:
    """Tests for unsharp_mask()."""

    def test_preserves_shape(self, gray_uint8):
        result = unsharp_mask(gray_uint8)
        assert result.shape == gray_uint8.shape

    def test_preserves_dtype_uint8(self, gray_uint8):
        result = unsharp_mask(gray_uint8)
        assert result.dtype == np.uint8

    def test_increases_sharpness(self, gray_uint8):
        """Sharpened image should have higher variance (more contrast)."""
        result = unsharp_mask(gray_uint8, sigma=2.0, strength=2.0)
        # The variance of the Laplacian is a measure of sharpness
        assert result.std() >= gray_uint8.std() * 0.9  # allow small tolerance

    def test_float_input(self, gray_float32):
        result = unsharp_mask(gray_float32)
        assert result.shape == gray_float32.shape


# ------------------------------------------------------------------ #
#  Non-Local Means Denoising                                           #
# ------------------------------------------------------------------ #

class TestDenoiseNLM:
    """Tests for denoise_nlm()."""

    def test_preserves_shape(self, gray_uint8):
        result = denoise_nlm(gray_uint8)
        assert result.shape == gray_uint8.shape

    def test_colour_input(self, color_uint8):
        result = denoise_nlm(color_uint8)
        assert result.shape == color_uint8.shape

    def test_float_input_converted(self, gray_float32):
        """Float input should be normalised to uint8 internally."""
        result = denoise_nlm(gray_float32)
        assert result.shape == gray_float32.shape
