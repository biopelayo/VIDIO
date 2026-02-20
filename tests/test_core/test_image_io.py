"""
Tests for :mod:`core.image_io`
================================

Unit tests for the universal image reader with auto-format detection.
Tests standard image reading and format detection without requiring
medical imaging libraries.
"""

import os
import tempfile

import numpy as np
import cv2
import pytest

from core.image_io import (
    read_standard, read_image,
    STANDARD_EXTENSIONS, DICOM_EXTENSIONS, NIFTI_EXTENSIONS,
    WSI_EXTENSIONS, H5AD_EXTENSIONS,
)


@pytest.fixture
def tmp_png(tmp_path):
    """Create a temporary PNG image file."""
    img = np.random.default_rng(42).integers(0, 256, (100, 100, 3), dtype=np.uint8)
    path = str(tmp_path / "test.png")
    cv2.imwrite(path, img)
    return path, img


@pytest.fixture
def tmp_jpg(tmp_path):
    """Create a temporary JPEG image file."""
    img = np.random.default_rng(42).integers(0, 256, (100, 100, 3), dtype=np.uint8)
    path = str(tmp_path / "test.jpg")
    cv2.imwrite(path, img)
    return path, img


@pytest.fixture
def tmp_tiff(tmp_path):
    """Create a temporary TIFF image file."""
    img = np.random.default_rng(42).integers(0, 256, (100, 100), dtype=np.uint8)
    path = str(tmp_path / "test.tif")
    cv2.imwrite(path, img)
    return path, img


class TestReadStandard:
    def test_reads_png(self, tmp_png):
        path, original = tmp_png
        result = read_standard(path)
        assert result.shape == original.shape

    def test_reads_jpg(self, tmp_jpg):
        path, original = tmp_jpg
        result = read_standard(path)
        assert result.shape == original.shape

    def test_reads_tiff(self, tmp_tiff):
        path, original = tmp_tiff
        result = read_standard(path)
        assert result.shape == original.shape

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_standard("/nonexistent/path/image.png")


class TestReadImage:
    """Tests for the auto-detecting read_image() dispatcher."""

    def test_dispatches_png(self, tmp_png):
        path, _ = tmp_png
        result = read_image(path)
        assert isinstance(result, np.ndarray)

    def test_dispatches_tiff(self, tmp_tiff):
        path, _ = tmp_tiff
        result = read_image(path)
        assert isinstance(result, np.ndarray)

    def test_unknown_extension_fallback(self, tmp_path):
        """Unknown extension should fall back to standard reader."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        path = str(tmp_path / "test.bmp")
        cv2.imwrite(path, img)
        result = read_image(path)
        assert isinstance(result, np.ndarray)


class TestExtensionSets:
    """Verify the extension sets are correctly defined."""

    def test_standard_extensions(self):
        assert ".png" in STANDARD_EXTENSIONS
        assert ".jpg" in STANDARD_EXTENSIONS
        assert ".tif" in STANDARD_EXTENSIONS

    def test_dicom_extensions(self):
        assert ".dcm" in DICOM_EXTENSIONS

    def test_nifti_extensions(self):
        assert ".nii" in NIFTI_EXTENSIONS
        assert ".nii.gz" in NIFTI_EXTENSIONS

    def test_wsi_extensions(self):
        assert ".svs" in WSI_EXTENSIONS

    def test_h5ad_extensions(self):
        assert ".h5ad" in H5AD_EXTENSIONS
