"""
Tests for :mod:`core.morphology`
=================================

Unit tests for morphological operations, hole filling, and small
object removal.
"""

import numpy as np
import cv2
import pytest

from core.morphology import (
    morph_open, morph_close, morph_dilate, morph_erode,
    fill_holes, remove_small_objects,
)


@pytest.fixture
def noisy_mask():
    """Binary mask with a large object and small noise spots."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 50, 255, -1)  # main object
    # Small noise dots
    mask[10, 10] = 255
    mask[20, 180] = 255
    mask[190, 30] = 255
    return mask


@pytest.fixture
def mask_with_hole():
    """Binary mask with a filled circle containing a hole."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 50, 255, -1)
    cv2.circle(mask, (100, 100), 15, 0, -1)  # hole
    return mask


class TestMorphOpen:
    def test_preserves_shape(self, noisy_mask):
        result = morph_open(noisy_mask)
        assert result.shape == noisy_mask.shape

    def test_removes_small_noise(self, noisy_mask):
        result = morph_open(noisy_mask, kernel_size=5, iterations=2)
        # Noise pixels should be removed
        assert result[10, 10] == 0
        # Main object should partially survive
        assert np.sum(result > 0) > 100


class TestMorphClose:
    def test_preserves_shape(self, noisy_mask):
        result = morph_close(noisy_mask)
        assert result.shape == noisy_mask.shape

    def test_fills_small_gaps(self, mask_with_hole):
        result = morph_close(mask_with_hole, kernel_size=31, iterations=3)
        # Hole should be filled or reduced
        assert np.sum(result > 0) >= np.sum(mask_with_hole > 0)


class TestMorphDilate:
    def test_expands_foreground(self, noisy_mask):
        result = morph_dilate(noisy_mask, kernel_size=3)
        assert np.sum(result > 0) >= np.sum(noisy_mask > 0)


class TestMorphErode:
    def test_shrinks_foreground(self, noisy_mask):
        result = morph_erode(noisy_mask, kernel_size=3)
        assert np.sum(result > 0) <= np.sum(noisy_mask > 0)


class TestFillHoles:
    def test_fills_inner_hole(self, mask_with_hole):
        result = fill_holes(mask_with_hole)
        # After filling, the center should be white
        assert result[100, 100] == 255

    def test_output_dtype(self, mask_with_hole):
        result = fill_holes(mask_with_hole)
        assert result.dtype == np.uint8

    def test_bool_input(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[20:80, 20:80] = True
        mask[40:60, 40:60] = False  # hole
        result = fill_holes(mask.astype(np.uint8) * 255)
        assert result[50, 50] == 255


class TestRemoveSmallObjects:
    def test_removes_small_components(self, noisy_mask):
        result = remove_small_objects(noisy_mask, min_area=100)
        # Small noise should be gone
        assert result[10, 10] == 0
        assert result[20, 180] == 0
        # Main circle (area ~7854) should remain
        assert np.sum(result > 0) > 5000

    def test_empty_mask(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        result = remove_small_objects(mask)
        assert np.sum(result) == 0

    def test_preserves_large_objects(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 60, 255, -1)
        result = remove_small_objects(mask, min_area=100)
        assert np.sum(result > 0) > 10000
