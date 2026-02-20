"""
Tests for :mod:`core.contours`
===============================

Unit tests for contour detection, geometric statistics computation,
and contour drawing overlay functionality.
"""

import numpy as np
import cv2
import pytest

from core.contours import contour_stats, find_contours, draw_contours


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

@pytest.fixture
def circle_mask():
    """Create a binary mask with a single filled circle (radius 40)."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 40, 255, -1)
    return mask


@pytest.fixture
def rectangle_mask():
    """Create a binary mask with a single filled rectangle."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(mask, (30, 50), (170, 150), 255, -1)
    return mask


@pytest.fixture
def multi_object_mask():
    """Create a binary mask with three separate objects."""
    mask = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(mask, (70, 70), 30, 255, -1)        # small circle
    cv2.circle(mask, (200, 100), 50, 255, -1)       # large circle
    cv2.rectangle(mask, (100, 200), (200, 280), 255, -1)  # rectangle
    return mask


@pytest.fixture
def sample_contour():
    """Extract a single contour from a circle mask for stats testing."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 40, 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


# ------------------------------------------------------------------ #
#  contour_stats                                                       #
# ------------------------------------------------------------------ #

class TestContourStats:
    """Tests for contour_stats()."""

    def test_returns_required_keys(self, sample_contour):
        stats = contour_stats(sample_contour)
        expected_keys = {'area', 'perimeter', 'solidity', 'elongation',
                         'convexity', 'circularity', 'centroid', 'angle'}
        assert expected_keys.issubset(stats.keys())

    def test_circle_has_high_circularity(self, sample_contour):
        stats = contour_stats(sample_contour)
        assert stats['circularity'] > 0.85

    def test_circle_has_high_solidity(self, sample_contour):
        stats = contour_stats(sample_contour)
        assert stats['solidity'] > 0.95

    def test_circle_centroid_near_center(self, sample_contour):
        stats = contour_stats(sample_contour)
        cx, cy = stats['centroid']
        assert abs(cx - 100) < 5
        assert abs(cy - 100) < 5

    def test_area_positive(self, sample_contour):
        stats = contour_stats(sample_contour)
        assert stats['area'] > 0

    def test_perimeter_positive(self, sample_contour):
        stats = contour_stats(sample_contour)
        assert stats['perimeter'] > 0


# ------------------------------------------------------------------ #
#  find_contours                                                       #
# ------------------------------------------------------------------ #

class TestFindContours:
    """Tests for find_contours()."""

    def test_finds_single_circle(self, circle_mask):
        results = find_contours(circle_mask, min_area=100)
        assert len(results) == 1

    def test_finds_multiple_objects(self, multi_object_mask):
        results = find_contours(multi_object_mask, min_area=50)
        assert len(results) == 3

    def test_sorted_by_area_descending(self, multi_object_mask):
        results = find_contours(multi_object_mask, min_area=50)
        areas = [r['stats']['area'] for r in results]
        assert areas == sorted(areas, reverse=True)

    def test_min_area_filters(self, multi_object_mask):
        results = find_contours(multi_object_mask, min_area=5000)
        # Only the large circle and rectangle should survive
        assert len(results) <= 2

    def test_max_area_filters(self, multi_object_mask):
        results = find_contours(multi_object_mask, min_area=50, max_area=3000)
        # Only the small circle should survive
        assert len(results) >= 1
        for r in results:
            assert r['stats']['area'] <= 3000

    def test_result_structure(self, circle_mask):
        results = find_contours(circle_mask, min_area=100)
        r = results[0]
        assert 'contour' in r
        assert 'bounding_rect' in r
        assert 'hull' in r
        assert 'stats' in r

    def test_empty_mask_returns_empty(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        results = find_contours(mask)
        assert results == []

    def test_non_uint8_conversion(self):
        """Boolean or float masks should be converted to uint8."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[30:70, 30:70] = True
        results = find_contours(mask, min_area=10)
        assert len(results) >= 1


# ------------------------------------------------------------------ #
#  draw_contours                                                       #
# ------------------------------------------------------------------ #

class TestDrawContours:
    """Tests for draw_contours()."""

    def test_returns_bgr_from_grayscale(self, circle_mask):
        contours = find_contours(circle_mask, min_area=100)
        canvas = draw_contours(circle_mask, contours)
        assert canvas.ndim == 3
        assert canvas.shape[2] == 3

    def test_does_not_modify_original(self, circle_mask):
        original = circle_mask.copy()
        contours = find_contours(circle_mask, min_area=100)
        draw_contours(circle_mask, contours)
        np.testing.assert_array_equal(circle_mask, original)

    def test_draws_on_color_image(self):
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 40, 255, -1)
        contours = find_contours(mask, min_area=100)
        canvas = draw_contours(img, contours, color=(0, 255, 0))
        # Green channel should have non-zero pixels from drawing
        assert np.any(canvas[:, :, 1] > 0)
