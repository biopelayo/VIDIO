"""
Tests for :mod:`core.shapes`
==============================

Unit tests for shape detection, classification, and feature extraction.
"""

import numpy as np
import cv2
import pytest

from core.shapes import detect_shape, shape_features, is_circular, is_irregular


@pytest.fixture
def circle_contour():
    """Generate a contour from a filled circle."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.circle(mask, (100, 100), 50, 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


@pytest.fixture
def square_contour():
    """Generate a contour from a filled square."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(mask, (50, 50), (150, 150), 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


@pytest.fixture
def rectangle_contour():
    """Generate a contour from a filled rectangle (not square)."""
    mask = np.zeros((200, 300), dtype=np.uint8)
    cv2.rectangle(mask, (20, 50), (280, 150), 255, -1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


@pytest.fixture
def triangle_contour():
    """Generate a contour from a filled triangle."""
    mask = np.zeros((200, 200), dtype=np.uint8)
    pts = np.array([[100, 20], [30, 180], [170, 180]], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]


class TestDetectShape:
    def test_circle(self, circle_contour):
        assert detect_shape(circle_contour) == "circle"

    def test_square(self, square_contour):
        assert detect_shape(square_contour) == "square"

    def test_rectangle(self, rectangle_contour):
        assert detect_shape(rectangle_contour) == "rectangle"

    def test_triangle(self, triangle_contour):
        assert detect_shape(triangle_contour) == "triangle"


class TestShapeFeatures:
    def test_returns_all_keys(self, circle_contour):
        features = shape_features(circle_contour)
        expected = {'area', 'perimeter', 'circularity', 'solidity', 'convexity',
                    'elongation', 'aspect_ratio', 'extent', 'equivalent_diameter',
                    'num_vertices', 'bounding_rect', 'min_enclosing_circle_radius',
                    'centroid', 'angle', 'hu_moments'}
        assert expected.issubset(features.keys())

    def test_circle_high_circularity(self, circle_contour):
        features = shape_features(circle_contour)
        assert features['circularity'] > 0.85

    def test_hu_moments_length(self, circle_contour):
        features = shape_features(circle_contour)
        assert len(features['hu_moments']) == 7

    def test_bounding_rect_format(self, square_contour):
        features = shape_features(square_contour)
        assert len(features['bounding_rect']) == 4


class TestIsCircular:
    def test_circle_is_circular(self, circle_contour):
        assert is_circular(circle_contour) is True

    def test_rectangle_not_circular(self, rectangle_contour):
        assert is_circular(rectangle_contour) is False


class TestIsIrregular:
    def test_circle_not_irregular(self, circle_contour):
        assert is_irregular(circle_contour) is False

    def test_star_is_irregular(self):
        """Create a concave star shape that should be irregular."""
        mask = np.zeros((200, 200), dtype=np.uint8)
        # Star-like shape
        pts = np.array([
            [100, 10], [120, 70], [180, 70], [135, 110],
            [155, 170], [100, 135], [45, 170], [65, 110],
            [20, 70], [80, 70]
        ], dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        assert is_irregular(contours[0]) is True
