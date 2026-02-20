"""
Tests for :mod:`core.edges`
============================

Unit tests for Laplacian, Canny, and Sobel edge detection.
"""

import numpy as np
import cv2
import pytest

from core.edges import laplacian_edges, canny_edges, sobel_edges


@pytest.fixture
def sharp_edge_image():
    """Create a 100x100 image with a sharp vertical edge at column 50."""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[:, 50:] = 255
    return img


@pytest.fixture
def smooth_gradient():
    """Create a smooth horizontal gradient image."""
    return np.tile(np.linspace(0, 255, 100, dtype=np.uint8), (100, 1))


@pytest.fixture
def uniform_image():
    """Create a completely uniform image (no edges)."""
    return np.full((100, 100), 128, dtype=np.uint8)


class TestLaplacianEdges:
    def test_returns_tuple(self, sharp_edge_image):
        result = laplacian_edges(sharp_edge_image)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_edge_map_shape(self, sharp_edge_image):
        edge_map, stats = laplacian_edges(sharp_edge_image)
        assert edge_map.shape == (100, 100)

    def test_stats_keys(self, sharp_edge_image):
        _, stats = laplacian_edges(sharp_edge_image)
        assert 'avg' in stats
        assert 'std' in stats
        assert 'median' in stats
        assert 'min' in stats
        assert 'max' in stats

    def test_detects_sharp_edge(self, sharp_edge_image):
        edge_map, stats = laplacian_edges(sharp_edge_image)
        # The edge at column 50 should produce high values
        assert stats['max'] > 0

    def test_uniform_has_low_response(self, uniform_image):
        edge_map, stats = laplacian_edges(uniform_image)
        assert stats['max'] < 1.0

    def test_colour_input(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, 50:] = 255
        edge_map, stats = laplacian_edges(img)
        assert edge_map.shape == (100, 100)


class TestCannyEdges:
    def test_output_shape(self, sharp_edge_image):
        edges = canny_edges(sharp_edge_image)
        assert edges.shape == (100, 100)

    def test_output_binary(self, sharp_edge_image):
        edges = canny_edges(sharp_edge_image)
        unique_vals = np.unique(edges)
        assert all(v in [0, 255] for v in unique_vals)

    def test_detects_edge(self, sharp_edge_image):
        edges = canny_edges(sharp_edge_image, low=50, high=150)
        assert np.any(edges > 0)

    def test_uniform_no_edges(self, uniform_image):
        edges = canny_edges(uniform_image)
        assert np.sum(edges) == 0


class TestSobelEdges:
    def test_returns_tuple(self, sharp_edge_image):
        result = sobel_edges(sharp_edge_image)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_magnitude_shape(self, sharp_edge_image):
        magnitude, direction = sobel_edges(sharp_edge_image)
        assert magnitude.shape == (100, 100)
        assert direction.shape == (100, 100)

    def test_detects_vertical_edge(self, sharp_edge_image):
        magnitude, _ = sobel_edges(sharp_edge_image)
        # Column 50 should have high magnitude
        assert magnitude[:, 49:51].max() > 0
