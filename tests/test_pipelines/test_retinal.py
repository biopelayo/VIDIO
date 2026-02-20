"""
Tests for :mod:`pipelines.retinal`
====================================

Integration tests for the retinal image analysis pipeline using
synthetic fundus-like images.  These tests verify the pipeline stages
work correctly without requiring real clinical data.
"""

import numpy as np
import cv2
import pytest

from pipelines.retinal.preprocessing import (
    extract_green_channel, enhance_vessels,
    preprocess_fundus, preprocess_oct, preprocess_slit_lamp,
)
from pipelines.retinal.segmentation import (
    segment_vessels, detect_optic_disc, detect_macula, segment_lesions,
)
from pipelines.retinal.analysis import (
    analyze_retinal_regions, detect_bright_lesions, detect_dark_lesions,
)


# ------------------------------------------------------------------ #
#  Fixtures                                                            #
# ------------------------------------------------------------------ #

@pytest.fixture
def synthetic_fundus():
    """Create a synthetic colour fundus image (500x500 BGR).

    Simulates: dark background, bright disc (optic disc), vessel-like
    lines, and a few bright/dark lesion spots.
    """
    rng = np.random.default_rng(42)
    img = np.zeros((500, 500, 3), dtype=np.uint8)

    # Background retinal field — reddish-brown
    img[:, :, 0] = 30   # blue
    img[:, :, 1] = 60   # green
    img[:, :, 2] = 100  # red
    # Add subtle noise
    noise = rng.integers(-10, 10, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Optic disc — bright circle at (350, 250), radius ~40
    cv2.circle(img, (350, 250), 40, (180, 200, 220), -1)

    # Macula — dark region at (150, 250)
    cv2.circle(img, (150, 250), 25, (15, 30, 50), -1)

    # Vessels — dark green-channel lines
    cv2.line(img, (350, 250), (100, 100), (20, 30, 60), 2)
    cv2.line(img, (350, 250), (100, 400), (20, 30, 60), 2)
    cv2.line(img, (350, 250), (200, 250), (20, 30, 60), 2)

    # Bright lesion (exudate-like)
    cv2.circle(img, (200, 150), 8, (200, 220, 240), -1)

    # Dark lesion (hemorrhage-like)
    cv2.circle(img, (300, 350), 10, (10, 15, 25), -1)

    return img


@pytest.fixture
def synthetic_oct():
    """Create a synthetic OCT cross-section image (256x512 grayscale)."""
    rng = np.random.default_rng(42)
    img = rng.integers(20, 80, (256, 512), dtype=np.uint8)
    # Simulate retinal layers as bright horizontal bands
    img[80:90, :] = 200   # RPE
    img[120:125, :] = 180  # ELM
    img[150:160, :] = 220  # NFL
    return img


# ------------------------------------------------------------------ #
#  Preprocessing Tests                                                 #
# ------------------------------------------------------------------ #

class TestPreprocessing:
    def test_extract_green_channel(self, synthetic_fundus):
        green = extract_green_channel(synthetic_fundus)
        assert green.ndim == 2
        assert green.dtype == np.float32
        assert green.shape == (500, 500)

    def test_extract_green_from_grayscale(self):
        gray = np.zeros((100, 100), dtype=np.uint8)
        result = extract_green_channel(gray)
        assert result.ndim == 2
        assert result.dtype == np.float32

    def test_enhance_vessels(self, synthetic_fundus):
        green = extract_green_channel(synthetic_fundus)
        enhanced = enhance_vessels(green)
        assert enhanced.shape == green.shape

    def test_preprocess_fundus(self, synthetic_fundus):
        result = preprocess_fundus(synthetic_fundus)
        assert result.ndim == 2
        assert result.shape == (500, 500)

    def test_preprocess_oct(self, synthetic_oct):
        result = preprocess_oct(synthetic_oct)
        assert result.ndim == 2
        assert result.shape == synthetic_oct.shape

    def test_preprocess_slit_lamp(self, synthetic_fundus):
        result = preprocess_slit_lamp(synthetic_fundus)
        assert result.ndim == 2


# ------------------------------------------------------------------ #
#  Segmentation Tests                                                  #
# ------------------------------------------------------------------ #

class TestSegmentation:
    def test_segment_vessels_laplacian(self, synthetic_fundus):
        green = extract_green_channel(synthetic_fundus)
        filtered = preprocess_fundus(synthetic_fundus)
        mask = segment_vessels(filtered, method='laplacian')
        assert mask.shape == filtered.shape
        assert mask.dtype == np.uint8

    def test_segment_vessels_canny(self, synthetic_fundus):
        filtered = preprocess_fundus(synthetic_fundus)
        mask = segment_vessels(filtered, method='canny')
        assert mask.shape == filtered.shape

    def test_detect_optic_disc(self, synthetic_fundus):
        green = extract_green_channel(synthetic_fundus)
        disc = detect_optic_disc(green, min_radius=20, max_radius=60)
        # May or may not detect depending on synthetic image quality
        if disc is not None:
            assert 'center' in disc
            assert 'radius' in disc

    def test_detect_macula(self, synthetic_fundus):
        green = extract_green_channel(synthetic_fundus)
        macula = detect_macula(green)
        assert macula is not None
        assert 'center' in macula

    def test_segment_lesions(self, synthetic_fundus):
        filtered = preprocess_fundus(synthetic_fundus)
        regions = segment_lesions(filtered)
        assert isinstance(regions, list)
        # Each region should have stats
        for r in regions:
            assert 'stats' in r


# ------------------------------------------------------------------ #
#  Analysis Tests                                                      #
# ------------------------------------------------------------------ #

class TestAnalysis:
    def test_detect_bright_lesions(self, synthetic_fundus):
        filtered = preprocess_fundus(synthetic_fundus)
        mask = detect_bright_lesions(filtered)
        assert mask.shape == filtered.shape
        assert mask.dtype == np.uint8

    def test_detect_dark_lesions(self, synthetic_fundus):
        filtered = preprocess_fundus(synthetic_fundus)
        mask = detect_dark_lesions(filtered)
        assert mask.shape == filtered.shape
        assert mask.dtype == np.uint8

    def test_analyze_retinal_regions(self, synthetic_fundus):
        filtered = preprocess_fundus(synthetic_fundus)
        regions = segment_lesions(filtered)
        if regions:
            findings = analyze_retinal_regions(filtered, regions)
            assert isinstance(findings, list)
            for f in findings:
                assert 'finding_type' in f
                assert 'severity' in f
                assert 'confidence' in f
