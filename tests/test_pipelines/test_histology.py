"""
Tests for :mod:`pipelines.histology`
======================================

Integration tests for the histology WSI analysis pipeline using
synthetic tissue-like images.
"""

import numpy as np
import cv2
import pytest

from pipelines.histology.preprocessing import detect_tissue, get_tissue_ratio
from pipelines.histology.tiling import extract_tiles
from pipelines.histology.segmentation import segment_tumor_regions
from pipelines.histology.analysis import classify_tiles
from core.statistics import roi_stats


@pytest.fixture
def synthetic_tissue():
    """Create a synthetic H&E-like image (512x512 BGR).

    Light pink background (slide glass) with darker stained tissue
    regions and a very dark "tumour" spot.
    """
    rng = np.random.default_rng(42)
    img = np.full((512, 512, 3), 240, dtype=np.uint8)  # white-ish background

    # Tissue region — pinkish (H&E staining)
    cv2.circle(img, (256, 256), 180, (180, 160, 200), -1)

    # Add noise within tissue
    mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(mask, (256, 256), 180, 255, -1)
    noise = rng.integers(-15, 15, img.shape, dtype=np.int16)
    img_noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img = np.where(mask[:, :, None] > 0, img_noisy, img)

    # Dark tumour region
    cv2.circle(img, (200, 200), 40, (80, 60, 100), -1)

    # Stroma — slightly lighter area
    cv2.circle(img, (300, 300), 30, (200, 185, 210), -1)

    return img


@pytest.fixture
def background_image():
    """A completely white image (background only)."""
    return np.full((256, 256, 3), 245, dtype=np.uint8)


class TestPreprocessing:
    def test_detect_tissue(self, synthetic_tissue):
        mask = detect_tissue(synthetic_tissue)
        assert mask.shape == (512, 512)
        assert mask.dtype == np.uint8
        assert np.any(mask > 0)

    def test_get_tissue_ratio(self, synthetic_tissue):
        ratio = get_tissue_ratio(synthetic_tissue)
        assert 0 < ratio < 1

    def test_background_low_tissue(self, background_image):
        ratio = get_tissue_ratio(background_image, threshold=220)
        # Almost all pixels > 220, so tissue ratio should be very low
        assert ratio < 0.1


class TestTiling:
    def test_extract_tiles(self, synthetic_tissue):
        tiles = extract_tiles(synthetic_tissue, tile_size=128, tissue_threshold=0.3)
        assert len(tiles) > 0
        for t in tiles:
            assert t['tile'].shape == (128, 128, 3)
            assert 'x' in t
            assert 'y' in t
            assert t['tissue_ratio'] >= 0.3

    def test_tile_count_matches_grid(self):
        """For a uniform dark image, all tiles should have tissue."""
        img = np.full((256, 256, 3), 100, dtype=np.uint8)
        tiles = extract_tiles(img, tile_size=128, tissue_threshold=0.5)
        # 256/128 = 2 tiles per axis => 4 tiles total
        assert len(tiles) == 4

    def test_overlap_produces_more_tiles(self, synthetic_tissue):
        tiles_no_overlap = extract_tiles(synthetic_tissue, tile_size=128, overlap=0)
        tiles_with_overlap = extract_tiles(synthetic_tissue, tile_size=128, overlap=64)
        assert len(tiles_with_overlap) >= len(tiles_no_overlap)

    def test_background_filtered(self, background_image):
        tiles = extract_tiles(background_image, tile_size=128, tissue_threshold=0.5)
        assert len(tiles) == 0


class TestSegmentation:
    def test_segment_tumor_regions(self, synthetic_tissue):
        regions = segment_tumor_regions(synthetic_tissue)
        assert isinstance(regions, list)
        for r in regions:
            assert 'stats' in r
            assert 'contour' in r


class TestAnalysis:
    def test_classify_tiles(self, synthetic_tissue):
        tiles = extract_tiles(synthetic_tissue, tile_size=128, tissue_threshold=0.3)
        stats_list = []
        for t in tiles:
            gray = np.mean(t['tile'], axis=2).astype(np.float32)
            stats_list.append(roi_stats(gray))

        classifications = classify_tiles(tiles, stats_list)
        assert len(classifications) == len(tiles)
        for c in classifications:
            assert c['label'] in ('tumor', 'normal', 'stroma', 'background')
            assert 'z_score' in c
