"""
Tests for :mod:`pipelines.radiology`
======================================

Integration tests for the radiology CT/MRI pipeline using synthetic
volumetric data.
"""

import numpy as np
import pytest

from pipelines.radiology.preprocessing import preprocess_ct, preprocess_mri
from pipelines.radiology.segmentation import (
    segment_brain_regions, segment_lesions_3d, measure_volume,
)
from pipelines.radiology.analysis import analyze_brain_atrophy, analyze_lesion


@pytest.fixture
def synthetic_ct_slice():
    """A synthetic CT slice in Hounsfield Units.

    - Air: -1000 HU (borders)
    - Soft tissue: 20-80 HU (center circle)
    - Bone: 500-1000 HU (outer ring)
    """
    rng = np.random.default_rng(42)
    img = np.full((256, 256), -1000.0, dtype=np.float64)  # air

    # Body outline
    y, x = np.ogrid[-128:128, -128:128]
    body_mask = x**2 + y**2 < 100**2
    img[body_mask] = 40 + rng.normal(0, 10, np.sum(body_mask))

    # Bone ring
    bone_mask = (x**2 + y**2 < 110**2) & (x**2 + y**2 > 100**2)
    img[bone_mask] = 700 + rng.normal(0, 50, np.sum(bone_mask))

    return img


@pytest.fixture
def synthetic_volume():
    """A synthetic 3D volume (32 slices x 128 x 128)."""
    rng = np.random.default_rng(42)
    volume = rng.normal(100, 20, (32, 128, 128)).astype(np.float32)

    # Add a bright lesion in slices 12-18 at center
    volume[12:18, 50:70, 50:70] = 200 + rng.normal(0, 10, (6, 20, 20))

    return volume


@pytest.fixture
def brain_mask():
    """Binary brain mask for the synthetic volume."""
    mask = np.zeros((32, 128, 128), dtype=np.uint8)
    for z in range(32):
        y, x = np.ogrid[-64:64, -64:64]
        brain = x**2 + y**2 < 50**2
        mask[z][brain] = 1
    return mask


class TestPreprocessing:
    def test_preprocess_ct(self, synthetic_ct_slice):
        result = preprocess_ct(synthetic_ct_slice, window_center=40, window_width=400)
        assert result.dtype == np.uint8
        assert result.min() == 0
        assert result.max() == 255

    def test_preprocess_mri(self, synthetic_volume):
        result = preprocess_mri(synthetic_volume[0])
        assert result.shape == (128, 128)
        # Should be normalized to 0-255 range
        assert result.max() <= 255
        assert result.min() >= 0


class TestSegmentation:
    def test_segment_brain_regions(self, synthetic_ct_slice):
        ct_uint8 = preprocess_ct(synthetic_ct_slice)
        segmented, labels, centers = segment_brain_regions(ct_uint8, k=4)
        assert segmented.shape == ct_uint8.shape
        assert labels.shape == ct_uint8.shape
        assert centers.shape[0] == 4

    def test_segment_lesions_3d(self, synthetic_volume):
        mask = segment_lesions_3d(synthetic_volume, threshold_factor=2.0)
        assert mask.shape == synthetic_volume.shape
        assert mask.dtype == np.uint8
        # Should detect the bright lesion
        assert np.any(mask[12:18, 50:70, 50:70] > 0)

    def test_measure_volume(self):
        mask = np.zeros((10, 50, 50), dtype=np.uint8)
        mask[2:8, 10:40, 10:40] = 1  # 6*30*30 = 5400 voxels
        result = measure_volume(mask, voxel_spacing=(1.0, 1.0, 1.0))
        assert result['n_voxels'] == 5400
        assert abs(result['volume_mm3'] - 5400.0) < 1e-6
        assert abs(result['volume_cm3'] - 5.4) < 1e-6

    def test_measure_volume_with_spacing(self):
        mask = np.zeros((10, 50, 50), dtype=np.uint8)
        mask[0:2, 0:10, 0:10] = 1  # 200 voxels
        result = measure_volume(mask, voxel_spacing=(2.0, 0.5, 0.5))
        assert result['n_voxels'] == 200
        assert abs(result['volume_mm3'] - 200 * 0.5) < 1e-6


class TestAnalysis:
    def test_analyze_brain_atrophy(self, synthetic_volume, brain_mask):
        result = analyze_brain_atrophy(synthetic_volume, brain_mask=brain_mask)
        assert 'brain_volume_voxels' in result
        assert 'brain_parenchymal_fraction' in result
        assert 0 < result['brain_parenchymal_fraction'] < 1

    def test_analyze_brain_atrophy_no_mask(self, synthetic_volume):
        result = analyze_brain_atrophy(synthetic_volume)
        assert 'brain_parenchymal_fraction' in result

    def test_analyze_lesion(self, synthetic_volume):
        lesion_mask = segment_lesions_3d(synthetic_volume, threshold_factor=2.0)
        result = analyze_lesion(synthetic_volume, lesion_mask)
        assert 'n_voxels' in result
        assert 'volume_mm3' in result
        if result['n_voxels'] > 0:
            assert 'mean_intensity' in result
