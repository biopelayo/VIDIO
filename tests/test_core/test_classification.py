"""
Tests for :mod:`core.classification`
======================================

Unit tests for severity classification and finding builder.
"""

import numpy as np
import pytest

from core.classification import classify_severity, build_finding


class TestClassifySeverity:
    """Tests for classify_severity()."""

    def test_critical(self):
        assert classify_severity(gradient=100, confidence=0.9) == "CRITICAL"

    def test_high(self):
        assert classify_severity(gradient=60, confidence=0.75) == "HIGH"

    def test_medium(self):
        assert classify_severity(gradient=30, confidence=0.55) == "MEDIUM"

    def test_low(self):
        assert classify_severity(gradient=5, confidence=0.1) == "LOW"

    def test_boundary_critical(self):
        """Exactly at CRITICAL threshold."""
        assert classify_severity(gradient=80.0, confidence=0.85) == "CRITICAL"

    def test_high_gradient_low_confidence(self):
        """High gradient but low confidence should not be CRITICAL."""
        result = classify_severity(gradient=100, confidence=0.3)
        assert result != "CRITICAL"

    def test_custom_thresholds(self):
        custom = [
            {"min_gradient": 10, "min_confidence": 0.5, "severity": "DANGER"},
            {"min_gradient": 0, "min_confidence": 0.0, "severity": "SAFE"},
        ]
        assert classify_severity(15, 0.6, thresholds=custom) == "DANGER"
        assert classify_severity(5, 0.3, thresholds=custom) == "SAFE"


class TestBuildFinding:
    """Tests for build_finding()."""

    def test_required_keys(self):
        finding = build_finding(
            finding_type="LESION",
            severity="HIGH",
            confidence=0.85,
            statistics={"mean": 150, "std": 20},
            geometric_properties={"area": 500},
        )
        assert finding['finding_type'] == "LESION"
        assert finding['severity'] == "HIGH"
        assert finding['confidence'] == 0.85
        assert 'created_at' in finding
        assert finding['version'] == "1.0"

    def test_sanitizes_numpy(self):
        finding = build_finding(
            finding_type="TEST",
            severity="LOW",
            confidence=np.float32(0.5),
            statistics={"mean": np.float64(100), "values": np.array([1, 2, 3])},
            geometric_properties={"area": np.int64(500)},
        )
        assert isinstance(finding['confidence'], float)
        assert isinstance(finding['statistics']['mean'], float)
        assert isinstance(finding['statistics']['values'], list)
        assert isinstance(finding['geometric_properties']['area'], int)

    def test_optional_location(self):
        finding = build_finding(
            finding_type="TEST",
            severity="LOW",
            confidence=0.5,
            statistics={},
            geometric_properties={},
            location={"x": 10, "y": 20},
        )
        assert finding['location'] == {"x": 10, "y": 20}

    def test_none_location(self):
        finding = build_finding(
            finding_type="TEST",
            severity="LOW",
            confidence=0.5,
            statistics={},
            geometric_properties={},
        )
        assert finding['location'] is None

    def test_classification_field(self):
        finding = build_finding(
            finding_type="TUMOR",
            severity="CRITICAL",
            confidence=0.95,
            statistics={},
            geometric_properties={},
            classification="malignant",
        )
        assert finding['classification'] == "malignant"
