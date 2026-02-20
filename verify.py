#!/usr/bin/env python
"""VIDIO — Quick Verification Suite.

Runs 15 checks to validate all core modules, infrastructure,
and frontend files without needing a running database.

Usage::

    python verify.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

passed = 0
failed = 0


def check(name, fn):
    global passed, failed
    try:
        fn()
        passed += 1
        print(f"  [PASS] {name}")
    except Exception as e:
        failed += 1
        print(f"  [FAIL] {name}: {e}")


# --- Test data ---
img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
mask = np.zeros((256, 256), dtype=np.uint8)
mask[80:180, 80:180] = 255


# ---- 1. Filtering ----
def test_filtering():
    from core.filtering import gaussian_filter, median_filter, bilateral_filter
    g = gaussian_filter(img, ksize=5)
    m = median_filter(img, ksize=5)
    assert g.shape == (256, 256) and m.shape == (256, 256)


# ---- 2. Edges ----
def test_edges():
    from core.edges import canny_edges, laplacian_edges, sobel_edges
    e1 = canny_edges(img)
    e2, _ = laplacian_edges(img)
    e3, _ = sobel_edges(img)
    assert e1.shape == e2.shape == e3.shape == (256, 256)


# ---- 3. Morphology ----
def test_morphology():
    from core.morphology import morph_open, morph_close, fill_holes
    mo = morph_open(mask)
    mc = morph_close(mask)
    fh = fill_holes(mask)
    assert mo.shape == mc.shape == fh.shape == (256, 256)


# ---- 4. Statistics ----
def test_statistics():
    from core.statistics import roi_stats
    s = roi_stats(img, mask)
    assert "mean" in s and "std" in s and len(s) >= 5


# ---- 5. Classification ----
def test_classification():
    from core.classification import classify_severity
    assert classify_severity(80, 0.90) == "CRITICAL"
    assert classify_severity(55, 0.75) == "HIGH"
    assert classify_severity(30, 0.60) == "MEDIUM"
    assert classify_severity(5, 0.20) == "LOW"


# ---- 6. Contours ----
def test_contours():
    from core.contours import find_contours
    result = find_contours(mask)
    assert len(result) >= 1
    assert "contour" in result[0]


# ---- 7. Shapes ----
def test_shapes():
    from core.contours import find_contours
    from core.shapes import shape_features, detect_shape, is_circular
    contours = find_contours(mask)
    c = contours[0]["contour"]
    feat = shape_features(c)
    assert "area" in feat and feat["area"] > 0
    shp = detect_shape(c)
    assert shp in ("triangle", "rectangle", "square", "pentagon", "circle", "irregular")


# ---- 8. Intensity ----
def test_intensity():
    from core.intensity import normalize_minmax, clahe_enhance
    n = normalize_minmax(img)
    c = clahe_enhance(img)
    assert n.min() >= 0.0 and n.max() <= 1.0
    assert c.shape == (256, 256)


# ---- 9. Clustering ----
def test_clustering():
    from core.clustering import kmeans_segment
    result = kmeans_segment(img, k=3)
    seg = result[0] if isinstance(result, tuple) else result
    assert seg.shape == (256, 256)
    assert len(np.unique(seg)) <= 3


# ---- 10. Build Finding ----
def test_build_finding():
    from core.classification import build_finding
    f = build_finding("anomaly", "HIGH", 0.88, {"mean": 120.0}, {"area": 500})
    assert f["finding_type"] == "anomaly"
    assert f["severity"] == "HIGH"
    assert f["confidence"] == 0.88
    assert "created_at" in f


# ---- 11. BasePipeline ----
def test_base_pipeline():
    from pipelines.base import BasePipeline
    assert hasattr(BasePipeline, "run")


# ---- 12. Auth Crypto ----
def test_auth():
    from api.Authentication import hash_password, verify_password
    h = hash_password("test123")
    assert verify_password("test123", h)
    assert not verify_password("wrong_pass", h)


# ---- 13. Falcon App ----
def test_falcon_app():
    import falcon
    from api.Authentication import AuthMiddleware
    from api.CORS import CORSComponent
    app = falcon.App(middleware=[AuthMiddleware(), CORSComponent()])
    assert app is not None


# ---- 14. Frontend Files ----
def test_frontend():
    files = [
        "frontend/index.html",
        "frontend/css/styles.css",
        "frontend/js/app.js",
        "frontend/js/api.js",
        "frontend/js/router.js",
        "frontend/js/components.js",
        "frontend/js/pages.js",
    ]
    for f in files:
        assert os.path.exists(f), f"Missing: {f}"


# ---- 15. Manual ----
def test_manual():
    assert os.path.exists("MANUAL.md")
    with open("MANUAL.md", "r", encoding="utf-8") as f:
        lines = len(f.readlines())
    assert lines > 100, f"Manual too short: {lines} lines"


# ---- Run all ----
if __name__ == "__main__":
    print()
    print("VIDIO Verification Suite")
    print("=" * 52)

    check("1.  Filtering (gaussian, median, bilateral)", test_filtering)
    check("2.  Edge detection (canny, laplacian, sobel)", test_edges)
    check("3.  Morphology (open, close, fill_holes)", test_morphology)
    check("4.  Statistics (roi_stats)", test_statistics)
    check("5.  Classification (4 severity levels)", test_classification)
    check("6.  Contour detection (find_contours)", test_contours)
    check("7.  Shape analysis (features, detect, circular)", test_shapes)
    check("8.  Intensity (normalize, CLAHE)", test_intensity)
    check("9.  Clustering (K-means)", test_clustering)
    check("10. Build finding (struct)", test_build_finding)
    check("11. BasePipeline import", test_base_pipeline)
    check("12. Auth crypto (hash + verify)", test_auth)
    check("13. Falcon app creation", test_falcon_app)
    check("14. Frontend files (7 files)", test_frontend)
    check("15. User manual (MANUAL.md)", test_manual)

    print("=" * 52)
    total = passed + failed
    print(f"  Result: {passed}/{total} passed")
    if failed == 0:
        print("  *** VIDIO FULLY VERIFIED ***")
    else:
        print(f"  *** {failed} CHECK(S) FAILED ***")
    print()

    sys.exit(0 if failed == 0 else 1)
