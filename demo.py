#!/usr/bin/env python
"""
VIDIO Standalone Demo
=====================

This script demonstrates the VIDIO image analysis pipeline **without**
needing PostgreSQL or the full server. It:

1. Generates synthetic biomedical images (retinal fundus, histology H&E,
   CT slice, brain MRI)
2. Runs the core analysis modules on each image
3. Saves annotated result images + JSON reports to ``demo_output/``
4. Opens the results in your default image viewer

Usage::

    python demo.py                    # run all 4 demos
    python demo.py retinal            # run only retinal demo
    python demo.py histology          # run only histology demo
    python demo.py radiology          # run only radiology demo
    python demo.py --no-open          # don't auto-open images

Output::

    demo_output/
      retinal_input.png
      retinal_segmented.png
      retinal_findings.json
      histology_input.png
      histology_segmented.png
      histology_findings.json
      radiology_input.png
      radiology_segmented.png
      radiology_findings.json
      summary_report.txt
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2

# Core modules
from core.filtering import gaussian_filter, bilateral_filter, median_filter
from core.edges import canny_edges, laplacian_edges, sobel_edges
from core.morphology import morph_open, morph_close, fill_holes, morph_dilate
from core.statistics import roi_stats, region_stats
from core.classification import classify_severity, build_finding
from core.contours import find_contours, draw_contours, contour_stats
from core.shapes import shape_features, detect_shape, is_circular
from core.intensity import normalize_minmax, clahe_enhance
from core.clustering import kmeans_segment

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_output")


# ================================================================
# IMAGE GENERATORS — Synthetic but realistic biomedical images
# ================================================================

def generate_retinal_fundus(size=512):
    """Generate a synthetic retinal fundus photograph.

    Creates a dark orange-red background with:
    - Bright optic disc
    - Dark macula
    - Branching vessel network
    - Small bright/dark lesions (simulating exudates/hemorrhages)
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # Background: dark orange-red gradient (typical fundus)
    for y in range(size):
        for x in range(size):
            dist = np.sqrt((x - size//2)**2 + (y - size//2)**2) / (size//2)
            if dist < 1.0:
                r = int(180 * (1 - dist * 0.5))
                g = int(80 * (1 - dist * 0.6))
                b = int(30 * (1 - dist * 0.7))
                img[y, x] = [b, g, r]

    # Optic disc (bright yellowish circle at right-center)
    od_x, od_y, od_r = int(size * 0.65), size // 2, int(size * 0.06)
    cv2.circle(img, (od_x, od_y), od_r, (60, 180, 230), -1)
    cv2.GaussianBlur(img, (15, 15), 5, dst=img)

    # Macula (dark region at center-left)
    mac_x, mac_y = int(size * 0.38), size // 2
    cv2.circle(img, (mac_x, mac_y), int(size * 0.04), (10, 30, 80), -1)

    # Vessels: draw branching lines from optic disc
    rng = np.random.RandomState(42)
    for i in range(8):
        angle = rng.uniform(-np.pi, np.pi)
        pts = [(od_x, od_y)]
        x, y = float(od_x), float(od_y)
        for step in range(60):
            angle += rng.uniform(-0.15, 0.15)
            x += np.cos(angle) * 5
            y += np.sin(angle) * 5
            pts.append((int(x), int(y)))
        thickness = max(1, 3 - i // 3)
        for j in range(len(pts) - 1):
            cv2.line(img, pts[j], pts[j+1], (10, 20, 60), thickness)

    # Bright lesions (exudates) — small bright spots
    for _ in range(12):
        lx = rng.randint(size // 4, 3 * size // 4)
        ly = rng.randint(size // 4, 3 * size // 4)
        lr = rng.randint(3, 8)
        cv2.circle(img, (lx, ly), lr, (80, 200, 240), -1)

    # Dark lesions (hemorrhages) — small dark spots
    for _ in range(8):
        lx = rng.randint(size // 4, 3 * size // 4)
        ly = rng.randint(size // 4, 3 * size // 4)
        lr = rng.randint(4, 12)
        cv2.circle(img, (lx, ly), lr, (5, 10, 40), -1)

    return img


def generate_histology_he(size=512):
    """Generate a synthetic H&E histology tile.

    Creates a pink-purple tissue background with:
    - Dense dark purple tumor region
    - Pink stroma
    - White background (glass)
    - Cell-like dots
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # White glass background
    img[:] = (245, 240, 250)

    # Tissue region (large irregular pink area)
    tissue_mask = np.zeros((size, size), dtype=np.uint8)
    pts = np.array([
        [size*0.1, size*0.15], [size*0.5, size*0.05], [size*0.9, size*0.2],
        [size*0.95, size*0.6], [size*0.85, size*0.95], [size*0.4, size*0.9],
        [size*0.05, size*0.7],
    ], dtype=np.int32)
    cv2.fillPoly(tissue_mask, [pts], 255)

    # Stroma (pink)
    stroma_color = (195, 170, 220)  # BGR: light pink-purple
    img[tissue_mask > 0] = stroma_color

    # Tumor region (dark purple cluster in center)
    tumor_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(tumor_mask, (size//2, size//2), (size//5, size//6), 30, 0, 360, 255, -1)
    tumor_mask = cv2.bitwise_and(tumor_mask, tissue_mask)

    tumor_color = (150, 80, 120)  # dark purple
    img[tumor_mask > 0] = tumor_color

    # Add cell-like texture (random small dots)
    rng = np.random.RandomState(42)
    for _ in range(2000):
        cx = rng.randint(0, size)
        cy = rng.randint(0, size)
        if tissue_mask[cy, cx] > 0:
            r = rng.randint(1, 3)
            if tumor_mask[cy, cx] > 0:
                color = (120 + rng.randint(-20, 20), 50 + rng.randint(-20, 20), 100 + rng.randint(-20, 20))
            else:
                color = (180 + rng.randint(-20, 20), 150 + rng.randint(-20, 20), 200 + rng.randint(-20, 20))
            cv2.circle(img, (cx, cy), r, color, -1)

    return img


def generate_ct_slice(size=512):
    """Generate a synthetic CT axial slice.

    Creates a grayscale image simulating:
    - Black background (air)
    - Soft tissue body contour
    - Brighter bone ring
    - A bright lesion (simulating calcification/tumor)
    - Internal organ structures
    """
    img = np.zeros((size, size), dtype=np.uint8)

    # Body contour (soft tissue ~60-80 HU equivalent, mapped to ~120-160 intensity)
    cv2.ellipse(img, (size//2, size//2), (int(size*0.38), int(size*0.32)),
                0, 0, 360, 130, -1)

    # Bone ring (spine area — bright)
    cv2.ellipse(img, (size//2, int(size*0.6)), (int(size*0.05), int(size*0.08)),
                0, 0, 360, 220, -1)
    cv2.ellipse(img, (size//2, int(size*0.6)), (int(size*0.03), int(size*0.06)),
                0, 0, 360, 130, -1)

    # Lungs (dark air regions)
    cv2.ellipse(img, (int(size*0.35), int(size*0.4)), (int(size*0.12), int(size*0.18)),
                -10, 0, 360, 20, -1)
    cv2.ellipse(img, (int(size*0.65), int(size*0.4)), (int(size*0.12), int(size*0.18)),
                10, 0, 360, 20, -1)

    # Heart (medium density)
    cv2.ellipse(img, (int(size*0.45), int(size*0.45)), (int(size*0.08), int(size*0.1)),
                20, 0, 360, 155, -1)

    # Lesion: bright mass in right lung
    cv2.circle(img, (int(size*0.68), int(size*0.38)), int(size*0.035), 200, -1)

    # Add noise
    noise = np.random.RandomState(42).normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


# ================================================================
# ANALYSIS FUNCTIONS
# ================================================================

def analyze_retinal(img, output_dir):
    """Run retinal fundus analysis pipeline."""
    print("\n" + "=" * 60)
    print("  RETINAL FUNDUS ANALYSIS")
    print("=" * 60)

    # Save input
    cv2.imwrite(os.path.join(output_dir, "retinal_input.png"), img)
    print(f"  [1/5] Input saved ({img.shape[1]}x{img.shape[0]} BGR)")

    # Extract green channel (best vessel contrast)
    green = img[:, :, 1]
    print(f"  [2/5] Green channel extracted: range [{green.min()}, {green.max()}]")

    # Enhance with CLAHE
    enhanced = clahe_enhance(green, clip_limit=3.0, grid_size=(8, 8))
    print(f"  [3/5] CLAHE enhanced: range [{enhanced.min()}, {enhanced.max()}]")

    # Edge detection for vessels
    edges = canny_edges(enhanced, low=30, high=100)
    print(f"  [3/5] Canny edges: {np.count_nonzero(edges)} edge pixels")

    # Segment with K-means clustering
    result = kmeans_segment(enhanced, k=4)
    labels = result[0] if isinstance(result, tuple) else result
    print(f"  [4/5] K-means segmentation: {len(np.unique(labels))} clusters")

    # Find contours on edge map
    edges_dilated = morph_dilate(edges, kernel_size=2)
    contours = find_contours(edges_dilated, min_area=20)
    print(f"  [4/5] Found {len(contours)} structures")

    # Build findings
    findings = []
    stats = roi_stats(green, edges_dilated)
    print(f"  [5/5] ROI statistics: mean={stats['mean']:.1f}, std={stats['std']:.1f}")

    # Analyze each region for anomalies
    region_masks = []
    for i in range(len(np.unique(labels))):
        rmask = (labels == i).astype(np.uint8) * 255
        region_masks.append(rmask)
        rstats = roi_stats(green, rmask)

        # Z-score anomaly: compare region mean to global mean
        if stats['std'] > 0:
            zscore = abs(rstats['mean'] - stats['mean']) / stats['std']
        else:
            zscore = 0

        if zscore > 1.5:
            severity = classify_severity(zscore * 30, min(zscore / 3.0, 1.0))
            finding = build_finding(
                f"retinal_region_{i}",
                severity,
                min(zscore / 3.0, 0.99),
                {"mean": round(rstats["mean"], 2), "std": round(rstats["std"], 2),
                 "zscore": round(zscore, 2), "pixel_count": int(np.count_nonzero(rmask))},
                {"cluster_id": int(i)},
            )
            findings.append(finding)
            print(f"         Anomaly: cluster {i}, z-score={zscore:.2f}, severity={severity}")

    # Create annotated result image
    result_img = img.copy()
    for c in contours:
        cv2.drawContours(result_img, [c["contour"]], -1, (0, 255, 0), 1)
    for f in findings:
        cluster_id = f["geometric_properties"]["cluster_id"]
        rmask = region_masks[cluster_id]
        # Color anomalous regions
        color_map = {"CRITICAL": (0, 0, 255), "HIGH": (0, 128, 255),
                     "MEDIUM": (0, 255, 255), "LOW": (0, 255, 0)}
        color = color_map.get(f["severity"], (255, 255, 255))
        result_img[rmask > 0] = cv2.addWeighted(
            result_img[rmask > 0], 0.6,
            np.full_like(result_img[rmask > 0], color), 0.4, 0
        )

    # Add legend
    y_off = 20
    for sev, col in [("CRITICAL", (0,0,255)), ("HIGH", (0,128,255)),
                      ("MEDIUM", (0,255,255)), ("LOW", (0,255,0))]:
        cv2.rectangle(result_img, (10, y_off-12), (25, y_off+2), col, -1)
        cv2.putText(result_img, sev, (30, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        y_off += 18

    cv2.imwrite(os.path.join(output_dir, "retinal_segmented.png"), result_img)
    print(f"  Result saved: retinal_segmented.png")

    return findings


def analyze_histology(img, output_dir):
    """Run histology H&E analysis pipeline."""
    print("\n" + "=" * 60)
    print("  HISTOLOGY H&E ANALYSIS")
    print("=" * 60)

    cv2.imwrite(os.path.join(output_dir, "histology_input.png"), img)
    print(f"  [1/5] Input saved ({img.shape[1]}x{img.shape[0]} BGR)")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Tissue detection (non-white regions)
    tissue_mask = (gray < 220).astype(np.uint8) * 255
    tissue_mask = morph_close(tissue_mask, kernel_size=5)
    tissue_mask = fill_holes(tissue_mask)
    tissue_ratio = np.count_nonzero(tissue_mask) / tissue_mask.size
    print(f"  [2/5] Tissue detected: {tissue_ratio*100:.1f}% of tile")

    # K-means segmentation (background, stroma, tumor)
    result = kmeans_segment(gray, k=3)
    labels = result[0] if isinstance(result, tuple) else result
    print(f"  [3/5] K-means: {len(np.unique(labels))} tissue clusters")

    # Identify darkest cluster as potential tumor
    unique_labels = np.unique(labels)
    cluster_means = []
    for i in unique_labels:
        cmask = (labels == i).astype(np.uint8) * 255
        overlap = cv2.bitwise_and(cmask, tissue_mask)
        overlap_ratio = np.count_nonzero(overlap) / max(np.count_nonzero(cmask), 1)
        cmean = cv2.mean(gray, mask=cmask)[0]
        cluster_means.append((int(i), cmean, overlap_ratio, np.count_nonzero(overlap)))
        print(f"         Cluster {i}: mean={cmean:.1f}, tissue_overlap={overlap_ratio:.2f}, pixels={np.count_nonzero(overlap)}")

    # Darkest cluster that overlaps substantially with tissue = tumor
    cluster_means.sort(key=lambda x: x[1])  # sort by brightness (darkest first)
    tumor_cluster = None
    for cid, cmean, overlap_ratio, overlap_px in cluster_means:
        if overlap_ratio > 0.3 and cmean < 200:  # must be in tissue and not white
            tumor_cluster = cid
            break

    tumor_mask = np.zeros_like(gray)
    if tumor_cluster is not None:
        tumor_mask = (labels == tumor_cluster).astype(np.uint8) * 255
        tumor_mask = cv2.bitwise_and(tumor_mask, tissue_mask)
        print(f"         Selected cluster {tumor_cluster} as tumor")

    tumor_area = np.count_nonzero(tumor_mask)
    tissue_area = max(np.count_nonzero(tissue_mask), 1)
    tumor_ratio = tumor_area / tissue_area
    print(f"  [4/5] Tumor region: {tumor_ratio*100:.1f}% of tissue")

    # Statistics
    findings = []
    if tumor_area > 0:
        tstats = roi_stats(gray, tumor_mask)
        gradient = tumor_ratio * 100  # use tumor ratio as severity proxy
        confidence = min(0.5 + tumor_ratio, 0.99)
        severity = classify_severity(gradient, confidence)

        finding = build_finding(
            "tumor_region",
            severity,
            round(confidence, 3),
            {"tumor_ratio": round(tumor_ratio, 4), "tumor_area_px": int(tumor_area),
             "mean_intensity": round(tstats["mean"], 2), "std": round(tstats["std"], 2)},
            {"tissue_ratio": round(tissue_ratio, 4)},
        )
        findings.append(finding)
        print(f"  [5/5] Finding: tumor ratio={tumor_ratio*100:.1f}%, severity={severity}, confidence={confidence:.2f}")

    # Annotated result
    result_img = img.copy()
    # Overlay tumor in red
    tumor_overlay = result_img.copy()
    tumor_overlay[tumor_mask > 0] = (0, 0, 200)
    result_img = cv2.addWeighted(result_img, 0.7, tumor_overlay, 0.3, 0)
    # Draw contours
    tcontours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_img, tcontours, -1, (0, 0, 255), 2)
    cv2.putText(result_img, f"Tumor: {tumor_ratio*100:.1f}%", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite(os.path.join(output_dir, "histology_segmented.png"), result_img)
    print(f"  Result saved: histology_segmented.png")

    return findings


def analyze_radiology(img, output_dir):
    """Run radiology CT slice analysis pipeline."""
    print("\n" + "=" * 60)
    print("  RADIOLOGY CT ANALYSIS")
    print("=" * 60)

    cv2.imwrite(os.path.join(output_dir, "radiology_input.png"), img)
    print(f"  [1/5] Input saved ({img.shape[1]}x{img.shape[0]} grayscale)")

    # CLAHE enhance
    enhanced = clahe_enhance(img, clip_limit=2.0, grid_size=(8, 8))
    print(f"  [2/5] CLAHE enhanced")

    # Bilateral filter for noise reduction
    filtered = bilateral_filter(enhanced, d=9, sigma_color=75, sigma_space=75)
    print(f"  [2/5] Bilateral filtered")

    # Body segmentation (threshold out air)
    _, body_mask = cv2.threshold(filtered, 30, 255, cv2.THRESH_BINARY)
    body_mask = morph_close(body_mask, kernel_size=7)
    body_mask = fill_holes(body_mask)
    print(f"  [3/5] Body segmented: {np.count_nonzero(body_mask)} pixels")

    # Lesion detection: bright regions within body (>180 intensity)
    _, bright_mask = cv2.threshold(filtered, 170, 255, cv2.THRESH_BINARY)
    lesion_mask = cv2.bitwise_and(bright_mask, body_mask)
    lesion_mask = morph_open(lesion_mask, kernel_size=3)  # remove noise
    print(f"  [3/5] Bright lesion candidates: {np.count_nonzero(lesion_mask)} pixels")

    # Find lesion contours
    contours = find_contours(lesion_mask, min_area=50)
    print(f"  [4/5] Found {len(contours)} lesion candidate(s)")

    # Analyze each lesion
    findings = []
    global_stats = roi_stats(img, body_mask)

    for i, c in enumerate(contours):
        cdata = c["contour"]
        feat = shape_features(cdata)

        # Create mask for this contour to get intensity stats
        cmask = np.zeros_like(img)
        cv2.drawContours(cmask, [cdata], -1, 255, -1)
        cstats = roi_stats(img, cmask)

        # Compute z-score from global body stats
        zscore = 0
        if global_stats["std"] > 0:
            zscore = abs(cstats["mean"] - global_stats["mean"]) / global_stats["std"]

        gradient = zscore * 25
        confidence = min(0.4 + zscore * 0.15, 0.99)
        severity = classify_severity(gradient, confidence)

        finding = build_finding(
            f"lesion_{i}",
            severity,
            round(confidence, 3),
            {"mean_intensity": round(cstats["mean"], 2),
             "area_px": round(feat["area"], 0),
             "circularity": round(feat["circularity"], 3),
             "zscore": round(zscore, 2),
             "equivalent_diameter_mm": round(feat["equivalent_diameter"] * 0.5, 1)},
            {"centroid": [round(feat["centroid"][0], 1), round(feat["centroid"][1], 1)],
             "bounding_rect": list(feat["bounding_rect"]),
             "shape": detect_shape(cdata)},
        )
        findings.append(finding)
        print(f"         Lesion {i}: area={feat['area']:.0f}px, z={zscore:.2f}, "
              f"severity={severity}, shape={detect_shape(cdata)}")

    # Annotated result (colorize)
    result_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Draw body contour
    body_contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_img, body_contours, -1, (100, 100, 100), 1)

    # Draw lesions with severity colors
    color_map = {"CRITICAL": (0, 0, 255), "HIGH": (0, 128, 255),
                 "MEDIUM": (0, 255, 255), "LOW": (0, 255, 0)}
    for i, (c, f) in enumerate(zip(contours, findings)):
        color = color_map.get(f["severity"], (255, 255, 255))
        cv2.drawContours(result_img, [c["contour"]], -1, color, 2)
        x, y, w, h = cv2.boundingRect(c["contour"])
        cv2.putText(result_img, f"L{i}:{f['severity']}", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.imwrite(os.path.join(output_dir, "radiology_segmented.png"), result_img)
    print(f"  Result saved: radiology_segmented.png")

    return findings


# ================================================================
# MAIN
# ================================================================

def run_demo(modalities=None, auto_open=True):
    """Run the VIDIO demo pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if modalities is None:
        modalities = ["retinal", "histology", "radiology"]

    all_findings = {}
    start = time.time()

    print("\n" + "#" * 60)
    print("  VIDIO — Standalone Demo")
    print("  Vision-Integrated Diagnostic Imaging Orchestrator")
    print("#" * 60)

    if "retinal" in modalities:
        img = generate_retinal_fundus(512)
        findings = analyze_retinal(img, OUTPUT_DIR)
        all_findings["retinal"] = findings
        with open(os.path.join(OUTPUT_DIR, "retinal_findings.json"), "w") as f:
            json.dump(findings, f, indent=2, default=str)

    if "histology" in modalities:
        img = generate_histology_he(512)
        findings = analyze_histology(img, OUTPUT_DIR)
        all_findings["histology"] = findings
        with open(os.path.join(OUTPUT_DIR, "histology_findings.json"), "w") as f:
            json.dump(findings, f, indent=2, default=str)

    if "radiology" in modalities:
        img = generate_ct_slice(512)
        findings = analyze_radiology(img, OUTPUT_DIR)
        all_findings["radiology"] = findings
        with open(os.path.join(OUTPUT_DIR, "radiology_findings.json"), "w") as f:
            json.dump(findings, f, indent=2, default=str)

    elapsed = time.time() - start

    # Summary report
    print("\n" + "=" * 60)
    print("  SUMMARY REPORT")
    print("=" * 60)
    total_findings = sum(len(v) for v in all_findings.values())
    sev_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for modality, findings in all_findings.items():
        print(f"\n  {modality.upper()}: {len(findings)} finding(s)")
        for f in findings:
            sev = f.get("severity", "LOW")
            sev_counts[sev] = sev_counts.get(sev, 0) + 1
            print(f"    - {f['finding_type']}: {sev} (conf={f['confidence']:.2f})")

    print(f"\n  Total findings: {total_findings}")
    print(f"  Severity breakdown: {sev_counts}")
    print(f"  Time elapsed: {elapsed:.2f}s")
    print(f"\n  Output directory: {OUTPUT_DIR}")
    print(f"  Files:")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        fsize = os.path.getsize(fpath)
        print(f"    {fname:40s} {fsize:>8,d} bytes")

    # Save text report
    report_path = os.path.join(OUTPUT_DIR, "summary_report.txt")
    with open(report_path, "w") as f:
        f.write("VIDIO Demo Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total findings: {total_findings}\n")
        f.write(f"Severity breakdown: {sev_counts}\n")
        f.write(f"Time: {elapsed:.2f}s\n\n")
        for modality, findings in all_findings.items():
            f.write(f"\n{modality.upper()}\n{'-'*30}\n")
            for finding in findings:
                f.write(f"  Type: {finding['finding_type']}\n")
                f.write(f"  Severity: {finding['severity']}\n")
                f.write(f"  Confidence: {finding['confidence']}\n")
                f.write(f"  Stats: {json.dumps(finding['statistics'], default=str)}\n\n")

    print(f"\n  Report saved: summary_report.txt")
    print("=" * 60)

    # Open output images
    if auto_open and sys.platform == "win32":
        try:
            os.startfile(OUTPUT_DIR)
        except Exception:
            pass

    return all_findings


if __name__ == "__main__":
    mods = None
    auto_open = True

    args = sys.argv[1:]
    if "--no-open" in args:
        auto_open = False
        args.remove("--no-open")
    if args:
        mods = [a.lower() for a in args if a in ("retinal", "histology", "radiology")]

    run_demo(mods or None, auto_open)
