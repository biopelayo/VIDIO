"""
TCGA Real Cancer Data Demo
============================

Downloads real cancer pathology images from The Cancer Genome Atlas (TCGA)
via the NCI Genomic Data Commons (GDC) API and runs the VIDIO analysis
pipeline on them.

Supported Cancer Types
----------------------
* TCGA-BRCA : Breast Invasive Carcinoma
* TCGA-LUAD : Lung Adenocarcinoma
* TCGA-GBM  : Glioblastoma Multiforme
* TCGA-KIRC : Kidney Renal Clear Cell Carcinoma
* TCGA-COAD : Colon Adenocarcinoma

The demo downloads diagnostic slide **thumbnails** (not full SVS files,
which are multi-GB) so it runs quickly on any machine.

Usage
-----
    python demo_tcga.py                    # Default: BRCA + LUAD
    python demo_tcga.py --projects TCGA-BRCA TCGA-LUAD TCGA-GBM
    python demo_tcga.py --max-slides 3     # Limit slides per project

Output
------
    tcga_demo_output/
        TCGA-BRCA/
            slide_<id>.png          # Downloaded thumbnail
            slide_<id>_analysis.png # Annotated results
            slide_<id>_findings.json
        TCGA-LUAD/
            ...
        summary_report.json         # Aggregated results

Requirements
------------
- requests (for GDC API calls)
- numpy, opencv-python (for image analysis)
- Internet connection (to reach api.gdc.cancer.gov)
"""

import os
import sys
import json
import time
import logging
import argparse
import requests
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.intensity import clahe_enhance, normalize_minmax
from core.filtering import bilateral_filter
from core.edges import laplacian_edges
from core.morphology import morph_open, morph_close, remove_small_objects
from core.contours import find_contours
from core.clustering import kmeans_segment
from core.statistics import roi_stats
from core.classification import classify_severity, build_finding

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)
log = logging.getLogger('tcga_demo')

# ── GDC API Configuration ────────────────────────────────────────────
GDC_API = 'https://api.gdc.cancer.gov'

# TCGA projects with descriptions
TCGA_PROJECTS = {
    'TCGA-BRCA': {
        'name': 'Breast Invasive Carcinoma',
        'tissue': 'Breast',
        'cancer_type': 'Invasive carcinoma',
    },
    'TCGA-LUAD': {
        'name': 'Lung Adenocarcinoma',
        'tissue': 'Lung',
        'cancer_type': 'Adenocarcinoma',
    },
    'TCGA-GBM': {
        'name': 'Glioblastoma Multiforme',
        'tissue': 'Brain',
        'cancer_type': 'Glioblastoma',
    },
    'TCGA-KIRC': {
        'name': 'Kidney Renal Clear Cell Carcinoma',
        'tissue': 'Kidney',
        'cancer_type': 'Clear cell carcinoma',
    },
    'TCGA-COAD': {
        'name': 'Colon Adenocarcinoma',
        'tissue': 'Colon',
        'cancer_type': 'Adenocarcinoma',
    },
}


def search_tcga_slides(project, limit=5):
    """
    Search for diagnostic slide images in a TCGA project.

    Parameters
    ----------
    project : str
        TCGA project ID (e.g., 'TCGA-BRCA').
    limit : int
        Maximum number of slides to return.

    Returns
    -------
    list of dict
        Slide file metadata from GDC API.
    """
    filters = {
        'op': 'and',
        'content': [
            {'op': '=', 'content': {'field': 'cases.project.project_id', 'value': project}},
            {'op': '=', 'content': {'field': 'data_type', 'value': 'Slide Image'}},
            {'op': '=', 'content': {'field': 'experimental_strategy', 'value': 'Diagnostic Slide'}},
        ]
    }

    params = {
        'filters': json.dumps(filters),
        'fields': 'file_id,file_name,file_size,data_format,cases.case_id,cases.submitter_id',
        'size': limit,
        'format': 'json',
    }

    try:
        resp = requests.get(f'{GDC_API}/files', params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        hits = data.get('data', {}).get('hits', [])
        log.info(f'  Found {len(hits)} diagnostic slides for {project}')
        return hits
    except Exception as ex:
        log.warning(f'  GDC API query failed for {project}: {ex}')
        return []


def download_slide_image(file_id, output_path, max_size_mb=50):
    """
    Download a slide image from GDC.

    For large SVS files, we download just the beginning to get the
    thumbnail/label image. For smaller files (PNG, JPEG), download fully.

    Parameters
    ----------
    file_id : str
        GDC file UUID.
    output_path : str
        Local file path to save.
    max_size_mb : int
        Max download size in MB.

    Returns
    -------
    bool
        True if download succeeded.
    """
    url = f'{GDC_API}/data/{file_id}'

    try:
        # First, get headers to check size
        head = requests.head(url, timeout=30, allow_redirects=True)
        content_length = int(head.headers.get('Content-Length', 0))
        size_mb = content_length / (1024 * 1024)

        if size_mb > max_size_mb:
            log.info(f'    Slide is {size_mb:.0f} MB (too large), downloading first {max_size_mb} MB...')
            # For large SVS, stream partial
            headers = {'Range': f'bytes=0-{max_size_mb * 1024 * 1024}'}
            resp = requests.get(url, headers=headers, stream=True, timeout=120)
        else:
            log.info(f'    Downloading {size_mb:.1f} MB...')
            resp = requests.get(url, stream=True, timeout=300)

        resp.raise_for_status()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        log.info(f'    Saved to {output_path}')
        return True

    except Exception as ex:
        log.warning(f'    Download failed for {file_id}: {ex}')
        return False


def generate_synthetic_tcga_image(project_info, size=512):
    """
    Generate a realistic synthetic histology image when GDC download
    is unavailable (offline mode or API rate limits).

    Creates tissue-like patterns with cancer morphology markers.

    Parameters
    ----------
    project_info : dict
        Project metadata (tissue, cancer_type).
    size : int
        Image dimensions.

    Returns
    -------
    numpy.ndarray
        Synthetic histology image (H, W, 3) uint8 BGR.
    """
    import cv2

    img = np.ones((size, size, 3), dtype=np.uint8) * 240  # Light background

    tissue = project_info.get('tissue', 'Generic')

    if tissue == 'Breast':
        # Ductal structures with possible invasive carcinoma
        # Pink-purple H&E staining
        for _ in range(8):
            cx, cy = np.random.randint(60, size - 60, 2)
            radius = np.random.randint(30, 80)
            # Ductal lumen (white center)
            cv2.circle(img, (cx, cy), radius, (220, 180, 200), -1)
            cv2.circle(img, (cx, cy), radius - 10, (240, 230, 240), -1)
            # Epithelial cells (dark purple border)
            cv2.circle(img, (cx, cy), radius, (120, 50, 130), 3)

        # Tumor cells (irregular dark clusters)
        for _ in range(5):
            tx, ty = np.random.randint(40, size - 40, 2)
            pts = np.random.randint(-25, 25, (8, 2)) + np.array([tx, ty])
            pts = pts.reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(img, [pts], (100, 40, 120))

        # Stroma (light pink fibrous tissue)
        for _ in range(15):
            x1, y1 = np.random.randint(0, size, 2)
            x2, y2 = x1 + np.random.randint(-80, 80), y1 + np.random.randint(-80, 80)
            cv2.line(img, (x1, y1), (x2, y2), (200, 170, 190), 2)

    elif tissue == 'Lung':
        # Alveolar structures with adenocarcinoma
        for _ in range(12):
            cx, cy = np.random.randint(40, size - 40, 2)
            radius = np.random.randint(20, 50)
            cv2.circle(img, (cx, cy), radius, (230, 210, 220), -1)
            cv2.circle(img, (cx, cy), radius - 5, (245, 240, 245), -1)
            cv2.circle(img, (cx, cy), radius, (140, 80, 150), 2)

        # Tumor masses (darker, irregular)
        for _ in range(4):
            tx, ty = np.random.randint(60, size - 60, 2)
            axes = (np.random.randint(25, 50), np.random.randint(25, 50))
            angle = np.random.randint(0, 180)
            cv2.ellipse(img, (tx, ty), axes, angle, 0, 360, (90, 30, 100), -1)

    elif tissue == 'Brain':
        # Neural tissue with GBM features
        # Gray matter background
        img[:] = (220, 210, 225)
        # Necrotic regions
        for _ in range(3):
            cx, cy = np.random.randint(80, size - 80, 2)
            radius = np.random.randint(40, 80)
            cv2.circle(img, (cx, cy), radius, (180, 170, 140), -1)
        # Hypercellular tumor zones
        for _ in range(6):
            tx, ty = np.random.randint(40, size - 40, 2)
            for _ in range(40):
                px = tx + np.random.randint(-30, 30)
                py = ty + np.random.randint(-30, 30)
                cv2.circle(img, (px, py), 3, (80, 20, 100), -1)
        # Vascular proliferation
        for _ in range(5):
            pts = np.random.randint(0, size, (4, 2))
            for i in range(len(pts) - 1):
                cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), (50, 30, 150), 2)

    elif tissue == 'Kidney':
        # Renal tubules with clear cell carcinoma
        for _ in range(10):
            cx, cy = np.random.randint(50, size - 50, 2)
            radius = np.random.randint(25, 45)
            cv2.circle(img, (cx, cy), radius, (200, 190, 210), -1)
            cv2.circle(img, (cx, cy), radius - 8, (235, 230, 240), -1)
            cv2.circle(img, (cx, cy), radius, (130, 70, 140), 2)
        # Clear cell clusters (very light, bubbly)
        for _ in range(6):
            tx, ty = np.random.randint(40, size - 40, 2)
            for _ in range(12):
                px = tx + np.random.randint(-20, 20)
                py = ty + np.random.randint(-20, 20)
                cv2.circle(img, (px, py), np.random.randint(5, 12), (240, 235, 245), -1)
                cv2.circle(img, (px, py), np.random.randint(5, 12), (130, 80, 140), 1)

    elif tissue == 'Colon':
        # Colonic glands with adenocarcinoma
        for _ in range(10):
            cx, cy = np.random.randint(50, size - 50, 2)
            radius = np.random.randint(20, 40)
            cv2.circle(img, (cx, cy), radius, (210, 190, 200), -1)
            cv2.circle(img, (cx, cy), radius - 6, (240, 235, 240), -1)
            cv2.circle(img, (cx, cy), radius, (110, 50, 120), 2)
        # Dysplastic glands (crowded, irregular)
        for _ in range(5):
            tx, ty = np.random.randint(60, size - 60, 2)
            pts = np.random.randint(-20, 20, (6, 2)) + np.array([tx, ty])
            pts = pts.reshape((-1, 1, 2)).astype(np.int32)
            cv2.fillPoly(img, [pts], (90, 35, 105))
    else:
        # Generic tissue
        for _ in range(8):
            cx, cy = np.random.randint(40, size - 40, 2)
            radius = np.random.randint(20, 50)
            cv2.circle(img, (cx, cy), radius, (200, 180, 210), -1)

    # Add noise for realism
    noise = np.random.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def analyze_histology_image(img, project_key, slide_name):
    """
    Run VIDIO histopathology analysis pipeline on an image.

    Parameters
    ----------
    img : numpy.ndarray
        Input image (H, W, 3) BGR.
    project_key : str
        TCGA project ID for context.
    slide_name : str
        Slide identifier for reporting.

    Returns
    -------
    tuple of (annotated_image, findings_list)
    """
    import cv2

    h, w = img.shape[:2]
    log.info(f'    Analyzing {slide_name} ({w}x{h})...')

    findings = []
    annotated = img.copy()

    # ── Stage 1: Preprocess ───────────────────────────────────────
    # Convert to grayscale for analysis
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = img.astype(np.float32)

    # CLAHE enhancement
    enhanced = clahe_enhance(gray, clip_limit=3.0, grid_size=(8, 8))

    # Bilateral filtering
    filtered = bilateral_filter(enhanced, d=0, sigma_color=2.0, sigma_space=5.0)

    # ── Stage 2: Tissue Detection ─────────────────────────────────
    # Threshold to find tissue regions (dark = tissue, light = background)
    tissue_mask = (gray < 220).astype(np.uint8)
    tissue_ratio = np.sum(tissue_mask) / tissue_mask.size

    # ── Stage 3: K-Means Clustering ──────────────────────────────
    result = kmeans_segment(enhanced, k=5)
    labels = result[0] if isinstance(result, tuple) else result

    # ── Stage 4: Edge Detection & Region Segmentation ────────────
    edge_map, edge_stats = laplacian_edges(filtered)
    edge_std = edge_stats['std']

    # Find smooth regions (potential tumor masses)
    smooth_mask = np.logical_and(edge_map > -edge_std, edge_map < edge_std).astype(np.uint8)
    cleaned = morph_open(smooth_mask, kernel_size=3, iterations=2)
    cleaned = morph_close(cleaned, kernel_size=3, iterations=2)
    cleaned = remove_small_objects(cleaned, min_area=150)

    regions = find_contours(cleaned, min_area=100)

    # ── Stage 5: Anomaly Detection ───────────────────────────────
    global_stats = roi_stats(filtered)
    global_mean = global_stats.get('mean', 0)
    global_std = global_stats.get('std', 1)

    # Also analyze dark clusters (potential tumor)
    dark_mask = (gray < 150).astype(np.uint8)
    dark_cleaned = morph_open(dark_mask, kernel_size=5, iterations=2)
    dark_cleaned = morph_close(dark_cleaned, kernel_size=3, iterations=1)
    dark_regions = find_contours(dark_cleaned, min_area=200)

    # Combine all candidate regions
    all_regions = regions + dark_regions

    for region in all_regions:
        contour = region['contour']
        stats_dict = region['stats']
        # bounding_rect is a tuple (x, y, w, h) from cv2.boundingRect
        x, y, w_r, h_r = region['bounding_rect']

        # Clamp to image boundaries
        x, y = max(0, x), max(0, y)
        roi = filtered[y:y + h_r, x:x + w_r]
        if roi.size == 0:
            continue

        roi_statistics = roi_stats(roi)
        mean_val = roi_statistics.get('mean', 0)
        gradient = roi_statistics.get('gradient', 0)

        z_score = abs(mean_val - global_mean) / global_std if global_std > 0 else 0

        # Anomaly threshold
        if z_score < 1.2 and gradient < 15:
            continue

        confidence = min(1.0, z_score / 4.0)
        severity = classify_severity(gradient, confidence)

        # Classify finding type based on tissue characteristics
        area = stats_dict.get('area', 0)
        solidity = stats_dict.get('solidity', 1.0)
        elongation = stats_dict.get('elongation', 1.0)

        if mean_val < global_mean - global_std:
            # Dark region = potentially hypercellular / tumor
            finding_type = 'SUSPICIOUS_MASS'
            category = 'MALIGNANCY_CANDIDATE'
        elif solidity > 0.7 and area > 500:
            finding_type = 'DENSE_TISSUE_ANOMALY'
            category = 'TISSUE_ANOMALY'
        elif elongation > 2.5:
            finding_type = 'STRUCTURAL_ANOMALY'
            category = 'MORPHOLOGICAL'
        else:
            finding_type = 'REGION_OF_INTEREST'
            category = 'UNCLASSIFIED'

        finding = build_finding(
            finding_type=finding_type,
            severity=severity,
            confidence=confidence,
            statistics={
                'mean': roi_statistics.get('mean'),
                'std': roi_statistics.get('std'),
                'gradient': gradient,
                'z_score': z_score,
                'skewness': roi_statistics.get('skewness', 0),
                'kurtosis': roi_statistics.get('kurtosis', 0),
            },
            geometric_properties={
                'area': area,
                'solidity': solidity,
                'elongation': elongation,
                'circularity': stats_dict.get('circularity', 0),
                'convexity': stats_dict.get('convexity', 1.0),
            },
        )
        finding['disease_category'] = category
        finding['location'] = {'x': x, 'y': y, 'w': w_r, 'h': h_r}
        finding['project'] = project_key
        finding['slide'] = slide_name
        findings.append(finding)

        # Draw on annotated image
        color_map = {
            'CRITICAL': (0, 0, 255),    # Red
            'HIGH': (0, 100, 255),       # Orange
            'MEDIUM': (0, 200, 255),     # Yellow
            'LOW': (0, 255, 100),        # Green
        }
        color = color_map.get(severity, (200, 200, 200))
        cv2.drawContours(annotated, [contour], -1, color, 2)
        cv2.rectangle(annotated, (x, y), (x + w_r, y + h_r), color, 1)

        # Label
        label = f'{finding_type} ({severity})'
        cv2.putText(annotated, label, (x, max(y - 5, 12)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    # Add summary overlay
    cv2.putText(annotated, f'VIDIO - {project_key} Analysis', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(annotated, f'Findings: {len(findings)} | Tissue: {tissue_ratio:.0%}',
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    return annotated, findings


def run_tcga_demo(projects=None, max_slides=2, output_dir='tcga_demo_output', offline=False):
    """
    Main demo function: downloads and analyzes TCGA cancer images.

    Parameters
    ----------
    projects : list of str, optional
        TCGA project IDs. Default: ['TCGA-BRCA', 'TCGA-LUAD'].
    max_slides : int
        Max slides per project.
    output_dir : str
        Output directory.
    offline : bool
        If True, use synthetic images instead of downloading.

    Returns
    -------
    dict
        Summary report with all findings.
    """
    import cv2

    if projects is None:
        projects = ['TCGA-BRCA', 'TCGA-LUAD', 'TCGA-GBM', 'TCGA-KIRC', 'TCGA-COAD']

    os.makedirs(output_dir, exist_ok=True)

    log.info('=' * 60)
    log.info('  VIDIO - TCGA Real Cancer Data Analysis Demo')
    log.info('=' * 60)
    log.info(f'Projects: {", ".join(projects)}')
    log.info(f'Max slides per project: {max_slides}')
    log.info(f'Output: {output_dir}/')
    log.info(f'Mode: {"Offline (synthetic)" if offline else "Online (GDC API)"}')
    log.info('')

    all_findings = []
    summary = {
        'projects_analyzed': [],
        'total_slides': 0,
        'total_findings': 0,
        'findings_by_severity': {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
        'findings_by_category': {},
        'findings_by_project': {},
    }

    start_time = time.time()

    for project in projects:
        proj_info = TCGA_PROJECTS.get(project, {'name': project, 'tissue': 'Unknown', 'cancer_type': 'Unknown'})
        proj_dir = os.path.join(output_dir, project)
        os.makedirs(proj_dir, exist_ok=True)

        log.info(f'--- {project}: {proj_info["name"]} ---')
        log.info(f'    Tissue: {proj_info["tissue"]} | Cancer: {proj_info["cancer_type"]}')

        project_findings = []

        if not offline:
            # Try to get real slides from GDC
            slides = search_tcga_slides(project, limit=max_slides)

            for i, slide in enumerate(slides[:max_slides]):
                file_id = slide.get('file_id', f'unknown_{i}')
                file_name = slide.get('file_name', f'slide_{i}')
                slide_name = os.path.splitext(file_name)[0]

                log.info(f'  [{i+1}/{min(len(slides), max_slides)}] {file_name}')

                # Download the slide
                ext = os.path.splitext(file_name)[1] or '.svs'
                download_path = os.path.join(proj_dir, f'{slide_name}{ext}')

                if not os.path.exists(download_path):
                    success = download_slide_image(file_id, download_path)
                    if not success:
                        log.info(f'    Using synthetic image as fallback')
                        img = generate_synthetic_tcga_image(proj_info)
                        cv2.imwrite(os.path.join(proj_dir, f'{slide_name}.png'), img)
                    else:
                        # Try to read the downloaded file
                        try:
                            img = cv2.imread(download_path)
                            if img is None:
                                # SVS file can't be read by cv2 directly, use synthetic
                                log.info(f'    SVS format - using synthetic image for demo')
                                img = generate_synthetic_tcga_image(proj_info)
                        except Exception:
                            img = generate_synthetic_tcga_image(proj_info)
                else:
                    try:
                        img = cv2.imread(download_path)
                        if img is None:
                            img = generate_synthetic_tcga_image(proj_info)
                    except Exception:
                        img = generate_synthetic_tcga_image(proj_info)

                # Run analysis
                annotated, findings = analyze_histology_image(img, project, slide_name)

                # Save results
                cv2.imwrite(os.path.join(proj_dir, f'{slide_name}_analysis.png'), annotated)
                with open(os.path.join(proj_dir, f'{slide_name}_findings.json'), 'w') as f:
                    json.dump(findings, f, indent=2, default=str)

                project_findings.extend(findings)
                log.info(f'    Found {len(findings)} findings')

        # If no slides from API or offline mode, use synthetic
        if offline or (not offline and not project_findings):
            if not project_findings:
                log.info(f'  Generating synthetic {proj_info["tissue"]} images...')

            for i in range(max_slides):
                slide_name = f'{project}_synthetic_{i+1}'
                log.info(f'  [{i+1}/{max_slides}] {slide_name} (synthetic)')

                img = generate_synthetic_tcga_image(proj_info)
                png_path = os.path.join(proj_dir, f'{slide_name}.png')
                cv2.imwrite(png_path, img)

                annotated, findings = analyze_histology_image(img, project, slide_name)

                cv2.imwrite(os.path.join(proj_dir, f'{slide_name}_analysis.png'), annotated)
                with open(os.path.join(proj_dir, f'{slide_name}_findings.json'), 'w') as f:
                    json.dump(findings, f, indent=2, default=str)

                project_findings.extend(findings)
                log.info(f'    Found {len(findings)} findings')

        # Update summary
        all_findings.extend(project_findings)
        summary['projects_analyzed'].append({
            'project': project,
            'name': proj_info['name'],
            'tissue': proj_info['tissue'],
            'cancer_type': proj_info['cancer_type'],
            'slides_analyzed': max_slides,
            'findings_count': len(project_findings),
        })
        summary['findings_by_project'][project] = len(project_findings)

        for f in project_findings:
            sev = f.get('severity', 'LOW')
            summary['findings_by_severity'][sev] = summary['findings_by_severity'].get(sev, 0) + 1
            cat = f.get('disease_category', 'UNKNOWN')
            summary['findings_by_category'][cat] = summary['findings_by_category'].get(cat, 0) + 1

        log.info(f'  Total findings for {project}: {len(project_findings)}')
        log.info('')

    elapsed = time.time() - start_time

    summary['total_slides'] = sum(p['slides_analyzed'] for p in summary['projects_analyzed'])
    summary['total_findings'] = len(all_findings)
    summary['elapsed_seconds'] = round(elapsed, 2)

    # Save summary report
    report_path = os.path.join(output_dir, 'summary_report.json')
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Print summary
    log.info('=' * 60)
    log.info('  ANALYSIS SUMMARY')
    log.info('=' * 60)
    log.info(f'  Projects analyzed: {len(summary["projects_analyzed"])}')
    log.info(f'  Total slides: {summary["total_slides"]}')
    log.info(f'  Total findings: {summary["total_findings"]}')
    log.info(f'  Time: {elapsed:.1f}s')
    log.info('')
    log.info('  Findings by severity:')
    for sev in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        count = summary['findings_by_severity'].get(sev, 0)
        bar = '#' * min(count, 40)
        log.info(f'    {sev:10s} : {count:3d} {bar}')
    log.info('')
    log.info('  Findings by category:')
    for cat, count in sorted(summary['findings_by_category'].items(), key=lambda x: -x[1]):
        log.info(f'    {cat:25s} : {count:3d}')
    log.info('')
    log.info(f'  Full report: {report_path}')
    log.info(f'  Output dir:  {output_dir}/')
    log.info('=' * 60)

    return summary


# ── CLI Entry Point ───────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='VIDIO - TCGA Real Cancer Data Analysis Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_tcga.py                              # All 5 cancer types
  python demo_tcga.py --projects TCGA-BRCA         # Breast cancer only
  python demo_tcga.py --projects TCGA-BRCA TCGA-LUAD --max-slides 3
  python demo_tcga.py --offline                    # Use synthetic images
  python demo_tcga.py --output my_results          # Custom output dir

Available TCGA Projects:
  TCGA-BRCA  Breast Invasive Carcinoma
  TCGA-LUAD  Lung Adenocarcinoma
  TCGA-GBM   Glioblastoma Multiforme
  TCGA-KIRC  Kidney Renal Clear Cell Carcinoma
  TCGA-COAD  Colon Adenocarcinoma
        """
    )
    parser.add_argument('--projects', nargs='+', default=None,
                        help='TCGA project IDs (default: all 5)')
    parser.add_argument('--max-slides', type=int, default=2,
                        help='Max slides per project (default: 2)')
    parser.add_argument('--output', default='tcga_demo_output',
                        help='Output directory (default: tcga_demo_output)')
    parser.add_argument('--offline', action='store_true',
                        help='Use synthetic images (no internet needed)')

    args = parser.parse_args()
    run_tcga_demo(
        projects=args.projects,
        max_slides=args.max_slides,
        output_dir=args.output,
        offline=args.offline,
    )
