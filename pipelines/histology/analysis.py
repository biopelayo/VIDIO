"""
Tile-level tissue classification for histopathology analysis.

This module assigns a tissue-type label to each tile based on its
intensity statistics relative to the global tile population.  It is a
lightweight, feature-engineered classifier (no deep learning required)
designed for rapid first-pass screening.

Classification Decision Tree
-----------------------------
For each tile the following rules are evaluated **in order**:

1. **tumour** -- ``z_score > 2.0 AND std > 20``
   The tile's mean intensity is a statistical outlier (more than 2
   standard deviations from the slide mean) *and* it shows high
   internal variability.  High std indicates heterogeneous tissue
   architecture typical of tumour regions (mixed nuclei densities,
   necrosis borders, irregular gland patterns).

2. **background** -- ``std < 10``
   Very low variability means the tile is nearly uniform, which is
   characteristic of glass background or empty slide areas that
   narrowly passed the tissue-content filter.

3. **stroma** -- ``mean > 180``
   Bright mean intensity with moderate variability suggests loosely
   packed connective tissue (stroma / fibrosis) which appears pinkish
   and relatively light in H&E.

4. **normal** -- all other tiles
   Tiles that do not match any of the above criteria are classified as
   histologically normal tissue.

Design Rationale
----------------
The thresholds (z > 2.0, std > 20, std < 10, mean > 180) were chosen
empirically from TCGA-BRCA and TCGA-LUAD pilot datasets.  They favour
specificity over sensitivity: the goal is to highlight only the most
clearly abnormal tiles for pathologist review, accepting that some
borderline tumour tiles may be labelled "normal".
"""

import numpy as np

from core.statistics import roi_stats
from core.classification import classify_severity, build_finding


def classify_tiles(tiles, stats_list, global_stats=None):
    """
    Classify each tile as tumour, background, stroma, or normal tissue.

    Parameters
    ----------
    tiles : list of dict
        Tile dictionaries as produced by
        :func:`~pipelines.histology.tiling.extract_tiles` or
        :meth:`~pipelines.histology.pipeline.HistologyPipeline._extract_tiles`.
        Each must contain at least ``'x'`` and ``'y'`` keys.
    stats_list : list of dict
        Per-tile statistics (one entry per tile, same order).  Each dict
        must contain ``'mean'`` and ``'std'`` keys (float).
    global_stats : dict, optional
        Pre-computed population statistics with keys ``'mean'`` and
        ``'std'``.  If ``None``, they are derived automatically from
        *stats_list* by averaging all per-tile means and computing the
        standard deviation across tiles.  Supplying pre-computed values
        is useful when classifying a subset of tiles against the full
        slide's distribution.

    Returns
    -------
    list of dict
        One classification record per tile.  Keys:

        ``label`` : str
            One of ``'tumor'``, ``'background'``, ``'stroma'``, or
            ``'normal'``.
        ``z_score`` : float
            Absolute z-score of the tile's mean intensity relative to
            *global_stats*.  Zero if global std is zero.
        ``tile_x``, ``tile_y`` : int
            Top-left coordinates of the tile (copied from *tiles*).

    Notes
    -----
    * The classification rules are applied in strict priority order (see
      module-level docstring).  A tile that matches the tumour rule will
      *not* be checked against subsequent rules even if it also matches
      the stroma brightness criterion.
    * ``z_score`` is based on the *mean* intensity only.  The *std* guard
      in the tumour rule (``std > 20``) prevents low-variability outliers
      (e.g. ink marks, air bubbles) from being labelled as tumour.

    Examples
    --------
    >>> from pipelines.histology.analysis import classify_tiles
    >>> classes = classify_tiles(tiles, stats)
    >>> tumour_tiles = [c for c in classes if c['label'] == 'tumor']
    >>> print(f'{len(tumour_tiles)} / {len(classes)} tiles flagged as tumour')
    """
    # Derive global distribution from per-tile means if not provided.
    if global_stats is None and stats_list:
        all_means = [s.get('mean', 0) for s in stats_list]
        global_stats = {'mean': np.mean(all_means), 'std': np.std(all_means)}

    classifications = []
    for tile_info, tile_stats in zip(tiles, stats_list):
        mean_val = tile_stats.get('mean', 0)
        std_val = tile_stats.get('std', 0)

        # Compute z-score: how far this tile's mean deviates from the
        # global tile population.
        if global_stats and global_stats.get('std', 0) > 0:
            z_score = abs(mean_val - global_stats['mean']) / global_stats['std']
        else:
            z_score = 0

        # --- Classification decision tree (priority order) ---
        if z_score > 2.0 and std_val > 20:
            # Outlier mean + high heterogeneity => tumour.
            label = 'tumor'
        elif std_val < 10:
            # Very uniform tile => residual background.
            label = 'background'
        elif mean_val > 180:
            # Bright, moderate-variability tile => stroma / fibrosis.
            label = 'stroma'
        else:
            # Default: histologically normal tissue.
            label = 'normal'

        classifications.append({
            'label': label,
            'z_score': z_score,
            'tile_x': tile_info['x'],
            'tile_y': tile_info['y'],
        })

    return classifications
