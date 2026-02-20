"""
Tile extraction utilities for histopathology images.

This module provides two tiling strategies:

1. :func:`extract_tiles` -- In-memory tiling of a NumPy array that has
   already been loaded.  Supports configurable overlap between adjacent
   tiles.
2. :func:`extract_tiles_from_wsi` -- On-disk tiling of a whole-slide
   image (SVS, NDPI, MRXS, ...) via the OpenSlide library.  The image is
   never loaded in full; instead, tiles are read one-by-one through
   OpenSlide's region API, making this suitable for gigapixel files.

Both functions apply the same tissue-content filter (grayscale < 220) and
return lists of tile dictionaries with identical schemas (plus an extra
``level`` key for the WSI variant).

Design Notes
------------
* **Overlap support** is only available in the in-memory path because
  OpenSlide's ``read_region`` already returns pixel data for arbitrary
  rectangles, so overlapping reads are trivially expressible by the
  caller.
* **Tile coordinates** are always expressed in the coordinate system of
  the *read level* (not the base level).  When working with pyramidal
  WSI at a down-sampled level the caller must multiply by the level's
  down-sample factor to recover base-level coordinates.
"""

import numpy as np
import logging

log = logging.getLogger(__name__)


def extract_tiles(img, tile_size=256, overlap=0, tissue_threshold=0.5):
    """
    Extract square tiles from an in-memory image array.

    The image is scanned in raster order (top-to-bottom, left-to-right)
    with a configurable step that allows overlap between neighbouring
    tiles.  Only tiles whose tissue-pixel fraction meets the threshold
    are retained.

    Parameters
    ----------
    img : numpy.ndarray
        Source image, shape ``(H, W)`` or ``(H, W, 3)``.
    tile_size : int, optional
        Edge length of each square tile in pixels.  Default ``256``.
    overlap : int, optional
        Number of overlapping pixels between adjacent tiles.  When set to
        ``0`` (default) tiles are non-overlapping and the step equals
        ``tile_size``.  Positive values produce a step of
        ``tile_size - overlap``, so neighbouring tiles share *overlap*
        columns/rows.  Useful for boundary-effect mitigation in downstream
        classifiers.
    tissue_threshold : float, optional
        Minimum fraction of tissue pixels (grayscale < 220) for a tile to
        be kept.  Default ``0.5`` (50 %).

    Returns
    -------
    list of dict
        Each dict contains:

        ``tile`` : numpy.ndarray
            Pixel data, shape ``(tile_size, tile_size, ...)``
        ``x``, ``y`` : int
            Top-left corner in the source image.
        ``w``, ``h`` : int
            Tile dimensions (always ``tile_size``).
        ``tissue_ratio`` : float
            Fraction of tissue pixels in ``[0, 1]``.

    Notes
    -----
    * The tissue heuristic (``gray < 220``) is identical to the one used
      by :meth:`~pipelines.histology.pipeline.HistologyPipeline._extract_tiles`
      and :func:`~pipelines.histology.preprocessing.detect_tissue`.
    * Edge tiles that would extend beyond the image boundary are silently
      skipped (no zero-padding).

    Examples
    --------
    >>> tiles = extract_tiles(wsi_thumbnail, tile_size=512, overlap=64)
    >>> len(tiles)
    347
    """
    h, w = img.shape[:2]
    step = tile_size - overlap
    tiles = []

    for y in range(0, h - tile_size + 1, step):
        for x in range(0, w - tile_size + 1, step):
            tile = img[y:y + tile_size, x:x + tile_size]

            # Convert to grayscale for tissue detection.
            if len(tile.shape) == 3:
                gray = np.mean(tile, axis=2)
            else:
                gray = tile

            # Tissue heuristic: stained tissue is darker than 220.
            tissue_ratio = np.sum(gray < 220) / gray.size
            if tissue_ratio >= tissue_threshold:
                tiles.append({
                    'tile': tile,
                    'x': x, 'y': y,
                    'w': tile_size, 'h': tile_size,
                    'tissue_ratio': tissue_ratio,
                })

    log.info(f'Extracted {len(tiles)} tiles ({tile_size}x{tile_size}) from {h}x{w} image')
    return tiles


def extract_tiles_from_wsi(wsi_path, level=0, tile_size=256, tissue_threshold=0.5):
    """
    Extract tiles directly from a whole-slide image file on disk.

    Uses the `OpenSlide <https://openslide.org/>`_ library to read
    individual regions from a pyramidal WSI without loading the entire
    image into memory.  This is essential for gigapixel slides (often
    50 000+ pixels on each side at the base level).

    Parameters
    ----------
    wsi_path : str or pathlib.Path
        Path to the WSI file.  Supported formats depend on the OpenSlide
        build and typically include Aperio SVS, Hamamatsu NDPI, MIRAX
        MRXS, Leica SCN, Ventana BIF, and generic TIFF.
    level : int, optional
        Pyramid level to read from.  ``0`` is the highest-resolution base
        layer; higher indices correspond to progressively down-sampled
        representations.  Default ``0``.

        Common pyramid layout (example for a 40x Aperio SVS):

        ======  =================  ==============
        Level   Dimensions         Magnification
        ======  =================  ==============
        0       80 000 x 60 000    40x (base)
        1       20 000 x 15 000    10x
        2        5 000 x  3 750     2.5x
        ======  =================  ==============

    tile_size : int, optional
        Edge length of each square tile in pixels *at the chosen level*.
        Default ``256``.
    tissue_threshold : float, optional
        Minimum tissue-pixel fraction for retention.  Default ``0.5``.

    Returns
    -------
    list of dict
        Each dict contains the same keys as :func:`extract_tiles` plus:

        ``level`` : int
            The pyramid level from which the tile was read.

    Raises
    ------
    RuntimeError
        If ``openslide-python`` is not installed.

    Notes
    -----
    * The OpenSlide handle is explicitly closed after tiling to release
      file descriptors -- this is important when processing hundreds of
      slides in batch.
    * ``read_region`` returns an RGBA PIL Image; the alpha channel is
      stripped by converting to RGB before creating the NumPy array.
    * Tile coordinates ``(x, y)`` are in the coordinate space of the
      requested *level*, **not** the base level.  To convert back to
      base-level coordinates, multiply by the level's down-sample factor
      (``slide.level_downsamples[level]``).
    """
    try:
        from openslide import OpenSlide
    except ImportError:
        raise RuntimeError('openslide-python required for WSI tiling')

    slide = OpenSlide(wsi_path)
    dims = slide.level_dimensions[level]
    w, h = dims

    tiles = []
    for y in range(0, h - tile_size + 1, tile_size):
        for x in range(0, w - tile_size + 1, tile_size):
            # read_region returns RGBA PIL Image; convert to RGB ndarray.
            region = slide.read_region((x, y), level, (tile_size, tile_size))
            tile = np.array(region.convert('RGB'))

            gray = np.mean(tile, axis=2)
            tissue_ratio = np.sum(gray < 220) / gray.size

            if tissue_ratio >= tissue_threshold:
                tiles.append({
                    'tile': tile,
                    'x': x, 'y': y,
                    'w': tile_size, 'h': tile_size,
                    'level': level,
                    'tissue_ratio': tissue_ratio,
                })

    slide.close()
    log.info(f'Extracted {len(tiles)} tiles from WSI level {level} ({w}x{h})')
    return tiles
