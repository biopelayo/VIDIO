import numpy as np
import logging

log = logging.getLogger(__name__)


def extract_tiles(img, tile_size=256, overlap=0, tissue_threshold=0.5):
    h, w = img.shape[:2]
    step = tile_size - overlap
    tiles = []

    for y in range(0, h - tile_size + 1, step):
        for x in range(0, w - tile_size + 1, step):
            tile = img[y:y + tile_size, x:x + tile_size]

            if len(tile.shape) == 3:
                gray = np.mean(tile, axis=2)
            else:
                gray = tile

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
