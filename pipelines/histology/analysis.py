import numpy as np

from core.statistics import roi_stats
from core.classification import classify_severity, build_finding


def classify_tiles(tiles, stats_list, global_stats=None):
    if global_stats is None and stats_list:
        all_means = [s.get('mean', 0) for s in stats_list]
        global_stats = {'mean': np.mean(all_means), 'std': np.std(all_means)}

    classifications = []
    for tile_info, tile_stats in zip(tiles, stats_list):
        mean_val = tile_stats.get('mean', 0)
        std_val = tile_stats.get('std', 0)

        if global_stats and global_stats.get('std', 0) > 0:
            z_score = abs(mean_val - global_stats['mean']) / global_stats['std']
        else:
            z_score = 0

        if z_score > 2.0 and std_val > 20:
            label = 'tumor'
        elif std_val < 10:
            label = 'background'
        elif mean_val > 180:
            label = 'stroma'
        else:
            label = 'normal'

        classifications.append({
            'label': label,
            'z_score': z_score,
            'tile_x': tile_info['x'],
            'tile_y': tile_info['y'],
        })

    return classifications
