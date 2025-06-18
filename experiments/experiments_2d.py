from cellpose_adapt.viz import plot_aggregated_metric_variation
from cellpose_adapt.viz import plot_aggregated_metric_variation

from .optimization import grid_search
from .results import save_best_config_per_image


def run_experiments(data, result_file: str):
    parameters = {
        # "do_3D": [False],
        "stitch_threshold": [0],
        "channel_axis": [0],
        "tile_overlap": [0.1],
        "model_name": ["cyto3"],
        "min_size": [i for i in range(1, 100)],
    }

    grid_search(parameters, data, result_file)

    best_config_files = save_best_config_per_image(result_file, metric="jaccard")
    return best_config_files


def eval_2d(data, result_file: str):
    plot_aggregated_metric_variation(result_file)
