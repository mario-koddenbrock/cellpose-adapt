import os

from cellpose_adapt.config import available_model_list
from cellpose_adapt.utils import set_all_seeds
from cellpose_adapt.viz import plot_aggregated_metric_variation
from .optimization import grid_search

# Set the random seed based on the image index
set_all_seeds(42)

main_folder = "results/Zagajewski_Data/"


def run_experiments(data, eval: bool = False):
    result_file_models = os.path.join(main_folder, "results_Zagajewski_Data.csv")
    parameters = {
        "do_3D": [False],
        "stitch_threshold": [0],
        "channel_axis": [0],
        "channel_nuclei": [0],
        "channel_segment": [0],
        "tile_overlap": [0.1],
        "model_name": available_model_list,
        "min_size": [
            100,
            500,
            1000,
            2500,
            5000,
            10000,
            15000,
            20000,
            30000,
            40000,
            50000,
            60000,
            70000,
        ],
    }
    grid_search(parameters, data, result_file_models)

    if eval:
        plot_aggregated_metric_variation(result_file_models)
