import os

from cellpose_adapt.config import available_model_list
from .eval import plot_eval
from .data import data
from .optimize import optimize_parameters


def run_model_experiments(main_folder, eval: bool = False):
    result_file_models = os.path.join(main_folder, "results_models.csv")
    options_model = {"model_name": available_model_list}
    optimize_parameters(options_model, data, result_file_models)
    if eval:
        plot_eval(result_file_models)


def run_channel_experiments(main_folder, eval: bool = False):
    result_file_channel = os.path.join(main_folder, "results_channel.csv")
    options_channel_segment = {"channel_segment": [0, 1, 2, 3]}
    optimize_parameters(options_channel_segment, data, result_file_channel)

    options_channel_nuclei = {"channel_nuclei": [0, 1, 2, 3]}
    optimize_parameters(options_channel_nuclei, data, result_file_channel, append_result=True)
    if eval:
        plot_eval(result_file_channel)


def run_normalization_experiments(main_folder, eval: bool = False):
    result_file_normalize = os.path.join(main_folder, "results_normalize.csv")
    options_normalization = {"normalization_min": [0, 1, 5, 10, 20, 30, 50], "normalize": [True, False]}
    optimize_parameters(options_normalization, data, result_file_normalize)

    options_normalization = {"normalization_max": [90, 93, 95, 97, 98, 99, 99.5, 100], "normalize": [True, False]}
    optimize_parameters(options_normalization, data, result_file_normalize, append_result=True)
    if eval:
        plot_eval(result_file_normalize)


def run_diameter_experiments(main_folder, eval: bool = False):
    result_file_diameter = os.path.join(main_folder, "results_diameter.csv")
    options_diameter = {"diameter": [1, 5, 10, 12, 17, 30, 40, 50, 70, 100, 200, 500]}
    optimize_parameters(options_diameter, data, result_file_diameter)
    if eval:
        plot_eval(result_file_diameter)


def run_cellprob_threshold_experiments(main_folder, eval: bool = False):
    result_file_cellprob_threshold = os.path.join(main_folder, "results_cellprob_threshold.csv")
    options_cellprob_threshold = {"cellprob_threshold": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.9, 1]}
    optimize_parameters(options_cellprob_threshold, data, result_file_cellprob_threshold)
    if eval:
        plot_eval(result_file_cellprob_threshold)


def run_min_size_experiments(main_folder, eval: bool = False):
    result_file_min_size = os.path.join(main_folder, "results_min_size.csv")
    options_min_size = {"min_size": [5, 10, 30, 50, 70, 90, 200]}
    optimize_parameters(options_min_size, data, result_file_min_size)
    if eval:
        plot_eval(result_file_min_size)


def run_stitch_threshold_experiments(main_folder, eval: bool = False):
    result_file_stitch_threshold = os.path.join(main_folder, "results_stitch_threshold.csv")
    options_stitch_threshold = {"stitch_threshold": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5], "do_3D": [False]}
    optimize_parameters(options_stitch_threshold, data, result_file_stitch_threshold)
    if eval:
        plot_eval(result_file_stitch_threshold)


def run_tile_overlap_experiments(main_folder, eval: bool = False):
    result_file_tile_overlap = os.path.join(main_folder, "results_tile_overlap.csv")
    options_tile_overlap = {"tile_overlap": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]}
    optimize_parameters(options_tile_overlap, data, result_file_tile_overlap)
    if eval:
        plot_eval(result_file_tile_overlap)