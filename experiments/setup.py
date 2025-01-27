import os

from cellpose_adapt.config import available_model_list
from cellpose_adapt.utils import set_all_seeds
from cellpose_adapt.viz import plot_aggregated_metric_variation
from .data import data
from .optimize import grid_search

# Set the random seed based on the image index
set_all_seeds(42)

main_folder = "results/P013T/"


def run_model_experiments(eval: bool = False):
    result_file_models = os.path.join(main_folder, "experiments_1_models.csv")
    options_model = {
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
    grid_search(options_model, data, result_file_models)
    if eval:
        plot_aggregated_metric_variation(result_file_models)


def run_channel_experiments(eval: bool = False):
    result_file_channel = os.path.join(main_folder, "experiments_2_channel.csv")
    options_channel_segment = {"channel_segment": [0, 1, 2, 3]}
    grid_search(options_channel_segment, data, result_file_channel)

    options_channel_nuclei = {"channel_nuclei": [0, 1, 2, 3]}
    grid_search(options_channel_nuclei, data, result_file_channel, append_result=True)
    if eval:
        plot_aggregated_metric_variation(result_file_channel)


def run_normalization_experiments(eval: bool = False):
    result_file_normalize = os.path.join(main_folder, "experiments_3_normalize.csv")

    options_normalization_off = {
        "normalize": [False],
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
    grid_search(options_normalization_off, data, result_file_normalize)

    options_normalization_min = {
        "percentile_min": [
            0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
        ],
        "normalize": [True],
        "norm3D": [True, False],
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
    grid_search(
        options_normalization_min, data, result_file_normalize, append_result=True
    )

    options_normalization_max = {
        "percentile_max": [
            90,
            91,
            92,
            93,
            94,
            95,
            96,
            97,
            98,
            99,
            99.1,
            99.2,
            99.3,
            99.4,
            99.5,
            99.6,
            99.7,
            99.8,
            99.9,
            100,
        ],
        "normalize": [True],
        "norm3D": [True, False],
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
    grid_search(
        options_normalization_max, data, result_file_normalize, append_result=True
    )
    if eval:
        plot_aggregated_metric_variation(result_file_normalize)


def run_diameter_experiments(eval: bool = False):
    result_file_diameter = os.path.join(main_folder, "experiments_4_diameter.csv")
    options_diameter = {
        "diameter": [
            1,
            5,
            10,
            12,
            17,
            30,
            40,
            50,
            70,
            100,
            200,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
            1500,
            2000,
            3000,
            4000,
            5000,
            7500,
            10000,
            15000,
            20000,
            25000,
            30000,
            35000,
            40000,
            45000,
            50000,
        ],
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
    grid_search(options_diameter, data, result_file_diameter)
    if eval:
        plot_aggregated_metric_variation(result_file_diameter)


def run_cellprob_threshold_experiments(eval: bool = False):
    result_file_cellprob_threshold = os.path.join(
        main_folder, "experiments_5_cellprob_threshold.csv"
    )
    options_cellprob_threshold = {
        "cellprob_threshold": [
            0.0,
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.5,
            9.0,
            10.0,
            15.0,
            20.0,
            25.0,
            30.0,
            -1.0,
            -2.0,
            -3.0,
            -4.0,
            -5.0,
            -6.0,
            -7.5,
            -9.0,
            -10.0,
            -15.0,
            -20.0,
            -25.0,
            -30.0,
        ],
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
    grid_search(options_cellprob_threshold, data, result_file_cellprob_threshold)
    if eval:
        plot_aggregated_metric_variation(result_file_cellprob_threshold)


def run_min_size_experiments(eval: bool = False):
    result_file_min_size = os.path.join(main_folder, "experiments_6_min_size.csv")
    options_min_size = {
        "min_size": [
            50,
            100,
            200,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
            1200,
            1400,
            1600,
            1800,
            2000,
            3000,
            4000,
            5000,
            6000,
            7000,
            8000,
            9000,
            10000,
            12000,
            14000,
            16000,
            18000,
            20000,
            25000,
            30000,
            35000,
            40000,
            45000,
            50000,
            55000,
            60000,
            65000,
            70000,
            75000,
            80000,
        ],
        "stitch_threshold": [0.0],
    }
    grid_search(options_min_size, data, result_file_min_size)
    if eval:
        plot_aggregated_metric_variation(result_file_min_size)


def run_stitch_threshold_experiments(eval: bool = False):
    result_file_stitch_threshold = os.path.join(
        main_folder, "experiments_7_stitch_threshold.csv"
    )
    options_stitch_threshold = {
        "stitch_threshold": [
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.75,
            0.8,
            0.9,
            0.95,
            0.99,
            1.0,
        ],
        "do_3D": [False],
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
    grid_search(options_stitch_threshold, data, result_file_stitch_threshold)
    if eval:
        plot_aggregated_metric_variation(result_file_stitch_threshold)


def run_tile_experiments(eval: bool = False):
    result_file_tile = os.path.join(main_folder, "experiments_8_tile.csv")

    options_tile_overlap = {
        "tile_overlap": [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3],
        "tile_norm_blocksize": [5],
        "tile_norm_smooth3D": [1],
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
    grid_search(options_tile_overlap, data, result_file_tile)

    options_tile_norm_blocksize = {
        "tile_overlap": [0.0],
        "tile_norm_blocksize": [0, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "tile_norm_smooth3D": [1],
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
    grid_search(options_tile_norm_blocksize, data, result_file_tile, append_result=True)

    options_tile_norm_smooth3D = {
        "tile_overlap": [0.0],
        "tile_norm_blocksize": [5],
        "tile_norm_smooth3D": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
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
    grid_search(options_tile_norm_smooth3D, data, result_file_tile, append_result=True)

    if eval:
        plot_aggregated_metric_variation(result_file_tile)


def run_smoothing_experiments(eval: bool = False):
    result_file_smoothing = os.path.join(main_folder, "experiments_9_smoothing.csv")

    options_sharpen_radius = {
        "sharpen_radius": [
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
        ],
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
    grid_search(options_sharpen_radius, data, result_file_smoothing)

    options_smooth_radius = {
        "smooth_radius": [0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10],
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
    grid_search(options_smooth_radius, data, result_file_smoothing, append_result=True)

    if eval:
        plot_aggregated_metric_variation(result_file_smoothing)
