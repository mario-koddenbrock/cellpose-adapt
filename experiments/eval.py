import os

from experiments.results import print_best_config_per_image
from cellpose_adapt import viz


def plot_eval(result_path="data/P013T/results.csv"):

    if isinstance(result_path, list):
        for path in result_path:
            if not os.path.exists(path):
                print(f"Result file {path} does not exist.")
                return
    else:
        if not os.path.exists(result_path):
            print(f"Result file {result_path} does not exist.")
            return

    viz.plot_aggregated_metric_variation(result_path, boxplot=True)
    viz.plot_aggregated_metric_variation(result_path, boxplot=False)
    viz.plot_best_scores_barplot(result_path, output_file=result_path[0].replace('.csv', '_best_score.png'))



if __name__ == "__main__":
    main_folder = "../results/P013T/"

    # get all the subfolder
    result_path = os.path.join(main_folder, "results.csv")
    print_best_config_per_image(result_path)
    plot_eval(result_path)

    # Define the result files
    result_file_models = os.path.join(main_folder, "experiments_1_models.csv")
    result_file_channel = os.path.join(main_folder, "experiments_2_channel.csv")
    result_file_normalize = os.path.join(main_folder, "experiments_3_normalize.csv")
    result_file_diameter = os.path.join(main_folder, "experiments_4_diameter.csv")
    result_file_cellprob_threshold = os.path.join(main_folder, "experiments_5_cellprob_threshold.csv")
    result_file_min_size = os.path.join(main_folder, "experiments_6_min_size.csv")
    result_file_stitch_threshold = os.path.join(main_folder, "experiments_7_stitch_threshold.csv")
    result_file_tile = os.path.join(main_folder, "experiments_8_tile.csv")
    result_file_smoothing = os.path.join(main_folder, "experiments_9_smoothing.csv")

    plot_eval(result_file_models)
    plot_eval(result_file_normalize)
    plot_eval(result_file_channel)
    plot_eval(result_file_diameter)
    plot_eval(result_file_cellprob_threshold)
    plot_eval(result_file_min_size)
    plot_eval(result_file_stitch_threshold)
    plot_eval(result_file_tile)
    plot_eval(result_file_smoothing)

    all_results = [
        result_file_models,
        result_file_normalize,
        result_file_channel,
        result_file_diameter,
        result_file_cellprob_threshold,
        result_file_min_size,
        result_file_stitch_threshold,
        result_file_tile,
        result_file_smoothing,
    ]

    plot_eval(all_results)
