import os

from cellpose_adapt import viz
from experiments.results import save_best_config_per_image

if __name__ == "__main__":
    main_folder = "../results/P013T/"

    # Define the result files
    result_file_models = os.path.join(main_folder, "experiments_1_models.csv")
    result_file_channel = os.path.join(main_folder, "experiments_2_channel.csv")
    result_file_normalize = os.path.join(main_folder, "experiments_3_normalize.csv")
    result_file_diameter = os.path.join(main_folder, "experiments_4_diameter.csv")
    result_file_cellprob_threshold = os.path.join(
        main_folder, "experiments_5_cellprob_threshold.csv"
    )
    result_file_min_size = os.path.join(main_folder, "experiments_6_min_size.csv")
    result_file_stitch_threshold = os.path.join(
        main_folder, "experiments_7_stitch_threshold.csv"
    )
    result_file_tile = os.path.join(main_folder, "experiments_8_tile.csv")
    result_file_smoothing = os.path.join(main_folder, "experiments_9_smoothing.csv")

    viz.plot_aggregated_metric_variation(result_file_models, boxplot=True)
    viz.plot_aggregated_metric_variation(result_file_normalize)
    viz.plot_aggregated_metric_variation(result_file_channel, boxplot=True)
    viz.plot_aggregated_metric_variation(result_file_diameter)
    viz.plot_aggregated_metric_variation(result_file_cellprob_threshold)
    viz.plot_aggregated_metric_variation(result_file_min_size)
    viz.plot_aggregated_metric_variation(result_file_stitch_threshold)
    viz.plot_aggregated_metric_variation(result_file_tile)
    viz.plot_aggregated_metric_variation(result_file_smoothing)

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

    # viz.plot_aggregated_metric_variation(all_results)
    viz.plot_best_scores_barplot(all_results, output_file="best_scores.png")

    save_best_config_per_image(all_results)
