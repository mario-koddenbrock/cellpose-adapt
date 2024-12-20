import os

from experiments.results import print_best_config_per_image
from cellpose_adapt import viz


def plot_eval(result_path="data/P013T/results.csv"):
    viz.plot_aggregated_metric_variation(result_path, boxplot=True)
    viz.plot_aggregated_metric_variation(result_path, boxplot=False)
    viz.plot_best_scores_barplot(result_path, output_file=result_path.replace('.csv', '_best_score.png'))



if __name__ == "__main__":
    main_folder = "data/P013T/"

    # get all the subfolder
    result_path = os.path.join(main_folder, "results.csv")

    if not os.path.exists(result_path):
        print("No results to display.")
        exit()

    print_best_config_per_image(result_path)
    plot_eval(result_path)
