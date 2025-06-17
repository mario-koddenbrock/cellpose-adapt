from experiments.experiments_3d import (
    run_model_experiments,
    run_normalization_experiments,
    run_diameter_experiments,
    run_cellprob_threshold_experiments,
    run_min_size_experiments,
    run_stitch_threshold_experiments,
    run_tile_experiments,
    run_smoothing_experiments,
)


def main():

    run_model_experiments()
    # run_channel_experiments()
    run_normalization_experiments()
    run_diameter_experiments()
    run_cellprob_threshold_experiments()
    run_min_size_experiments()
    run_stitch_threshold_experiments()
    run_tile_experiments()
    run_smoothing_experiments()


if __name__ == "__main__":
    main()
