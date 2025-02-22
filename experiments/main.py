from experiments import setup


def main():

    setup.run_model_experiments()
    # setup.run_channel_experiments()
    setup.run_normalization_experiments()
    setup.run_diameter_experiments()
    setup.run_cellprob_threshold_experiments()
    setup.run_min_size_experiments()
    setup.run_stitch_threshold_experiments()
    setup.run_tile_experiments()
    setup.run_smoothing_experiments()


if __name__ == "__main__":
    main()
