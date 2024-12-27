from cellpose_adapt.utils import set_all_seeds
from experiments import setup


def main():
    # Set the random seed based on the image index
    set_all_seeds(42)
    main_folder = "results/P013T/"

    setup.run_model_experiments(main_folder)
    setup.run_channel_experiments(main_folder)
    setup.run_normalization_experiments(main_folder)
    setup.run_diameter_experiments(main_folder)
    setup.run_cellprob_threshold_experiments(main_folder)
    setup.run_min_size_experiments(main_folder)
    setup.run_stitch_threshold_experiments(main_folder)
    setup.run_tile_overlap_experiments(main_folder)

if __name__ == "__main__":
    main()
