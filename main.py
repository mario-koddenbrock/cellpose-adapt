import glob
import os

from cellpose_adapt.utils import set_all_seeds
from experiments.eval import plot_eval
from experiments.optimize import optimize_parameters

    # options = {
    #     "model_name": ["cyto2_cp3", "cyto", "cyto2", "cyto3", "nuclei", "tissuenet_cp3", "livecell_cp3", "yeast_PhC_cp3", "yeast_BF_cp3", "bact_phase_cp3", "bact_fluor_cp3", "deepbacs_cp3"],
    #     "channel_segment": [0],  # [0, 1, 2, 3]
    #     "channel_nuclei": [0],
    #     "channel_axis": [None],  # TODO
    #     "invert": [False],  # Dont do this
    #     "normalize": [True],  # Always do this
    #     "normalization_min": [0.5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #     "normalization_max": [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 99.5],
    #     "diameter": [10, 17, 30, 50, 70, 100],  # TODO good values (btw. None not working for 3D)
    #     "do_3D": [True],  # TODO try False too
    #     "flow_threshold": [0.1, 0.3, 0.5, 0.7],  # [0.3, 0.4, 0.5, 0.6]
    #     "cellprob_threshold": [0.0, 0.1, 0.2, 0.5],  # [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    #     "interp": [False],  # NOT AVAILABLE FOR 3D
    #     "min_size": [15, 25, 50],  # TODO
    #     "max_size_fraction": [0.5],  # TODO
    #     "niter": [100],  # TODO
    #     "stitch_threshold": [0.0],  # TODO
    #     "tile_overlap": [0.1],  # TODO
    #     "type": ["Nuclei", "Membranes"],
    # }

def main():
    # Set the random seed based on the image index
    set_all_seeds(42)
    main_folder = "data/P013T/"

    # get all the subfolder
    subfolders = glob.glob(os.path.join(main_folder, "*"))

    # Combine all image paths into one list
    image_paths = [image_path for folder in subfolders if os.path.isdir(folder) for image_path in
                   glob.glob(os.path.join(folder, "images_cropped_isotropic", "*.tif"))]

    # image_paths = [image_paths[5]]

    # Define the result files
    result_file_models = os.path.join(main_folder, "results_models.csv")
    result_file_normalize = os.path.join(main_folder, "results_normalize.csv")
    result_file_diameter = os.path.join(main_folder, "results_diameter.csv")
    result_file_flow_threshold = os.path.join(main_folder, "results_flow_threshold.csv")
    result_file_cellprob_threshold = os.path.join(main_folder, "results_cellprob_threshold.csv")
    result_file_min_size = os.path.join(main_folder, "results_min_size.csv")
    result_file_stitch_threshold = os.path.join(main_folder, "results_stitch_threshold.csv")
    result_file_tile_overlap = os.path.join(main_folder, "results_tile_overlap.csv")

    # TODO check if all images are on the server
    for image_idx, image_path in enumerate(image_paths):

        # TODO check for other models like "bact_omni"
        options_model = {
            "model_name": ["cyto2_cp3", "cyto", "cyto2", "cyto3", "nuclei", "tissuenet_cp3", "livecell_cp3",
                           "yeast_PhC_cp3", "yeast_BF_cp3", "bact_phase_cp3", "bact_fluor_cp3", "deepbacs_cp3"]}
        optimize_parameters(options_model, image_path, result_file_models, append_result=image_idx > 0)

        options_normalization = {"normalization_min": [0, 0.5, 1, 2, 3, 5, 7, 10], "normalize": [True, False]}
        optimize_parameters(options_normalization, image_path, result_file_normalize, append_result=image_idx > 0)

        options_normalization = {"normalization_max": [90, 93, 95, 97, 98, 99, 99.5, 100], "normalize": [True, False]}
        optimize_parameters(options_normalization, image_path, result_file_normalize, append_result=True)

        options_diameter = {"diameter": [5, 10, 12, 17, 30, 40, 50, 70, 100]}
        optimize_parameters(options_diameter, image_path, result_file_diameter, append_result=image_idx > 0)

        options_flow_threshold = {"flow_threshold": [0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]}
        optimize_parameters(options_flow_threshold, image_path, result_file_flow_threshold, append_result=image_idx > 0)

        options_cellprob_threshold = {"cellprob_threshold": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]}
        optimize_parameters(options_cellprob_threshold, image_path, result_file_cellprob_threshold,
                            append_result=image_idx > 0)

        options_min_size = {"min_size": [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]}
        optimize_parameters(options_min_size, image_path, result_file_min_size, append_result=image_idx > 0)

        options_stitch_threshold = {"stitch_threshold": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]}
        optimize_parameters(options_stitch_threshold, image_path, result_file_stitch_threshold,
                            append_result=image_idx > 0)

        options_tile_overlap = {"tile_overlap": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]}
        optimize_parameters(options_tile_overlap, image_path, result_file_tile_overlap, append_result=image_idx > 0)

    plot_eval(result_file_models)
    plot_eval(result_file_normalize)
    plot_eval(result_file_diameter)
    plot_eval(result_file_flow_threshold)
    plot_eval(result_file_cellprob_threshold)
    plot_eval(result_file_min_size)
    plot_eval(result_file_stitch_threshold)
    plot_eval(result_file_tile_overlap)

if __name__ == "__main__":
    main()
