import argparse
import os

import napari
import numpy as np

from .file_io import read_yaml, load_image_with_gt
from .core import evaluate_model
from .viz import extract_cellpose_video, plot_intensity


def cellpose_eval(
        image_path,
        ground_truth_path,
        param_file,
        output_dir,
        cache_dir=".cache",
        show_gt=True,
        show_prediction=False,
        video_3d=True,
        show_viewer = True,
        export_video = False,
        only_std_out = False,
        type="Nuclei", # Nuclei or Membranes
):

    params = read_yaml(param_file)
    params.type = type

    image_orig, ground_truth = load_image_with_gt(image_path, ground_truth_path)

    if image_orig is None:
        raise ValueError(f"Image not found: {image_path}")

    if ground_truth is None:
        raise ValueError(f"Ground truth not found: {ground_truth_path}")

    if type == "Nuclei":
        channel_idx = 0
    elif type == "Membranes":
        channel_idx = 1
    else:
        raise ValueError(f"Invalid type: {params.type}")

    # Get the right channel
    image = image_orig[:, channel_idx, :, :] if image_orig.ndim == 4 else image_orig
    image_name = os.path.basename(image_path).replace(".tif", "")

    # # Enhance the contrast of the image
    # image = enhance_contrast(image, ball(1))

    # q1, q3 = np.percentile(image, [1, 99.5])
    # image = np.clip(image, q1, q3)

    # plot the intensity distribution of the image
    # plot_intensity(image)

    if show_prediction or only_std_out:
        results = evaluate_model(image_name, image, ground_truth, params, cache_dir)

        if results is None:
            print(f"Failed to process image {image_path}")
        else:
            masks = results['masks']

            num_regions_mask = len(np.unique(masks)) - 1
            print(f"\tNumber of regions mask: {num_regions_mask}")
            num_regions_gt = len(np.unique(ground_truth)) - 1
            print(f"\tNumber of regions gt: {num_regions_gt}")

        if only_std_out:
            return


    # Initialize the Napari viewer
    viewer = napari.Viewer()

    # get the interquartile range of the intensities
    q1, q3 = np.percentile(image, [5, 99])

    # Add the image to the viewer
    viewer.add_image(
        image,
        # contrast_limits=[q1, q3],
        name='Organoids',
        colormap='gray',
    )

    if show_gt and (ground_truth is not None):
        layer = viewer.add_labels(
            ground_truth,
            name="Ground truth",
            opacity=0.7,
            blending='translucent',
            # colormap='magma',
        )
        # layer.contour = 2

    if show_prediction:
            # Add the labels to the viewer
            layer = viewer.add_labels(
                masks,
                name=params.model_name,
                opacity=0.7,
                blending='translucent',
                # colormap='magma',
            )
            layer.contour = 2

    if export_video:
        # Save the animation
        video_filename = f"{image_name}_{type}.mp4"

        if show_prediction:
            model_name = params.model_name
            video_filename = video_filename.replace(".mp4", f"_{model_name}.mp4")

        if show_gt:
            video_filename = video_filename.replace(".mp4", "_GT.mp4")

        if video_3d:
            video_filename = video_filename.replace(".mp4", "_3D.mp4")

        mode = "3D" if video_3d else "2D"
        num_z_slices = image.shape[0]
        extract_cellpose_video(viewer, output_dir, video_filename, num_z_slices, mode=mode)

        # Close the viewer to release resources
        viewer.close()

    if show_viewer:
        # setting the viewer to the center of the image
        center = image.shape[0] // 2
        viewer.dims.set_point(0, center)

        napari.run()


if __name__ == "__main__":

    # python main.py path/to/image.tif path/to/params.yaml path/to/output --show_gt --show_prediction --show_viewer

    parser = argparse.ArgumentParser(description="View cellpose results with Napari.")
    parser.add_argument("--image_path", type=str, default="data/P013T/20240305_P013T_40xSil_Hoechst_SiRActin/images_cropped_isotropic/20240305_P013T_A003_cropped_isotropic.tif", help="Path to the image file.")
    parser.add_argument("--ground_truth_path", type=str, default="data/P013T/20240305_P013T_40xSil_Hoechst_SiRActin/labelmaps/Nuclei/20240305_P013T_A003_cropped_isotropic_nuclei-labels.tif", help="Path to the ground truth file.")
    parser.add_argument("--type", type=str, default="Nuclei", help="Membranes or Nuclei.")
    parser.add_argument("--param_file", type=str, default="data/P013T/20240305_P013T_A003_cropped_isotropic_Nuclei_config.yaml", help="Path to the parameter YAML file.")
    parser.add_argument("--output_dir", type=str, default="Segmentation", help="Directory to save the output.")
    parser.add_argument("--cache_dir", type=str, default=".cache", help="Directory for cache files.")
    parser.add_argument("--show_gt", action="store_true", help="Show ground truth labels.")
    parser.add_argument("--show_prediction", action="store_true", help="Show prediction labels.")
    parser.add_argument("--video_3d", action="store_true", help="Export 3D video.")
    parser.add_argument("--show_viewer", action="store_true", help="Show Napari viewer.")
    parser.add_argument("--export_video", action="store_true", help="Export video.")

    args = parser.parse_args()

    cellpose_eval(
        image_path=args.image_path,
        ground_truth_path=args.ground_truth_path,
        param_file=args.param_file,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        show_gt=args.show_gt,
        show_prediction=args.show_prediction,
        video_3d=args.video_3d,
        show_viewer=args.show_viewer,
        export_video=args.export_video,
        type=args.type,
    )
