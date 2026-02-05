import logging
import os

import cv2
import numpy as np
import pandas as pd
import tifffile as tiff
from matplotlib import pyplot as plt

from cellpose_adapt import caching, io
from cellpose_adapt.core import initialize_model, CellposeRunner
from cellpose_adapt.metrics import calculate_segmentation_stats
from cellpose_adapt.plotting.plotting_utils import prepare_3d_slice_for_display, create_opencv_overlay, \
    generate_comparison_panel
from cellpose_adapt.plotting.visualization import export_napari_video

logger = logging.getLogger(__name__)


def _process_single_image_and_get_masks(image_path, gt_path, runner, channel_to_segment):
    """Loads image/gt, runs prediction, and returns the original image and masks."""
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    logging.debug(f"  - Processing image: {base_name}")
    image, ground_truth, _ = io.load_image_with_gt(image_path, gt_path, channel_to_segment)

    if image is None:
        return base_name, None, None, None

    pred_mask, _ = runner.run(image)
    # pred_mask = np.zeros(image.shape, dtype=np.uint16)

    if pred_mask is None:
        if ground_truth is not None:
            pred_mask = np.zeros_like(ground_truth)  # Create empty mask for stats
        else:
            logging.warning(f"Prediction failed and no ground truth available for image {base_name}")
            return base_name, image, ground_truth, np.zeros_like(image, dtype=np.uint16)

    return base_name, image, ground_truth, pred_mask


def _generate_and_save_visual_panel(image, ground_truth, pred_mask, plotting_config, base_name, results_dir, show_panels, plot_original_image=False):
    """Generates and saves the visual comparison panel for a single image."""
    num_instances_gt = np.unique(ground_truth).size - 1 if ground_truth is not None else 0
    num_instances_pred = np.unique(pred_mask).size - 1

    # Prepare a 2D slice for visualization if data is 3D
    is_3d = image.ndim == 4 or (image.ndim == 3 and np.min(image.shape) > 3)
    display_image = prepare_3d_slice_for_display(image) if is_3d else image
    if ground_truth is not None:
        display_gt = prepare_3d_slice_for_display(ground_truth, is_mask=True) if is_3d else ground_truth
    else:
        display_gt = None
    display_pred = prepare_3d_slice_for_display(pred_mask, is_mask=True) if is_3d else pred_mask

    overlay = create_opencv_overlay(display_image, display_gt, display_pred, plotting_config)
    panel = generate_comparison_panel(display_image, overlay, plotting_config, num_instances_gt, num_instances_pred, plot_original_image)

    cv2.imwrite(os.path.join(results_dir, f"{base_name}_report.png"), panel)

    if show_panels:
        plt.imshow(panel)
        plt.axis('off')
        plt.title(f"{base_name} - {num_instances_pred} instances")
        plt.show()


def _save_quantitative_report(report_data, results_dir):
    """Saves the collected quantitative statistics to a CSV file."""
    if not report_data:
        logging.warning("No images with ground truth were processed for the quantitative report.")
        return

    report_df = pd.DataFrame(report_data)
    report_df.to_csv(os.path.join(results_dir, 'report_summary.csv'), index=False, float_format='%.4f')
    logging.info(f"Quantitative report saved. Mean Jaccard: {report_df['jaccard'].mean():.4f}")


def _export_3d_video_report(image, ground_truth, pred_mask, base_name, results_dir):
    """Exports a 3D video report if the image is 3D."""
    is_3d = image.ndim == 4 or (image.ndim == 3 and np.min(image.shape) > 3)
    if not is_3d:
        return

    logging.info(f"  - Generating 3D video for {base_name}")
    video_dir = os.path.join(results_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)
    export_napari_video(
        image=image,
        ground_truth=ground_truth,
        pred_mask=pred_mask,
        output_dir=video_dir,
        video_filename=f"{base_name}_3d_report.mp4"
    )


def generate_visual_and_quantitative_report(
         cfg,
         project_cfg,
         plotting_config,
         results_dir,
         device,
         config_filename="config.json",
         show_panels=False,
         export_video=False,
         plot_original_image=False,
         export_predicted_masks=False
 ):
    """
    Generates visual and quantitative reports for a given configuration.

    Args:
        cfg (ModelConfig): The configuration for the Cellpose model.
        project_cfg (dict): The loaded project configuration JSON data.
        plotting_config (PlottingConfig): The configuration for visual outputs.
        results_dir (str): The directory to save the report files.
        device (torch.device): The device to run the model on.
        config_filename (str): The name for the config file saved in the results dir.
        show_panels (bool): If True, displays each generated panel in an OpenCV window.
        export_video (bool): If True, exports a 3D video for 3D images.
        plot_original_image (bool): If True, plots the original image next to the overlay.
    """
    logging.info(f"Generating visual and quantitative reports in: {results_dir}")
    cfg.to_json(os.path.join(results_dir, config_filename))

    data_pairs = io.find_image_gt_pairs(
        project_cfg['data_sources'],
        project_cfg.get('gt_mapping', None),
        project_cfg['project_settings'].get('limit_images_per_source', None)
    )
    if not data_pairs:
        logging.error("No data pairs found. Cannot generate report.")
        return

    cache_dir = caching.get_cache_dir(project_cfg['project_settings'])
    model = initialize_model(cfg.model_name, device)
    runner = CellposeRunner(model, cfg, device, cache_dir=cache_dir)
    report_data = []

    # Prepare predicted masks output directory (if requested)
    predicted_masks_outdir = None
    if export_predicted_masks:
        predicted_masks_outdir = project_cfg['project_settings'].get('output_dir')
        if predicted_masks_outdir:
            os.makedirs(predicted_masks_outdir, exist_ok=True)
            logging.info(f"Predicted masks will be exported to: {predicted_masks_outdir}")
        else:
            logging.warning("export_predicted_masks=True but no 'output_dir' set in project_cfg['project_settings']; skipping mask export")

    for image_path, gt_path in data_pairs:

        # if "20241023_P021N_A004_cropped_isotropic" in image_path:
        #     cfg.channel_to_segment = 1

        base_name, image, ground_truth, pred_mask = _process_single_image_and_get_masks(
            image_path, gt_path, runner, cfg.channel_to_segment
        )

        if image is None or pred_mask is None:
            continue

        if ground_truth is not None:
            stats = calculate_segmentation_stats(ground_truth, pred_mask, iou_threshold=project_cfg['project_settings'].get('iou_threshold'))
            report_data.append({'image_name': base_name, **stats})
            logging.info(f"  - Processing image: {base_name} | Jaccard: {stats['jaccard']:.4f}")
        else:
            logging.info(f"  - Processing image: {base_name} | No ground truth available for quantitative metrics.")

        _generate_and_save_visual_panel(
            image, ground_truth, pred_mask, plotting_config, base_name, results_dir, show_panels, plot_original_image
        )

        # Export predicted mask to the project's output_dir if requested
        if export_predicted_masks and predicted_masks_outdir and pred_mask is not None:
            mask_path = os.path.join(predicted_masks_outdir, f"{base_name}_cellpose_mask.tif")
            try:
                # Use tifffile for robust multi-page/3D tiff writing
                tiff.imwrite(mask_path, np.asarray(pred_mask).astype(np.uint16))
                logging.debug(f"Saved predicted mask: {mask_path}")

                # # read the saved mask back to verify
                # reloaded_mask = tiff.imread(mask_path)
                # if not np.array_equal(reloaded_mask, pred_mask):
                #     logging.warning(f"Verification failed: reloaded mask does not match the original for {base_name}")

            except Exception as e:
                logging.warning(f"Failed to save predicted mask for {base_name}: {e}")

        if export_video:
            _export_3d_video_report(image, ground_truth, pred_mask, base_name, results_dir)

    if show_panels:
        plt.close('all')
        logging.info("Closed all display windows.")

    _save_quantitative_report(report_data, results_dir)
