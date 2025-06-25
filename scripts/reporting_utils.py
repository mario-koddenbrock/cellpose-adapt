import logging
import os

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from cellpose_adapt import io, core
from cellpose_adapt.metrics import calculate_segmentation_stats
from cellpose_adapt.plotting.plotting_utils import prepare_3d_slice_for_display, create_opencv_overlay, \
    generate_comparison_panel

logger = logging.getLogger(__name__)


def generate_visual_and_quantitative_report(
        pipeline_config,
        project_config_data,
        plotting_config,
        results_dir,
        device,
        config_filename="config.json",
        show_panels=False,
):
    """
    Generates visual and quantitative reports for a given configuration.

    Args:
        pipeline_config (PipelineConfig): The configuration for the Cellpose model.
        project_config_data (dict): The loaded project configuration JSON data.
        plotting_config (PlottingConfig): The configuration for visual outputs.
        results_dir (str): The directory to save the report files.
        device (torch.device): The device to run the model on.
        config_filename (str): The name for the config file saved in the results dir.
        show_panels (bool): If True, displays each generated panel in an OpenCV window.
    """
    logging.info(f"Generating visual and quantitative reports in: {results_dir}")

    # Save a copy of the config inside the results folder for full reproducibility
    pipeline_config.to_json(os.path.join(results_dir, config_filename))

    # Load data
    data_pairs = io.find_image_gt_pairs(
        project_config_data['data_sources'],
        project_config_data.get('gt_mapping', None),
        project_config_data['project_settings'].get('limit_images_per_source', None)
    )
    if not data_pairs:
        logging.error("No data pairs found. Cannot generate report.")
        return

    # Initialize model and process images
    model = core.initialize_model(pipeline_config.model_name, device)
    runner = core.CellposeRunner(model, pipeline_config, device)
    channel_to_segment = pipeline_config.channel_to_segment

    report_data = []
    for image_path, gt_path in data_pairs:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        logging.info(f"  - Processing image: {base_name}")
        image, ground_truth, _ = io.load_image_with_gt(image_path, gt_path, channel_to_segment)

        if image is None:
            continue

        pred_mask, _ = runner.run(image)
        if pred_mask is None and ground_truth is not None:
            pred_mask = np.zeros_like(ground_truth)
        elif pred_mask is None:
            logging.warning(f"Prediction failed and no ground truth available for image {base_name}")
            continue

        if ground_truth is not None:
            stats = calculate_segmentation_stats(ground_truth, pred_mask)
            report_data.append({'image_name': base_name, **stats})

        # Prepare a 2D slice for visualization if data is 3D
        if image.ndim == 4:
            display_image = prepare_3d_slice_for_display(image)
            # Ensure GT/Pred are also sliced if they exist
            mid_slice_idx = (ground_truth.shape[0] // 2) if ground_truth is not None else (pred_mask.shape[0] // 2)
            display_gt = ground_truth[mid_slice_idx, :, :] if ground_truth is not None else None
            display_pred = pred_mask[mid_slice_idx, :, :]
        else:
            display_image = image
            display_gt = ground_truth
            display_pred = pred_mask

        overlay = create_opencv_overlay(display_image, display_gt, display_pred, plotting_config)
        panel = generate_comparison_panel(display_image, overlay, plotting_config)

        # Save the panel to disk
        cv2.imwrite(os.path.join(results_dir, f"{base_name}_report.png"), panel)

        # --- NEW: Display the panel if requested ---
        if show_panels:
            plt.imshow(panel)
            plt.axis('off')
            plt.show()

    # --- Clean up windows after the loop ---
    if show_panels:
        plt.close()
        logging.info("Closed all display windows.")

    # Save stats to CSV
    if report_data:
        report_df = pd.DataFrame(report_data)
        report_df.to_csv(os.path.join(results_dir, '_report_summary.csv'), index=False, float_format='%.4f')
        logging.info(f"Quantitative report saved. Mean F1-score: {report_df['f1_score'].mean():.4f}")
    else:
        logging.warning("No images with ground truth were processed for the quantitative report.")