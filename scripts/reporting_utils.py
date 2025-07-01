import logging
import os

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from cellpose_adapt import io
from cellpose_adapt.core import initialize_model, CellposeRunner
from cellpose_adapt.metrics import calculate_segmentation_stats
from cellpose_adapt.plotting.plotting_utils import prepare_3d_slice_for_display, create_opencv_overlay, \
    generate_comparison_panel

logger = logging.getLogger(__name__)


def generate_visual_and_quantitative_report(
        cfg,
        project_cfg,
        plotting_config,
        results_dir,
        device,
        config_filename="config.json",
        show_panels=False,
):
    """
    Generates visual and quantitative reports for a given configuration.

    Args:
        cfg (PipelineConfig): The configuration for the Cellpose model.
        project_cfg (dict): The loaded project configuration JSON data.
        plotting_config (PlottingConfig): The configuration for visual outputs.
        results_dir (str): The directory to save the report files.
        device (torch.device): The device to run the model on.
        config_filename (str): The name for the config file saved in the results dir.
        show_panels (bool): If True, displays each generated panel in an OpenCV window.
    """
    logging.info(f"Generating visual and quantitative reports in: {results_dir}")

    # Save a copy of the config inside the results folder for full reproducibility
    cfg.to_json(os.path.join(results_dir, config_filename))

    # Load data
    data_pairs = io.find_image_gt_pairs(
        project_cfg['data_sources'],
        project_cfg.get('gt_mapping', None),
        project_cfg['project_settings'].get('limit_images_per_source', None)
    )
    if not data_pairs:
        logging.error("No data pairs found. Cannot generate report.")
        return

    # Initialize model and process images
    model = initialize_model(cfg.model_name, device)
    runner = CellposeRunner(model, cfg, device)
    channel_to_segment = cfg.channel_to_segment

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
        if image.ndim == 4 or (image.ndim == 3 and np.min(image.shape) > 3):
            display_image = prepare_3d_slice_for_display(image)
            display_gt = prepare_3d_slice_for_display(ground_truth, True)
            display_pred = prepare_3d_slice_for_display(pred_mask, True)
        else:
            display_image = image
            display_gt = ground_truth
            display_pred = pred_mask

        overlay = create_opencv_overlay(display_image, display_gt, display_pred, plotting_config)
        panel = generate_comparison_panel(display_image, overlay, plotting_config)

        # Save the panel to disk
        cv2.imwrite(os.path.join(results_dir, f"{base_name}_report.png"), panel)

        # ---
        # Display the panel if requested ---
        if show_panels:
            num_instances = np.unique(display_pred).size - 1
            plt.imshow(panel)
            plt.axis('off')
            plt.title(f"{base_name} - {num_instances} instances")
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