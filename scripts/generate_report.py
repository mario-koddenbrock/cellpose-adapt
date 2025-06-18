import argparse
import json
import logging
import os

import cv2
import numpy as np
import optuna

from cellpose_adapt import io, core, config
from cellpose_adapt.logging_config import setup_logging


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Normalizes an image of any dtype to a displayable BGR uint8 image using cv2.normalize.
    """
    # Use cv2.normalize to perform min-max scaling to the 0-255 range and convert to uint8
    img_norm = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Convert to 3-channel BGR if it's grayscale, as this is the expected format for drawing/stacking
    if img_norm.ndim == 2:
        img_norm = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)

    return img_norm

def create_opencv_overlay(image: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """
    Creates an overlay image using OpenCV to draw contours of the GT and prediction masks.
    """
    # Create a clean, normalized BGR image to draw on
    overlay = normalize_to_uint8(image)

    # Ensure the array is C-contiguous for OpenCV drawing functions
    overlay = np.ascontiguousarray(overlay, dtype=np.uint8)

    # --- Ground Truth Contours (Green) ---
    for label in np.unique(gt_mask):
        if label == 0: continue
        binary_mask = np.uint8(gt_mask == label)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1) # Green

    # --- Prediction Contours (Red) ---
    for label in np.unique(pred_mask):
        if label == 0: continue
        binary_mask = np.uint8(pred_mask == label)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1) # Red (BGR)

    return overlay

def generate_comparison_panel(image: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Stitches the original image and the overlay side-by-side."""

    # Create a clean, normalized BGR image for the left panel
    image_display = normalize_to_uint8(image)

    # Add labels to the display images
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image_display, 'Original Image', (10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(overlay, 'GT (Green) vs Pred (Red)', (10, 30), font, 0.8, (255, 255, 255), 2)

    return np.hstack((image_display, overlay))

def main():
    # ... (The main function is completely unchanged) ...
    parser = argparse.ArgumentParser(description="Generate visual reports from the best Optuna trial.")
    parser.add_argument("--study_db", type=str, required=True, help="Path to the Optuna study SQLite DB.")
    parser.add_argument("--project_config", type=str, required=True, help="Path to the original project JSON config file.")
    parser.add_argument("--output_dir", type=str, default="reports/", help="Directory to save the report images.")
    args = parser.parse_args()

    setup_logging(log_file="report_generation.log")

    if not os.path.exists(args.study_db):
        logging.error("Study database not found at %s", args.study_db)
        return

    study = optuna.load_study(study_name=None, storage=f"sqlite:///{args.study_db}")
    best_trial = study.best_trial

    logging.info("Loaded best trial #%d with Jaccard score: %.4f", best_trial.number, best_trial.value)

    best_config = config.PipelineConfig()
    try:
        project_cfg_data = json.load(open(args.project_config, 'r'))
        search_space_config_path = project_cfg_data['search_space_config_path']
        search_space_config = json.load(open(search_space_config_path, 'r'))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error("Failed to load project or search space config: %s", e)
        return

    final_params = search_space_config.get("fixed_params", {})
    final_params.update(best_trial.params)

    for key, value in final_params.items():
        if hasattr(best_config, key):
            setattr(best_config, key, value)

    data_sources = project_cfg_data['data_sources']
    gt_mapping = project_cfg_data['gt_mapping']
    limit = project_cfg_data['project_settings'].get('limit_images_per_source')
    data_pairs = io.find_image_gt_pairs(data_sources, gt_mapping, limit)

    if not data_pairs:
        logging.error("No data pairs found. Cannot generate report.")
        return

    logging.info("Initializing model with best configuration...")
    model = core.initialize_model(best_config.model_name)
    runner = core.CellposeRunner(model, best_config)

    os.makedirs(args.output_dir, exist_ok=True)

    num_processed = 0
    for image_path, gt_path in data_pairs:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        logging.info("Processing %s", base_name)

        image, ground_truth = io.load_image_with_gt(image_path, gt_path)
        if image is None or ground_truth is None:
            logging.warning("Skipping non-2D image or missing data: %s", image_path)
            continue

        pred_mask, _ = runner.run(image)
        if pred_mask is None:
            pred_mask = np.zeros_like(ground_truth)

        overlay = create_opencv_overlay(image, ground_truth, pred_mask)
        panel = generate_comparison_panel(image, overlay)

        output_filename = f"{base_name}_report.png"
        output_path = os.path.join(args.output_dir, output_filename)

        cv2.imwrite(output_path, panel)
        logging.info("Saved report to %s", output_path)
        num_processed += 1

    if num_processed == 0:
        logging.error("No valid 2D images were processed. No reports were generated.")
    else:
        logging.info("Successfully generated %d individual reports in directory: %s", num_processed, args.output_dir)


if __name__ == "__main__":
    main()