import argparse
import json
import logging
import os
import time

import cv2
import numpy as np
import optuna
import pandas as pd

from cellpose_adapt import io, core, config
from cellpose_adapt.logging_config import setup_logging
from cellpose_adapt.metrics import calculate_segmentation_stats
from scripts.run_optimization import get_device


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalizes an image of any dtype to a displayable BGR uint8 image using cv2.normalize."""
    img_norm = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    if img_norm.ndim == 2:
        img_norm = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
    return img_norm

def create_opencv_overlay(image: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """Creates an overlay image using OpenCV to draw contours."""
    overlay = normalize_to_uint8(image)
    overlay = np.ascontiguousarray(overlay, dtype=np.uint8)
    for label in np.unique(gt_mask):
        if label == 0: continue
        binary_mask = np.uint8(gt_mask == label)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
    for label in np.unique(pred_mask):
        if label == 0: continue
        binary_mask = np.uint8(pred_mask == label)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), 1)
    return overlay

def generate_comparison_panel(image: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """Stitches the original image and the overlay side-by-side."""
    image_display = normalize_to_uint8(image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image_display, 'Original Image', (10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(overlay, 'GT (Green) vs Pred (Red)', (10, 30), font, 0.8, (255, 255, 255), 2)
    return np.hstack((image_display, overlay))


def main():
    parser = argparse.ArgumentParser(description="Generate visual and statistical reports from the best Optuna trial.")
    parser.add_argument("--study_db", type=str, required=True, help="Path to the Optuna study SQLite DB.")
    parser.add_argument("--project_config", type=str, required=True, help="Path to the original project JSON config file.")
    parser.add_argument("--output_dir", type=str, default="reports/", help="Directory to save the report images and stats.")
    args = parser.parse_args()

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    setup_logging(log_level=logging.INFO, log_file=f"report_{timestamp}.log")

    # --- 1. Load Best Config ---
    if not os.path.exists(args.study_db):
        logging.error("Study database not found at %s", args.study_db)
        return
    study = optuna.load_study(study_name=None, storage=f"sqlite:///{args.study_db}")
    best_trial = study.best_trial
    logging.info("Loaded best trial #%d with score: %.4f", best_trial.number, best_trial.value)
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

    os.makedirs(args.output_dir, exist_ok=True)
    best_config_path = os.path.join(args.output_dir, 'best_config.json')
    best_config.to_json(best_config_path)
    logging.info("Saved best configuration to %s", best_config_path)

    # --- 2. Load Data ---
    data_sources = project_cfg_data['data_sources']
    gt_mapping = project_cfg_data['gt_mapping']
    settings = project_cfg_data["project_settings"]
    limit = settings.get('limit_images_per_source')
    data_pairs = io.find_image_gt_pairs(data_sources, gt_mapping, limit)
    if not data_pairs:
        logging.error("No data pairs found. Cannot generate report.")
        return

    # --- 3. Initialize Model and Process Images ---
    logging.info("Initializing model and generating reports...")
    device = get_device(settings.get("device"))
    model = core.initialize_model(best_config.model_name, device)
    runner = core.CellposeRunner(model, best_config, device)

    report_data = []

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

        # --- Collect Statistics using F1-score ---
        stats = calculate_segmentation_stats(ground_truth, pred_mask)

        row_data = {
            'image_name': base_name,
            'f1_score': stats['f1_score'],
            'precision': stats['precision'],
            'recall': stats['recall'],
            'true_positives': stats['tp'],
            'false_positives': stats['fp'],
            'false_negatives': stats['fn'],
        }
        report_data.append(row_data)
        logging.info(
            f"  - F1: {stats['f1_score']:.3f}, Precision: {stats['precision']:.3f}, Recall: {stats['recall']:.3f}"
        )

        # --- Generate Visual Report ---
        overlay = create_opencv_overlay(image, ground_truth, pred_mask)
        panel = generate_comparison_panel(image, overlay)
        output_filename = f"{base_name}_report.png"
        output_path = os.path.join(args.output_dir, output_filename)
        cv2.imwrite(output_path, panel)

    # --- 4. Save Final Statistics Report ---
    if not report_data:
        logging.error("No valid 2D images were processed. No reports were generated.")
        return

    report_df = pd.DataFrame(report_data)
    csv_path = os.path.join(args.output_dir, '_report_summary.csv')
    report_df.to_csv(csv_path, index=False, float_format='%.4f')
    logging.info("Saved summary statistics to %s", csv_path)

    mean_f1 = report_df['f1_score'].mean()
    logging.info("Mean F1-score across all reported images: %.4f", mean_f1)


if __name__ == "__main__":
    main()