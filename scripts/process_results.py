import argparse
import json
import logging
import os

import cv2
import numpy as np
import optuna
import pandas as pd
import torch
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice

from cellpose_adapt import io, core, config
from cellpose_adapt.logging_config import setup_logging
from cellpose_adapt.metrics import calculate_segmentation_stats


def get_device(device_str: str = None) -> torch.device:
    """Determines the torch device, either from config or by auto-detection."""
    if device_str:
        logging.info(f"Using device specified in config: '{device_str}'")
        return torch.device(device_str)
    if torch.cuda.is_available(): device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): device = torch.device("mps")
    else: device = torch.device("cpu")
    logging.info(f"No device specified, auto-detected: '{device}'")
    return device

def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    img_norm = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    if img_norm.ndim == 2:
        img_norm = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
    return img_norm

def create_opencv_overlay(image: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
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
    image_display = normalize_to_uint8(image)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image_display, 'Original Image', (10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(overlay, 'GT (Green) vs Pred (Red)', (10, 30), font, 0.8, (255, 255, 255), 2)
    return np.hstack((image_display, overlay))

def main():
    parser = argparse.ArgumentParser(description="Analyze a study and generate all reports.")
    parser.add_argument("--study_db", type=str, required=True, help="Path to the Optuna study SQLite DB.")
    parser.add_argument("--project_config", type=str, required=True, help="Path to the original project JSON config file.")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating HTML analysis plots.")
    parser.add_argument("--no-report", action="store_true", help="Skip generating visual/quantitative reports.")
    args = parser.parse_args()

    setup_logging(log_level=logging.INFO, log_file="processing.log")

    # --- 1. Load Study and Best Trial ---
    if not os.path.exists(args.study_db):
        logging.error("Study database not found at %s", args.study_db)
        return
    study = optuna.load_study(study_name=None, storage=f"sqlite:///{args.study_db}")
    best_trial = study.best_trial
    study_name = study.study_name

    logging.info(f"--- Processing results for study: {study_name} ---")
    logging.info(f"Best trial #{best_trial.number} with score: {best_trial.value:.4f}")

    # --- 2. Create Output Directories ---
    plots_dir = os.path.join("reports", f"{study_name}_analysis_plots")
    results_dir = os.path.join("reports", f"{study_name}_results")
    if not args.no_plots: os.makedirs(plots_dir, exist_ok=True)
    if not args.no_report: os.makedirs(results_dir, exist_ok=True)
    os.makedirs("configs", exist_ok=True)

    # --- 3. Create and Save Best Config ---
    best_config = config.PipelineConfig()
    try:
        project_cfg_data = json.load(open(args.project_config, 'r'))
        search_space_config_path = project_cfg_data['search_space_config_path']
        search_space_config = json.load(open(search_space_config_path, 'r'))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Failed to load project or search space config: {e}")
        return

    final_params = search_space_config.get("fixed_params", {})
    final_params.update(best_trial.params)
    for key, value in final_params.items():
        if hasattr(best_config, key):
            setattr(best_config, key, value)

    output_config_path = os.path.join("configs", f"best_{study_name}_config.json")
    best_config.to_json(output_config_path)
    logging.info(f"Best configuration saved to: {output_config_path}")

    # --- 4. Generate Analysis Plots ---
    if not args.no_plots:
        logging.info(f"Generating analysis plots in: {plots_dir}")
        try:
            plot_optimization_history(study).write_html(os.path.join(plots_dir, "optimization_history.html"))
            plot_param_importances(study).write_html(os.path.join(plots_dir, "param_importances.html"))
            plot_slice(study).write_html(os.path.join(plots_dir, "slice_plot.html"))
            logging.info("Analysis plots saved successfully.")
        except (ValueError, ImportError) as e:
            logging.warning(f"Could not generate plots: {e}")

    # --- 5. Generate Visual and Quantitative Reports ---
    if not args.no_report:
        logging.info(f"Generating visual and quantitative reports in: {results_dir}")

        # Save a copy of the best config inside the results folder for full reproducibility
        best_config.to_json(os.path.join(results_dir, 'best_config.json'))

        # Load data
        settings = project_cfg_data["project_settings"]
        data_pairs = io.find_image_gt_pairs(
            project_cfg_data['data_sources'],
            project_cfg_data['gt_mapping'],
            settings.get('limit_images_per_source')
        )
        if not data_pairs:
            logging.error("No data pairs found. Cannot generate report.")
            return

        # Initialize model and process images
        device = get_device(settings.get("device"))
        model = core.initialize_model(best_config.model_name, device)
        runner = core.CellposeRunner(model, best_config, device)

        report_data = []
        for image_path, gt_path in data_pairs:
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            logging.info(f"  - Processing image: {base_name}")
            image, ground_truth = io.load_image_with_gt(image_path, gt_path)
            if image is None or ground_truth is None: continue

            pred_mask, _ = runner.run(image)
            if pred_mask is None: pred_mask = np.zeros_like(ground_truth)

            stats = calculate_segmentation_stats(ground_truth, pred_mask)
            report_data.append({'image_name': base_name, **stats})

            panel = generate_comparison_panel(image, create_opencv_overlay(image, ground_truth, pred_mask))
            cv2.imwrite(os.path.join(results_dir, f"{base_name}_report.png"), panel)

        # Save stats to CSV
        if report_data:
            report_df = pd.DataFrame(report_data)
            report_df.to_csv(os.path.join(results_dir, '_report_summary.csv'), index=False, float_format='%.4f')
            logging.info(f"Quantitative report saved. Mean F1-score: {report_df['f1_score'].mean():.4f}")
        else:
            logging.warning("No images were processed for the report.")

    logging.info("--- Processing complete. ---")

if __name__ == "__main__":
    main()