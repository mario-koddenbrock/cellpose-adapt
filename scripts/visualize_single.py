import argparse
import json
import logging
import os

import napari

from cellpose_adapt import io, core, config
from cellpose_adapt.logging_config import setup_logging
from cellpose_adapt.metrics import calculate_segmentation_stats
from cellpose_adapt.utils import get_device


def main():
    parser = argparse.ArgumentParser(
        description="Run and visualize Cellpose results for a single image."
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the final pipeline configuration JSON file (e.g., best_config.json)."
    )
    parser.add_argument(
        "--project_config",
        type=str,
        required=True,
        help="Path to the project config file to find GT mapping and device settings."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device setting ('cpu', 'cuda', 'mps')."
    )
    args = parser.parse_args()

    # --- 1. Setup ---
    setup_logging(log_level=logging.INFO, log_file="visualization.log")

    # --- 2. Load Configurations ---
    if not os.path.exists(args.config_path):
        logging.error(f"Pipeline config file not found at {args.config_path}")
        return
    pipeline_config = config.PipelineConfig.from_json(args.config_path)

    try:
        with open(args.project_config, 'r') as f:
            project_cfg_data = json.load(f)
        gt_mapping = project_cfg_data.get('gt_mapping')
        config_device = project_cfg_data.get('project_settings', {}).get('device')
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.warning(f"Could not load project config: {e}. GT/Device settings may be unavailable.")
        gt_mapping = None
        config_device = None

    # --- Determine Device ---
    device = get_device(cli_device=args.device, config_device=config_device)

    # --- Load Data ---
    gt_path = io._find_gt_path(args.image_path, gt_mapping) if gt_mapping else None
    image, ground_truth = io.load_image_with_gt(args.image_path, gt_path)
    if image is None:
        logging.error(f"Failed to load image from {args.image_path}")
        return
    if gt_path and ground_truth is None:
        logging.warning(f"Ground truth was not found at the inferred path: {gt_path}")

    # --- 3. Initialize Model and Run Pipeline ---
    logging.info(f"Initializing model '{pipeline_config.model_name}' on device '{device}'...")
    model = core.initialize_model(pipeline_config.model_name, device=device)

    runner = core.CellposeRunner(model, pipeline_config, device=device)

    logging.info("Running segmentation on the image...")
    masks, duration = runner.run(image)
    logging.info(f"Segmentation completed in {duration:.2f} seconds.")

    # --- 4. Evaluate and Launch Napari Viewer ---
    metrics = {}
    if ground_truth is not None and masks is not None:
        metrics = calculate_segmentation_stats(ground_truth, masks)
        logging.info(f"Performance (full data): F1={metrics.get('f1_score', 0):.3f}, P={metrics.get('precision', 0):.3f}, R={metrics.get('recall', 0):.3f}")
    viewer = napari.Viewer(title="Cellpose Single Image Visualization")
    is_3d = image.ndim == 4
    if is_3d:
        mid_slice = image.shape[0] // 2
        viewer.dims.set_point(0, mid_slice)
        viewer.add_image(image[:, 0], name="Channel 1", colormap="cyan")
        viewer.add_image(image[:, 1], name="Channel 2", colormap="magenta")
    else:
        viewer.add_image(image, name="Image")
    if ground_truth is not None:
        viewer.add_labels(ground_truth, name="Ground Truth", opacity=0.5)
    if masks is not None:
        f1_score = metrics.get('f1_score', 0.0)
        mask_name = f"Prediction (F1={f1_score:.2f})"
        viewer.add_labels(masks, name=mask_name, opacity=0.7)
    logging.info("Launching Napari viewer. Close the window to exit.")
    napari.run()

if __name__ == "__main__":
    main()