import argparse
import json
import logging
import os
import time

import napari

from cellpose_adapt import core, caching
from cellpose_adapt import io
from cellpose_adapt.config.model_config import ModelConfig
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
        help="Path to the final pipeline configuration JSON file (e.g., best_cfg.json)."
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
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    setup_logging(log_level=logging.INFO, log_file=f"visualization_{timestamp}.log")


    # --- 2. Load Configurations ---
    if not os.path.exists(args.config_path):
        logging.error(f"Pipeline config file not found at {args.config_path}")
        return

    cfg = ModelConfig.from_json(args.config_path)
    # print(f"Loaded configuration from {args.config_path}:\n{cfg}")

    with open(args.project_config, 'r') as f:
        project_cfg = json.load(f)

    gt_mapping = project_cfg.get('gt_mapping')
    project_settings = project_cfg.get('project_settings', {})
    config_device = project_settings.get('device')
    cache_dir = caching.get_cache_dir(project_settings)

    # --- Determine Device ---
    device = get_device(cli_device=args.device, config_device=config_device)

    # --- Load Data ---
    gt_path = io.find_gt_path(args.image_path, gt_mapping) if gt_mapping else None
    image_segment, ground_truth, image = io.load_image_with_gt(args.image_path, gt_path, channel_to_segment=cfg.channel_to_segment)
    if image_segment is None:
        logging.error(f"Failed to load image from {args.image_path}")
        return
    if gt_path and ground_truth is None:
        logging.warning(f"Ground truth was not found at the inferred path: {gt_path}")

    # --- 3. Initialize Model and Run Pipeline ---
    logging.info(f"Initializing model '{cfg.model_name}' on device '{device}'...")
    model = core.initialize_model(cfg.model_name, device=device)

    runner = core.CellposeRunner(model, cfg, device=device, cache_dir=cache_dir)

    logging.info("Running segmentation on the image...")
    masks, duration = runner.run(image_segment)
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