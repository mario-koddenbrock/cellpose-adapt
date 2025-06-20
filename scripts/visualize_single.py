import argparse
import logging
import os

import napari
import torch

from cellpose_adapt import io, core, visualization, config
from cellpose_adapt.logging_config import setup_logging
from cellpose_adapt.metrics import calculate_segmentation_stats


def main():
    parser = argparse.ArgumentParser(
        description="Run and visualize Cellpose results with a specific configuration."
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the input image."
    )
    parser.add_argument(
        "--gt_path", type=str, help="Optional: Path to the ground truth mask."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the final pipeline configuration JSON file (e.g., best_config.json)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use ('cpu', 'cuda', 'mps'). If None, will auto-detect."
    )
    args = parser.parse_args()

    # --- 1. Setup Logging and Device ---
    setup_logging(log_level=logging.INFO, log_file="visualization.log")

    if args.device:
        device = torch.device(args.device)
    else:
        # Auto-detect device
        if torch.cuda.is_available(): device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): device = torch.device("mps")
        else: device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    # --- 2. Load Configuration and Data ---
    if not os.path.exists(args.config_path):
        logging.error("Configuration file not found at %s", args.config_path)
        return

    pipeline_config = config.PipelineConfig.from_json(args.config_path)

    image, ground_truth = io.load_image_with_gt(args.image_path, args.gt_path)
    if image is None:
        logging.error("Failed to load image from %s", args.image_path)
        return

    # --- 3. Initialize Model and Run Pipeline ---
    logging.info("Initializing model '%s'...", pipeline_config.model_name)
    model = core.initialize_model(pipeline_config.model_name, device=device)

    # Create the runner with the model, config, and device
    runner = core.CellposeRunner(model, pipeline_config, device=device)

    logging.info("Running segmentation on the image...")
    masks, duration = runner.run(image)
    logging.info("Segmentation completed in %.2f seconds.", duration)

    # --- 4. Evaluate and Visualize ---
    metrics = {}
    if ground_truth is not None and masks is not None:
        # We now use the more comprehensive F1-score stats
        metrics = calculate_segmentation_stats(ground_truth, masks)
        logging.info("Performance on this image: F1=%.3f, P=%.3f, R=%.3f",
                     metrics.get('f1_score', 0),
                     metrics.get('precision', 0),
                     metrics.get('recall', 0))

    # Launch Napari viewer
    viewer = napari.Viewer(title="Cellpose Visualization")
    visualization.add_image_layer(viewer, image, name="Image")

    if ground_truth is not None:
        visualization.add_labels_layer(
            viewer, ground_truth, name="Ground Truth", opacity=0.5
        )

    if masks is not None:
        f1_score = metrics.get('f1_score', 0.0)
        mask_name = f"Prediction (F1={f1_score:.2f})"
        visualization.add_labels_layer(viewer, masks, name=mask_name, opacity=0.7)

    logging.info("Launching Napari viewer. Close the window to exit.")
    napari.run()


if __name__ == "__main__":
    main()