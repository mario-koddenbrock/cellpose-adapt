import argparse
import logging
import os

import napari

from cellpose_adapt import io, core, visualization
from cellpose_adapt.config import PipelineConfig
from cellpose_adapt.logging_config import setup_logging


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
        help="Path to the pipeline configuration JSON file.",
    )
    args = parser.parse_args()

    setup_logging(log_file="visualization.log")

    # Load configuration
    if not os.path.exists(args.config_path):
        logging.error("Configuration file not found at %s", args.config_path)
        return
    config = PipelineConfig.from_json(args.config_path)

    # Load data
    image, ground_truth = io.load_image_with_gt(args.image_path, args.gt_path)
    if image is None:
        return

    # Run the pipeline
    runner = core.CellposeRunner(config)
    masks, duration = runner.run(image)

    # Evaluate if ground truth is available
    metrics = {}
    if ground_truth is not None and masks is not None:
        metrics = runner.evaluate_performance(ground_truth, masks)
        logging.info("Performance on this image: %s", metrics)

    # Visualize
    viewer = napari.Viewer()
    visualization.add_image_layer(viewer, image, name="Image")
    if ground_truth is not None:
        visualization.add_labels_layer(
            viewer, ground_truth, name="Ground Truth", opacity=0.4
        )
    if masks is not None:
        jaccard_score = metrics.get("jaccard_cellpose", 0.0)
        mask_name = f"Prediction (J={jaccard_score:.2f})"
        visualization.add_labels_layer(viewer, masks, name=mask_name, opacity=0.7)

    logging.info("Launching Napari viewer. Close the window to exit.")
    napari.run()


if __name__ == "__main__":
    main()
