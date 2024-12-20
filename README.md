# CellposeAdapt: Tailoring Models to Unique Data

This project is part of the FIP project appl-fm and provides tools for visualizing and evaluating cell segmentation results using the Cellpose model. It includes scripts for processing images, evaluating model performance, and visualizing results with Napari.

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/mario-koddenbrock/cellpose-adapt.git
   cd cellpose-viz
   ```

2. Install the required packages:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Main Script

The `main.py` script processes images, evaluates the model, and visualizes results with Napari.

#### Parameters:
- `image_path` (str): Path to the image file.
- `param_file` (str): Path to the parameter YAML file.
- `output_dir` (str): Directory to save the output.
- `cache_dir` (str): Directory for cache files (default is '.cache').
- `show_gt` (bool): Show ground truth labels.
- `show_prediction` (bool): Show prediction labels.
- `video_3d` (bool): Export 3D video.
- `show_viewer` (bool): Show Napari viewer.
- `export_video` (bool): Export video.
- `type` (str): Type of segmentation ('Nuclei' or 'Membranes').

#### Example:
```sh
python cellpose_adapt/main.py --image_path path/to/image.tif --param_file path/to/params.yaml --output_dir Segmentation --show_gt --show_prediction --show_viewer
```

### Visualize Best Scores

The `viz.py` script visualizes the best score for each image and type as a grouped bar plot.

#### Parameters:
- `file_path` (str): Path to the CSV file containing experiment results.
- `metric` (str): The column name of the metric to visualize (default is 'jaccard').
- `output_file` (str): Path to save the bar plot (default is 'best_scores_barplot.png').

#### Example:
```sh
python cellpose_adapt/viz.py --file_path results.csv --metric jaccard --output_file best_scores_barplot.png
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
```
