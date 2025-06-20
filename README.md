# Automated Cellpose Optimization Pipeline

This project provides a robust and automated pipeline for optimizing [Cellpose](https://github.com/MouseLand/cellpose)
segmentation parameters for unique 2D and 3D imaging datasets. It replaces manual trial-and-error with an intelligent,
configuration-driven workflow using Optuna for hyperparameter search, enabling reproducible and high-quality results.

## Key Features

- **Intelligent Hyperparameter Optimization**: Uses [Optuna](https://optuna.org/) to efficiently search for the best
  segmentation parameters, saving significant time over grid searches.
- **Configuration-Driven**: All aspects of an experiment—from data paths to the optimization search space—are controlled
  via simple JSON files, keeping the code clean and experiments reproducible.
- **Reproducible Workflow**: A clear, multi-step process for optimization, analysis, and reporting ensures that results
  can be consistently reproduced.
- **Advanced Caching**: Implements smart caching for both image loading (in-memory) and model predictions (on-disk) to
  dramatically speed up repeated runs.
- **Robust Logging**: Centralized logging provides detailed insight into the pipeline's execution, with configurable
  verbosity levels.
- **Automated Reporting**: Automatically generates visual (side-by-side images) and quantitative (CSV with F1-scores,
  precision, recall) reports to summarize the performance of the best-found parameters.

## The Workflow at a Glance

The pipeline follows a logical, sequential process:

[Project & Search Configs] -> [1. Optimize] -> [2. Analyze] -> [3. Report/Visualize]
(You create) (run_optimization) (analyze_study) (generate_report /
visualize_results)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/mario-koddenbrock/cellpose-adapt.git
   cd cellpose-adapt
   ```

2. **Create a virtual environment and install packages:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **FFMPEG (Optional)**: For exporting videos from Napari, ensure `ffmpeg` is installed and accessible in your system's
   PATH.

## Usage: A Step-by-Step Guide

Follow these steps to find the best parameters for your dataset and generate reports.

### Step 1: Create Your Configuration Files

This is the most important step. You define your entire experiment in two types of JSON files.

**A. Project Configuration (`project_configs/`)**

This file defines a single experiment: what data to use, what to name the study, and which search space to apply.

*Example: `project_configs/my_bacteria_experiment.json`*

```json
{
  "description": "Experiment for 2D bacteria on the new microscope.",
  "project_settings": {
    "study_name": "bacteria_exp_v1",
    "n_trials": 100,
    "logging_level": "INFO",
    "device": "cuda",
    "limit_images_per_source": 10
  },
  "data_sources": [
    "data/bacteria_2d_demo"
  ],
  "gt_mapping": {
    "replace": [],
    "suffix": "_masks.tif",
    "img_suffix": ".tif"
  },
  "search_space_config_path": "search_spaces/bacteria_2d_coarse.json"
}
```

**B. Search Space Configuration (`search_spaces/`)**
This file defines the hyperparameters Optuna will search over. You can have different files for coarse vs. fine-tuning,
or for 2D vs. 3D.
Example: search_spaces/bacteria_2d_coarse.json

```json
{
  "description": "A coarse search for 2D bacteria.",
  "fixed_params": {
    "do_3D": false,
    "model_name": "cyto3"
  },
  "search_space": {
    "diameter": [
      "suggest_float",
      15.0,
      80.0,
      "log"
    ],
    "flow_threshold": [
      "suggest_float",
      0.1,
      0.8
    ],
    "cellprob_threshold": [
      "suggest_float",
      -4.0,
      4.0
    ]
  }
}
```

### Step 2: Run the Optimization

Use scripts/run_optimization.py and point it to your project configuration file. This will start the Optuna search and
save the results to a .db file in the studies/ directory.

```bash
python scripts/run_optimization.py project_configs/my_bacteria_experiment.json
```

### Step 3: Analyze the Study

Once the optimization is complete, use scripts/analyze_study.py to find the best trial and save its parameters to a new
JSON file. This script also generates helpful plots about the search.

```bash
python scripts/analyze_study.py \
    --study_db "studies/bacteria_exp_v1.db" \
    --output_config "configs/best_bacteria_config.json"
```

### Step 4: Generate a Full Report

Use scripts/generate_report.py to apply the best configuration to your dataset and generate a comprehensive report,
including side-by-side comparison images and a CSV file with detailed metrics (F1-score, precision, recall).

```bash
python scripts/generate_report.py \
    --study_db "studies/bacteria_exp_v1.db" \
    --project_config "project_configs/my_bacteria_experiment.json" \
    --output_dir "reports/bacteria_exp_v1_results/"
```

### (Optional) Step 5: Visualize a Single Image

If you want to interactively view the segmentation for a single image in Napari, use scripts/visualize_results.py. This
is great for debugging or creating figures.

```bash
python scripts/visualize_results.py \
    --image_path "data/bacteria_2d_demo/image_01.tif" \
    --gt_path "data/bacteria_2d_demo/image_01_masks.tif" \
    --config_path "configs/best_bacteria_config.json"
```

## Project Directory Structure

```plaintext
cellpose-adapt/
├── cellpose_adapt/   # The core Python library
│   ├── __init__.py
│   ├── caching.py
│   ├── config.py
│   ├── core.py
│   ├── io.py
│   ├── logging_config.py
│   ├── metrics.py
│   └── optimization.py
├── configs/                 # Stores best parameter sets found by analysis
│   └── best_bacteria_config.json
├── data/                    # Your imaging data
│   └── bacteria_2d_demo/
├── logs/                    # Log files are saved here
│   └── optimization.log
├── project_configs/         # Main configuration files for experiments
│   └── my_bacteria_experiment.json
├── reports/                 # Output directory for visual & quantitative reports
│   └── bacteria_exp_v1_results/
├── scripts/                 # Executable scripts for the workflow
│   ├── run_optimization.py
│   ├── analyze_study.py
│   ├── generate_report.py
│   └── visualize_results.py
├── search_spaces/           # Defines the hyperparameter search spaces
│   └── bacteria_2d_coarse.json
├── studies/                 # Optuna study databases are saved here
│   └── bacteria_exp_v1.db
└── requirements.txt
```

## Contributing

Contributions are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

