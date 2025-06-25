import os
import json
import tifffile
import pytest
from .test_utils import create_synthetic_data # Import the helper
import json
import os

import pytest
import tifffile

from .test_utils import create_synthetic_data  # Import the helper


@pytest.fixture
def workflow_test_files():
    """Creates temporary config and data files for a full workflow test."""
    print("\n--- Setting up files for workflow test ---")
    os.makedirs("tests/temp_data", exist_ok=True)
    os.makedirs("tests/temp_configs", exist_ok=True)
    os.makedirs("studies", exist_ok=True)
    os.makedirs("configs", exist_ok=True)

    # 1. Generate synthetic data using the helper and save to disk
    img, masks = create_synthetic_data()
    tifffile.imwrite("tests/temp_data/image_01.tif", img)
    tifffile.imwrite("tests/temp_data/image_01_masks.tif", masks)

    # 2. Create project config file
    project_config = {
        "project_settings": {"study_name": "e2e_test_study", "n_trials": 1, "device": "cpu"},
        "data_sources": ["tests/temp_data"],
        "gt_mapping": {"suffix": "_masks.tif", "img_suffix": ".tif"},
        "search_space_config_path": "tests/temp_configs/search_space.json"
    }
    with open("tests/temp_configs/project_config.json", "w") as f:
        json.dump(project_config, f)

    # 3. Create search space config file
    search_config = {
        "fixed_params": {"do_3D": False, "model_name": "cyto", "min_size": 500},
        "search_space": {"diameter": ["suggest_float", 50.0, 70.0]}
    }
    with open("tests/temp_configs/search_space.json", "w") as f:
        json.dump(search_config, f)

    yield

    # 4. Teardown: Clean up all created files and directories
    print("\n--- Tearing down workflow test files ---")
    # ... (teardown logic is unchanged) ...
    files_to_remove = [
        "tests/temp_data/image_01.tif",
        "tests/temp_data/image_01_masks.tif",
        "tests/temp_configs/project_config.json",
        "tests/temp_configs/search_space.json",
        "studies/e2e_test_study.db",
        "configs/best_e2e_test_study_config.json"
    ]
    for f_path in files_to_remove:
        if os.path.exists(f_path):
            os.remove(f_path)
    for d_path in ["tests/temp_data", "tests/temp_configs"]:
        try:
            if os.path.exists(d_path):
                os.rmdir(d_path)
        except OSError:
            pass

# TODO - Implement the full workflow test
# def test_full_workflow(workflow_test_files):
#     result_opt = subprocess.run(...)
#     assert "Optimization finished" in result_opt.stdout
#     result_proc = subprocess.run(...)
#     assert "Processing complete" in result_proc.stdout