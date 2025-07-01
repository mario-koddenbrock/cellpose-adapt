import pytest
import torch

from cellpose_adapt.config.model_config import ModelConfig
from cellpose_adapt.core import initialize_model
from .test_utils import create_synthetic_data  # Import the helper


@pytest.fixture(scope="session")
def synthetic_test_data():
    """Generates synthetic data once per session using the utility function."""
    print("\n--- Generating synthetic test data ---")
    img, masks = create_synthetic_data()
    # The API expects (seg_image, gt_mask, display_image)
    return img, masks, img

@pytest.fixture(scope="session")
def device():
    """Provides the CPU device for testing."""
    return torch.device("cpu")

@pytest.fixture(scope="session")
def simple_config():
    """Provides a simple ModelConfig suitable for the synthetic data."""
    # The area of the smaller circle is > 2000 pixels, so min_size=500 is safe.
    return ModelConfig(diameter=60, do_3D=False, min_size=500)

@pytest.fixture(scope="session")
def cellpose_model(device):
    """Initializes a lightweight Cellpose model once per session."""
    return initialize_model("cyto", device=device)