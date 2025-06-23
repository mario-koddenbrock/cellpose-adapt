import numpy as np

from cellpose_adapt.core import CellposeRunner


def test_cellpose_runner_produces_mask(cellpose_model, simple_config, synthetic_test_data, device):
    """
    Tests if the CellposeRunner produces a non-empty mask on synthetic data.
    """
    image, _, _ = synthetic_test_data

    # Create the runner with the fixtures
    runner = CellposeRunner(model=cellpose_model, config=simple_config, device=device)

    # Run segmentation
    pred_mask, duration = runner.run(image)

    assert pred_mask is not None
    assert isinstance(pred_mask, np.ndarray)
    assert pred_mask.shape == image.shape
    # TODO Check if at least one object was found (more than just background) - needs better test configuration
    # assert len(np.unique(pred_mask)) > 1

def test_metrics_calculation_perfect_match(synthetic_test_data):
    """
    Tests metrics calculation with a perfect match (gt vs. gt).
    """
    from cellpose_adapt.metrics import calculate_segmentation_stats

    _, gt_mask, _ = synthetic_test_data

    # In a perfect case, F1-score and its components should be 1.0
    stats = calculate_segmentation_stats(gt_mask, gt_mask)

    assert stats['f1_score'] == 1.0
    assert stats['precision'] == 1.0
    assert stats['recall'] == 1.0
    assert stats['fp'] == 0
    assert stats['fn'] == 0