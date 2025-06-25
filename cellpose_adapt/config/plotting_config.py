from dataclasses import dataclass, field
from typing import Optional, Tuple

@dataclass
class PlottingConfig:
    """Configuration for generating visual reports."""

    # Target resolution for output panels (width, height). If None, uses original size.
    resolution: Optional[Tuple[int, int]] = (2400, 1180)

    # Ground Truth Appearance
    gt_contour_color: Tuple[int, int, int] = (70, 70, 240)
    gt_contour_thickness: int = 1

    # Prediction Appearance
    pred_contour_color: Tuple[int, int, int] = (255, 40, 155)
    pred_contour_thickness: int = 1

    # Font settings for labels
    font_face: int = field(default=0, metadata={"description": "e.g., cv2.FONT_HERSHEY_SIMPLEX"})
    font_scale: float = 0.8
    font_color: Tuple[int, int, int] = (255, 255, 255) # White
    font_thickness: int = 2