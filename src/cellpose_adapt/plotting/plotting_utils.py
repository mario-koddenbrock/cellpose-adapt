import cv2
import numpy as np

from cellpose_adapt.config.plotting_config import PlottingConfig


# --- Helper function for resizing ---
def resize_image(image: np.ndarray, resolution: tuple) -> np.ndarray:
    """Resizes an image to a target resolution using INTER_AREA for downscaling."""
    return cv2.resize(image, resolution, interpolation=cv2.INTER_AREA)

# --- Updated plotting functions ---

def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """Normalizes an image to a displayable BGR uint8 image using cv2.normalize."""
    img_norm = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    if img_norm.ndim == 2:
        img_norm = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
    return img_norm

def prepare_3d_slice_for_display(image_3d: np.ndarray, is_mask:bool = False) -> np.ndarray:
    """
    Takes a 3D image (Z, C, Y, X), extracts the middle Z-slice,
    and maps the first two channels to an RGB image for display.
    """
    # if image_3d.ndim != 4 or image_3d.shape[1] < 2:
    #     logging.warning("Expected 3D image with shape (Z, C, Y, X) and at least 2 channels. Returning black image.")
    #     return np.zeros((image_3d.shape[2], image_3d.shape[3], 3), dtype=np.uint8)

    if image_3d.ndim == 4:
        mid_slice_idx = image_3d.shape[0] // 2
        img_slice = image_3d[mid_slice_idx, :, :, :]

        ch1 = cv2.normalize(img_slice[0], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        ch2 = cv2.normalize(img_slice[1], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        ch_blue = np.zeros_like(ch1, dtype=np.uint8)

        # Merge channels: Ch1 (e.g., nuclei) -> Red, Ch2 (e.g., membrane) -> Green
        return cv2.merge([ch_blue, ch2, ch1])

    elif image_3d.ndim == 3:
        mid_slice_idx = image_3d.shape[0] // 2
        img_slice = image_3d[mid_slice_idx, :, :]
        if is_mask:
            return img_slice
        else:
            return cv2.cvtColor(img_slice, cv2.COLOR_GRAY2RGB)

    else:
        raise ValueError(f"Unsupported image shape: {image_3d.shape}. Expected 3D or 4D image.")


def create_opencv_overlay(
        display_image: np.ndarray,
        gt_mask: np.ndarray,
        pred_mask: np.ndarray,
        p_config: PlottingConfig
) -> np.ndarray:
    """Creates an overlay image using OpenCV to draw contours based on PlottingConfig."""
    overlay = normalize_to_uint8(display_image)
    overlay = np.ascontiguousarray(overlay, dtype=np.uint8)

    # Draw Ground Truth Contours
    if gt_mask is not None:
        for label in np.unique(gt_mask):
            if label == 0 or label is None: continue
            binary_mask = np.uint8(gt_mask == label)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, p_config.gt_contour_color, p_config.gt_contour_thickness)

    # Draw Prediction Contours
    for label in np.unique(pred_mask):
        if label == 0 or label is None: continue
        binary_mask = np.uint8(pred_mask == label)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, p_config.pred_contour_color, p_config.pred_contour_thickness)

    return overlay


def generate_comparison_panel(
        display_image: np.ndarray,
        overlay: np.ndarray,
        p_config: PlottingConfig,
        num_instances_gt: int,
        num_instances_pred: int,
        plot_original_image: bool = False
) -> np.ndarray:
    """Stitches the original image and the overlay side-by-side, resizing if needed."""
    image_display = normalize_to_uint8(display_image)

    h, w = image_display.shape[:2]
    h_sep = 50

    # Add text labels
    min_dim = min(h, w)
    if min_dim > 500:
        font_scale = p_config.font_scale
        font_thickness = p_config.font_thickness
    elif min_dim > 400:
        font_scale = p_config.font_scale * 0.8
        font_thickness = p_config.font_thickness - 1
    elif min_dim > 300:
        font_scale = p_config.font_scale * 0.6
        font_thickness = p_config.font_thickness - 1
    else:
        font_scale = p_config.font_scale * 0.4
        font_thickness = p_config.font_thickness - 1


    cv2.putText(image_display, 'Original Image', (30, h - h_sep), p_config.font_face, font_scale, p_config.font_color, font_thickness)
    cv2.putText(overlay, f'CellposeSAM ({num_instances_pred})', (30, h - h_sep), p_config.font_face, font_scale, p_config.pred_contour_color, font_thickness)
    cv2.putText(overlay, f'GT ({num_instances_gt})', (30, h - h_sep-30), p_config.font_face, font_scale, p_config.gt_contour_color, font_thickness)

    # Stack images
    if plot_original_image:
        panel = np.hstack((image_display, overlay))
    else:
        panel = overlay

    # Resize the final panel if a resolution is specified
    if p_config.resolution:
        if plot_original_image:
            panel = resize_image(panel, p_config.resolution)
        else:
            res = (p_config.resolution[0] // 2, p_config.resolution[1])
            panel = resize_image(panel, res)

    return panel
