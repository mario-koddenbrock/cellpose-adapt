import napari
import numpy as np
from napari.utils.colormaps import Colormap, ColorArray

def add_image_layer(viewer: napari.Viewer, image: np.ndarray, **kwargs):
    """Adds an image layer to the Napari viewer."""
    viewer.add_image(image, **kwargs)

def add_labels_layer(viewer: napari.Viewer, labels: np.ndarray, **kwargs):
    """Adds a labels layer to the Napari viewer."""
    # Example of setting a default color map if not provided
    if "colormap" not in kwargs:
        colors = ColorArray(np.array([[0, 0, 0, 0], [0.4, 0.7, 1, 1]])) # Transparent background, light blue foreground
        kwargs["colormap"] = Colormap(colors, name="custom_blue")

    viewer.add_labels(labels, **kwargs)