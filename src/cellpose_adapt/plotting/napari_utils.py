import logging

import napari


def show_napari(image, ground_truth, masks, metrics):
    viewer = napari.Viewer(title="Cellpose Single Image Visualization")
    is_3d = image.ndim == 4
    if is_3d:
        mid_slice = image.shape[0] // 2
        viewer.dims.set_point(0, mid_slice)
        viewer.add_image(image[:, 0], name="Channel 1", colormap="cyan")
        viewer.add_image(image[:, 1], name="Channel 2", colormap="magenta")
    else:
        viewer.add_image(image, name="Image")
    if ground_truth is not None:
        viewer.add_labels(ground_truth, name="Ground Truth", opacity=0.5)
    if masks is not None:
        f1_score = metrics.get('f1_score', 0.0)
        mask_name = f"Prediction (F1={f1_score:.2f})"
        viewer.add_labels(masks, name=mask_name, opacity=0.7)
    logging.info("Launching Napari viewer. Close the window to exit.")
    napari.run()



