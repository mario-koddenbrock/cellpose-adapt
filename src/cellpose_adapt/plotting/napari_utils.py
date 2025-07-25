import logging

import napari

logger = logging.getLogger(__name__)

def show_napari(image=None, mask_nuclei=None, mask_membrane=None, gt_nuclei=None, gt_membrane=None, metrics_nuclei=None, metrics_membrane=None):

    if metrics_membrane is None:
        metrics_membrane = {}
    if metrics_nuclei is None:
        metrics_nuclei = {}

    viewer = napari.Viewer(title="Cellpose Single Image Visualization")

    if not image is None:
        is_3d = image.ndim == 4
        if is_3d:
            mid_slice = image.shape[0] // 2
            viewer.dims.set_point(0, mid_slice)
            viewer.add_image(image[:, 0], name="Nuclei", colormap="cyan")
            viewer.add_image(image[:, 1], name="Membranes", colormap="magenta")
        else:
            viewer.add_image(image, name="Image")

    if gt_nuclei is not None:
        viewer.add_labels(gt_nuclei, name="Nuclei (GT)", opacity=0.5)

    if gt_membrane is not None:
        viewer.add_labels(gt_membrane, name="Membrane (GT)", opacity=0.5)

    if mask_nuclei is not None:
        f1_score = metrics_nuclei.get('f1_score', 0.0)
        mask_name = f"Nuclei (Pred) (F1={f1_score:.2f})"
        viewer.add_labels(mask_nuclei, name=mask_name, opacity=0.7)

    if mask_membrane is not None:
        f1_score = metrics_membrane.get('f1_score', 0.0)
        mask_name = f"Membrane (Pred) (F1={f1_score:.2f})"
        viewer.add_labels(mask_membrane, name=mask_name, opacity=0.7)

    logging.info("Launching Napari viewer. Close the window to exit.")
    napari.run()



