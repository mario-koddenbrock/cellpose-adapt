import os

import napari
from tqdm import tqdm


def export_napari_video(image, ground_truth, pred_mask, output_dir, video_filename, rotation_steps=72):
    """
    Initializes a Napari viewer and exports a 3D rotation video.

    Args:
        image (np.ndarray): The input image data.
        ground_truth (np.ndarray): The ground truth segmentation mask.
        pred_mask (np.ndarray): The predicted segmentation mask.
        model_name (str): The name of the model, used for the layer name.
        output_dir (str): Directory to save the video.
        video_filename (str): Name of the output video file.
        rotation_steps (int): Number of steps for the 3D rotation animation.
    """
    viewer = napari.Viewer(show=False, ndisplay=3)

    viewer.add_image(
        image,
        name="Image",
        colormap="gray",
        blending="additive",
    )

    viewer.add_labels(
        pred_mask,
        name="CellposeSAM",
        opacity=0.8,
        blending="translucent",
        contour=2
    )

    if ground_truth is not None:
        viewer.add_labels(
            ground_truth,
            name="Ground Truth",
            opacity=0.4,
            blending="translucent",
            contour=2
        )

    # Set camera zoom and center
    viewer.camera.zoom = 0.8
    center = [d / 2 for d in image.shape]
    viewer.camera.center = center

    extract_cellpose_video(
        viewer=viewer,
        output_dir=output_dir,
        video_filename=video_filename,
        num_z_slices=image.shape[0],  # Required by function but not used in 3D mode
        mode="3D",
        rotation_steps=rotation_steps
    )

    viewer.close()


def extract_cellpose_video(
    viewer, output_dir, video_filename, num_z_slices, mode="2D", rotation_steps=72
):
    """
    Create a video from a Napari viewer, with optional 2D z-slice animation or 3D rotation.

    Parameters:
        viewer: napari.Viewer
            The Napari viewer instance.
        output_dir: str
            Directory to save the video.
        video_filename: str
            Name of the output video file.
        num_z_slices: int
            Number of z-slices for the 2D animation.
        mode: str
            Animation mode, either '2D' or '3D'. Defaults to '2D'.
        rotation_angle: int
            Total angle (in degrees) to rotate in 3D mode. Defaults to 360 degrees.
        rotation_steps: int
            Number of steps (frames) for the 3D rotation animation. Defaults to 36.
    """

    try:
        from napari_animation import Animation
    except ImportError:
        print(
            "Warning: napari_animation module is not available. Video extraction is not possible."
        )
        return

    video_path = os.path.join(output_dir, video_filename)

    # Set FPS based on mode
    fps = 25 if mode == "2D" else 30  # Higher FPS for 2D, lower for 3D

    # Create an animation object
    animation = Animation(viewer)

    if mode == "2D":
        # 2D animation: iterate through z-slices
        for z in tqdm(range(int(num_z_slices / 2)), desc="Capturing 2D frames"):
            viewer.dims.set_point(0, 2 * z)  # Set the z-slice
            animation.capture_keyframe()

    elif mode == "3D":
        # Ensure viewer is in 3D mode
        viewer.dims.ndisplay = 3

        # 3D animation: rotate the camera
        for i in tqdm(range(rotation_steps), desc="Capturing 3D frames"):
            azimuth = (i * 360) / rotation_steps
            viewer.camera.angles = (0, azimuth, 0)  # Adjust elevation and azimuth
            animation.capture_keyframe()

    else:
        raise ValueError("Invalid mode. Use '2D' or '3D'.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the animation
    animation.animate(video_path, canvas_only=True, fps=fps, quality=9)
    print(f"Saved animation to {video_path}")

