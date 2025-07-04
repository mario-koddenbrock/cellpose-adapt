import math
import os

import cv2
import napari
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm


# import matplotlib
# matplotlib.use('TkAgg')


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


def show_napari(results, params):

    # Initialize the Napari viewer
    viewer = napari.Viewer()

    # Add the image to the viewer
    image = viewer.add_image(
        results["image"],
        contrast_limits=[113, 1300],
        name="Organoids",
        colormap="gray",
    )
    # Add the labels to the viewer
    masks = viewer.add_labels(
        results["masks"],
        name=params.model_name,
        opacity=0.8,
        blending="translucent",
    )
    masks.contour = 2
    if results["ground_truth"] is not None:
        ground_truth = viewer.add_labels(
            results["ground_truth"],
            name="Ground Truth",
            opacity=0.3,
            blending="translucent",
        )
    # setting the viewer to the center of the image
    center = results["image"].shape[0] // 2
    viewer.dims.set_point(0, center)
    napari.run()


def plot_intensity(image):
    plt.hist(image.flatten(), bins=1000)
    # make log scale
    plt.yscale("log")
    plt.ylabel("count")
    plt.xlabel("intensity")
    plt.title("intensity distribution")
    plt.show()


def save_as_video(output_video_path, image_with_labels, labels, regions):
    # Get the number of unique labels (excluding the background)
    unique_labels = np.unique(labels)
    num_instances = len(unique_labels) - 1  # Assuming 0 is the background
    # Directory to save frames temporarily
    os.makedirs("frames", exist_ok=True)
    # Create frames progressively increasing the count
    frame_files = []

    # sort the region list by y value
    y_values = [r.centroid[0] for r in regions]
    sorted_idx = np.argsort(y_values)
    regions = [regions[i] for i in sorted_idx]

    for count, region in enumerate(regions, start=1):
        # Create a figure for each count
        plt.figure(figsize=(10, 10))
        plt.imshow(image_with_labels)
        plt.axis("off")

        # Highlight the current instance with a large circle
        y, x = region.centroid
        plt.plot(
            x,
            y,
            marker="o",
            markersize=30,
            markeredgewidth=4,
            markeredgecolor="yellow",
            fillstyle="none",
        )

        # Calculate length and width from the bounding box
        min_row, min_col, max_row, max_col = region.bbox
        length = max_row - min_row
        width = max_col - min_col

        # Display additional info about the region, including length and width
        info_text = f"Count: {count}\nArea: {region.area}\nEccentricity: {region.eccentricity:.2f}\nLength: {length}\nWidth: {width}"
        plt.text(
            10,
            250,
            info_text,
            color="blue",
            fontsize=15,
            bbox=dict(facecolor="white", alpha=0.7),
        )

        # Save the frame to disk
        frame_filename = f"frames/frame_{count:03d}.png"
        plt.savefig(frame_filename, bbox_inches="tight", pad_inches=0)
        frame_files.append(frame_filename)
        plt.close()

    # Create a video from the saved frames
    frame_rate = 1  # 1 frame per second
    # Load the first frame to get dimensions
    frame = cv2.imread(frame_files[0])
    height, width, layers = frame.shape
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))
    # Write each frame into the video
    for frame_file in frame_files:
        frame = cv2.imread(frame_file)
        out.write(frame)
    # Release the VideoWriter and clean up
    out.release()
    # Optionally, delete the temporary frames
    for frame_file in frame_files:
        os.remove(frame_file)
    print(f"Video saved at {output_video_path}")


def plot_aggregated_metric_variation(
    result_path, metric="f1_score", boxplot=False, save_plot=True
):
    """
    Detect varying parameters and plot the aggregated metric over these parameters with uncertainty bands
    aggregated over all image_name and type combinations. Optionally display a boxplot.

    Parameters:
        result_path (str): Path to the CSV file containing experiment results.
        metric (str): The column name of the metric to evaluate (default is 'f1_score').
        boxplot (bool): If True, display boxplots instead of error bars (default is False).
    """

    if isinstance(result_path, list):
        # run over the result_path list and only keep existing files
        result_path = [f for f in result_path if os.path.exists(f)]
        if len(result_path) == 0:
            print(f"No result files found in {result_path}")
            return
    else:
        if not os.path.exists(result_path):
            print(f"Result file {result_path} does not exist.")
            return

    if not isinstance(result_path, list):
        result_path = [result_path]

    # Load data
    df = pd.concat([pd.read_csv(f) for f in result_path])

    # Create output directory in the same folder as the input file
    output_dir = os.path.dirname(result_path[0])

    if len(result_path) > 1:
        result_name = "all_results"
    else:
        result_name = os.path.basename(result_path[0]).replace(".csv", "")

    unique_images = df[["image_name", "type"]].drop_duplicates()

    # Identify varying parameters (excluding fixed columns and specified metrics)
    excluded_columns = [
        "image_name",
        "type",
        metric,
        "duration",
        "are",
        "precision",
        "recall",
        "f1",
        "f1_score",
        "jaccard_cellpose",
        "jaccard",
    ]
    varying_columns = [
        col
        for col in df.columns
        if df[col].nunique() > 1 and col not in excluded_columns
    ]

    print("Varying parameters detected:", varying_columns)

    if not varying_columns:
        print("No varying parameters detected.")
        return

    # Aggregate metric over all image_name and type
    for param in varying_columns:
        fig, ax = plt.subplots(figsize=(6, 4))

        if boxplot:
            # Create boxplot for the metric grouped by the parameter
            df.boxplot(column=metric, by=param, grid=False, ax=ax)
            # ax.set_title(f"Boxplot of {metric} vs {param} (over all images)")

            if len(pd.unique(df[param])) > 10:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            else:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")

            plt.xticks(rotation=45, ha="right")
            ax.set_xlabel(param)
            ax.set_ylabel(metric)
            plt.suptitle("")  # Remove default title
            output_path = os.path.join(
                output_dir, "Plots", f"boxplot_{result_name}_{param}_{metric}.png"
            )

        else:

            # iterate over the unique images and find the best configuration
            for idx, row in unique_images.iterrows():
                image_name = row["image_name"]
                type = row["type"]

                if not math.isnan(type):
                    filtered = df[
                        (df["image_name"] == image_name) & (df["type"] == type)
                    ]
                else:
                    filtered = df[df["image_name"] == image_name]

                grouped = (
                    filtered.groupby(param)[metric].agg(["mean", "std"]).reset_index()
                )
                x = grouped[param]
                y = grouped["mean"]
                yerr = grouped["std"]

                # ax.errorbar(x, y, yerr=yerr, fmt='-o', capsize=5, label=f"{image_name}-{type}")
                # plot x and y values with an uncertainty band
                ax.plot(x, y, "-o", label=f"{image_name}-{type}")
                ax.fill_between(x, y - yerr, y + yerr, alpha=0.2)

                # show legend above the axis in 3 rows, but make it extremely small
                ax.legend(
                    loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize=4.5
                )

            if len(x) > 10:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

            ax.set_xlabel(param)
            ax.set_ylabel(f"Aggregated {metric}")
            # ax.set_title(f"Aggregated {metric} vs {param} (over all images)")

            # ax.legend(bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=2)

            ax.grid(True)
            output_path = os.path.join(
                output_dir, "Plots", f"errorbar_{result_name}_{param}_{metric}.png"
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.title(result_name)
        plt.ylim(0, 1)
        plt.tight_layout()
        if save_plot:
            plt.savefig(output_path)
        plt.show()
        plt.close(fig)
        print(f"Saved plot to {output_path}")


def plot_best_scores_barplot(
    result_path, metric="f1_score", output_file="best_scores_barplot.png", save_plot=True
):
    """
    Visualize the best score for each image_name and type as a grouped bar plot.

    Parameters:
        result_path (str): Path to the CSV file containing experiment results.
        metric (str): The column name of the metric to visualize (default is 'f1_score').
        output_file (str): Path to save the bar plot (default is 'best_scores_barplot.png').
    """

    if isinstance(result_path, list):
        # run over the result_path list and only keep existing files
        result_path = [f for f in result_path if os.path.exists(f)]
        if len(result_path) == 0:
            print(f"No result files found in {result_path}")
            return
    else:
        if not os.path.exists(result_path):
            print(f"Result file {result_path} does not exist.")
            return

    if not isinstance(result_path, list):
        result_path = [result_path]

        # Load data
    df = pd.concat([pd.read_csv(f) for f in result_path])

    # Ensure the metric column exists
    if metric not in df.columns:
        print(f"Error: '{metric}' is not a valid column in the file.")
        print("Available columns:", ", ".join(df.columns))
        return

    # Find the best configuration per image_name and type based on the metric
    best_cfgs = df.loc[df.groupby(["image_name", "type"])[metric].idxmax()]

    # Prepare data for plotting
    grouped = (
        best_cfgs.groupby(["image_name", "type"])[metric].max().unstack(fill_value=0)
    )

    grouped.plot(kind="bar", figsize=(12, 6), alpha=0.8, edgecolor="black")
    # grouped.plot(kind='box', figsize=(12, 6))

    # Plot customization
    # plt.title(f"Best {metric} Scores for Each Image and Type")
    plt.xlabel("Image Name")
    plt.ylabel(f"Best {metric} Score")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.grid(axis="y")
    plt.legend(title="Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    output_dir = os.path.join(os.path.dirname(result_path[0]), "Plots")
    output_path = os.path.join(output_dir, output_file)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the plot
    if save_plot:
        plt.savefig(output_path)
    plt.show()
    print(f"Saved bar plot to {output_path}")
