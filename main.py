import os

from cellpose_adapt.main import cellpose_eval
from experiments.data import data


def main():

    root = "./"
    output_dir = os.path.join(root, "results", "P013T", "Visualizations")

    for image_path, ground_truth_path in data:

        if not os.path.exists(os.path.join(root, image_path)):
            print(f"Image file {image_path} does not exist.")
        if not os.path.exists(os.path.join(root, ground_truth_path)):
            print(f"Labels file {ground_truth_path} does not exist.")

        if "nuclei" in ground_truth_path.lower():
            type = "Nuclei"
        elif "membrane" in ground_truth_path.lower():
            type = "Membranes"
        else:
            raise ValueError(f"Invalid ground truth: {ground_truth_path}")

        image_name = os.path.basename(image_path).replace(".tif", "")
        param_file = f"{image_name}_{type}_config.yaml"
        param_path = os.path.join(root, "results", "P013T", param_file)

        if not os.path.exists(param_path):
            print(f"Parameter file {param_path} does not exist.")
            continue

        cellpose_eval(
            image_path=image_path,
            ground_truth_path=ground_truth_path,
            param_file=param_path,
            output_dir=output_dir,
            cache_dir=".cache",
            show_gt=True,
            show_prediction=True,
            video_3d=True,
            show_viewer=True,
            export_video=False,
            only_std_out=False,
            type=type,
        )



if __name__ == "__main__":
    main()
