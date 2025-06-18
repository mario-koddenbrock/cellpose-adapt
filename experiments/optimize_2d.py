import os

from experiments.experiments_2d import run_experiments, eval_2d


def main(max_num_images: int = 1):

    data, result_file = get_zagajewski_data(max_num_images)

    best_config_files = run_experiments(data, result_file=result_file)

    eval_2d(data, result_file=result_file)


def get_zagajewski_data(max_num_images):
    # get parent of current path
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_parent_dir = os.path.join(root, "results", "Zagajewski_Data")
    print("output_parent_dir", output_parent_dir)
    image_parent_dir = "/Users/koddenbrock/Repository/Deep-Learning-and-Single-Cell-Phenotyping-for-Rapid-Antimicrobial-Susceptibility-Testing/Zagajewski_Data/Data/MG1655/All_images"
    result_file = os.path.join(output_parent_dir, "results_Zagajewski_Data.csv")
    # get all subfolders in the image parent directory
    image_subfolder = os.listdir(image_parent_dir)
    data = []
    count = 0
    for subfolder in image_subfolder:
        subfolder_path = os.path.join(image_parent_dir, subfolder)
        if os.path.isdir(subfolder_path):
            subfolder_name = os.path.basename(subfolder_path)
            output_dir = os.path.join(output_parent_dir, subfolder_name)
            # os.makedirs(output_dir, exist_ok=True)

            # iterate through all tiff files in the subfolder
            for file in os.listdir(subfolder_path):
                if file.endswith(".tif"):
                    image_path = os.path.join(subfolder_path, file)
                    gt_path = image_path.replace("All_images", "All_segmentations")

                    if not os.path.exists(os.path.join(root, image_path)):
                        print(f"Image file {image_path} does not exist.")
                    if not os.path.exists(os.path.join(root, gt_path)):
                        print(f"Labels file {gt_path} does not exist.")

                    image_name = os.path.basename(image_path).replace(".tif", "")
                    param_file = f"{image_name}_config.yaml"
                    param_path = os.path.join(
                        root, "results", "Zagajewski_Data", param_file
                    )

                    data.append((image_path, gt_path))
                    count += 1
                    if count >= max_num_images:
                        return data, result_file

    return data, result_file


if __name__ == "__main__":
    main()
