import logging
import os

from skimage.io import imread

logger = logging.getLogger(__name__)
logger.debug("Starting script to scan for TIFF images in the specified directory.")

def find_tiff_images(root_dir):
    tiff_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith(('.tif', '.tiff')) and not file.lower().endswith('-labels.tif'):
                tiff_files.append(os.path.join(dirpath, file))
    return tiff_files


def print_image_resolutions(tiff_files):
    for file_path in tiff_files:
        image = imread(file_path)
        print(f"{os.path.basename(file_path)}: Shape {image.shape}")


if __name__ == "__main__":
    root_directory = "../data/Organoids"
    tiff_files = find_tiff_images(root_directory)
    print_image_resolutions(tiff_files)
