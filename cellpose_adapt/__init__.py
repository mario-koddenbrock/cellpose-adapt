import os
import warnings

import cellpose
import napari
import torch

# Ignore specific warnings
ignored_warnings = [
    DeprecationWarning,
    FutureWarning,
    UserWarning
]

for warning in ignored_warnings:
    warnings.filterwarnings("ignore", category=warning)

# Set the path to the ffmpeg executable - only needed for exporting animations
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"
print(f"Cellpose version: {cellpose.version}")
print(f"torch version: {torch.__version__}")
print(f"torch cuda available: {torch.cuda.is_available()}")
print(f"torch cuda version: {torch.version.cuda}")
print(f"napari version: {napari.__version__}")
print(f"ffmpeg available: {os.path.exists(os.environ['IMAGEIO_FFMPEG_EXE'])}")
