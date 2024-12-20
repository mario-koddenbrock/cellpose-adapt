import os
import warnings

import cellpose

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
