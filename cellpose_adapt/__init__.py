import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import warnings
import torch

# Ignore specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set the path to the ffmpeg executable if needed for exporting animations
if os.path.exists("/opt/homebrew/bin/ffmpeg"):
    os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/bin/ffmpeg"

print(f"cellpose_adapt initialized.")
print(f"PyTorch version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA is available. Version: {torch.version.cuda}")