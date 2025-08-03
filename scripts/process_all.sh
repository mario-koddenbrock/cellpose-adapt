#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Set the PyTorch MPS fallback environment variable.
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Activate conda environment. On macOS, the default shell is zsh, which uses .zshrc.
# If you use bash, you might need to change this back to .bashrc.
if [ -f ~/.zshrc ]; then
  source ~/.zshrc
elif [ -f ~/.bashrc ]; then
  source ~/.bashrc
fi

conda activate cellpose-adapt

# Define an array of project configuration files to process.
declare -a configs=(
    "configs/project_configs/organoid_3d_nuclei_20231108_local.json"
    "configs/project_configs/organoid_3d_nuclei_20240220_local.json"
    "configs/project_configs/organoid_3d_nuclei_20240305_local.json"
    "configs/project_configs/organoid_3d_nuclei_20240701_local.json"
)

# Loop through the configuration files and run the Python script for each.
for config in "${configs[@]}"
do
  echo "--- Processing project: ${config} ---"
  python -m scripts.process_results --project_config "${config}"
done

echo "--- All projects processed successfully. ---"