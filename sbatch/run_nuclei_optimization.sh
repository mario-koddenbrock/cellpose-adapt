#!/bin/bash

CONFIGS=(
  "configs/project_configs/organoid_3d_nuclei_20231108.json"
  "configs/project_configs/organoid_3d_nuclei_20240220.json"
  "configs/project_configs/organoid_3d_nuclei_20240305.json"
  "configs/project_configs/organoid_3d_nuclei_20240701.json"
)

for config in "${CONFIGS[@]}"; do
  # Extract a name from the config file path for the job name and logs
  shortname=$(basename "$config" .json)
  echo "Submitting job for config: $config"
  sbatch \
    --job-name="opt_${shortname}" \
    --output="logs/${shortname}_%j.out" \
    --error="logs/${shortname}_%j.err" \
    sbatch/optimization.sbatch "$config"
done

echo "All nuclei optimization jobs submitted."