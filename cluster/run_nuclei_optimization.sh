#!/bin/bash

#CONFIGS=(
#  "configs/project_configs/organoid_3d_nuclei_20231108_cluster.json"
#  "configs/project_configs/organoid_3d_nuclei_20240220_cluster.json"
#  "configs/project_configs/organoid_3d_nuclei_20240305_cluster.json"
#  "configs/project_configs/organoid_3d_nuclei_20241009_cluster.json"
#  "configs/project_configs/organoid_3d_nuclei_20241023_cluster.json"
#)
CONFIGS=(
  "configs/project_configs/nuclei_3d_25x_cluster.json"
  "configs/project_configs/nuclei_3d_40x_cluster.json"
)

for config in "${CONFIGS[@]}"; do
  # Extract a name from the config file path for the job name and logs
  shortname=$(basename "$config" .json)
  echo "Submitting job for config: $config"
  sbatch \
    --job-name="${shortname}" \
    --output="logs/%j_${shortname}.log" \
    --error="logs/%j_${shortname}.log" \
    cluster/optimization.sbatch "$config"
done

echo "All nuclei optimization jobs submitted."
