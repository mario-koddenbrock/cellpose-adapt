#!/bin/bash

CONFIGS=(
  "configs/project_configs/organoid_3d_nuclei_20231108_cluster.json"
  "configs/project_configs/organoid_3d_nuclei_20240220_cluster.json"
  "configs/project_configs/organoid_3d_nuclei_20240305_cluster.json"
#  "configs/project_configs/organoid_3d_nuclei_20241009_cluster.json"
#  "configs/project_configs/organoid_3d_nuclei_20241023_cluster.json"
)

for config in "${CONFIGS[@]}"; do
  # Extract a name from the config file path for the job name and logs
  shortname=$(basename "$config" .json)
  echo "Submitting job for config: $config"
  sbatch \
    --job-name="opt_${shortname}" \
    --output="logs/${shortname}_%j.log" \
    --error="logs/${shortname}_%j.log" \
    cluster/optimization.sbatch "$config"
done

echo "All nuclei optimization jobs submitted."