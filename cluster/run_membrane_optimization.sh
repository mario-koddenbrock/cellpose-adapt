#!/bin/bash

CONFIGS=(
  "configs/project_configs/organoid_3d_nuclei_20231108_cluster.json"
  "configs/project_configs/organoid_3d_nuclei_20240220_cluster.json"
  "configs/project_configs/organoid_3d_nuclei_20240305_cluster.json"
  "configs/project_configs/organoid_3d_nuclei_20240701_cluster.json"
)

for config in "${CONFIGS[@]}"; do
  # Extract a name from the config file path for the job name and logs
  shortname=$(basename "$config" .json)
  echo "Submitting job for config: $config"
  sbatch \
    --job-name="opt_${shortname}" \
    --output="logs/${shortname}_%j.out" \
    --error="logs/${shortname}_%j.err" \
    cluster/optimization.sbatch "$config"
done

echo "All membrane optimization jobs submitted."