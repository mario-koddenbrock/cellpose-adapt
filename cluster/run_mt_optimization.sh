#!/bin/bash

CONFIGS=(
  "configs/project_configs/mt_2d_real.json"
  "configs/project_configs/mt_2d_synthetic.json"
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

echo "All optimization jobs submitted."