#!/bin/bash

#SBATCH --job-name=cellpose
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=05-16:00:00
#SBATCH --qos=normal
#SBATCH --output=%j_cp-out.log
#SBATCH --error=%j_cp-err.log
#SBATCH --mail-type=ALL

# Log SLURM environment variables
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Job Name: $SLURM_JOB_NAME"
echo "SLURM Nodes: $SLURM_NODELIST"
echo "SLURM GPUs: $CUDA_VISIBLE_DEVICES"
echo "SLURM CPUs per task: $SLURM_CPUS_PER_TASK"
echo "SLURM Memory per node: $SLURM_MEM_PER_NODE"
echo "SLURM Partition: $SLURM_JOB_PARTITION"

# Log the current date and hostname
echo "Job started on $(hostname) at $(date)"

# Load necessary modules and activate conda environment
source ~/.bashrc
conda activate cellpose-adapt

# Print the Python executable path for debugging
which python

# Run the evaluation script
#python scripts/run_optimization.py configs/project_configs/organoid_3d_nuclei_20231108.json
#python scripts/run_optimization.py configs/project_configs/organoid_3d_nuclei_20240220.json
#python scripts/run_optimization.py configs/project_configs/organoid_3d_nuclei_20240305.json
python scripts/run_optimization.py configs/project_configs/organoid_3d_nuclei_20240701.json

# Log completion time
echo "Job finished at $(date)"
