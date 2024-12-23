#!/bin/bash

#SBATCH --job-name=cellpose
#SBATCH --gres=gpu:4               # Request 4 GPUs
#SBATCH --nodes=2
#SBATCH --ntasks=4                 # Number of tasks (1 task that uses multiple GPUs)
#SBATCH --cpus-per-task=64         # Adjust CPUs per task based on the total available cores
#SBATCH --time=48:00:00            # Time limit (24 hours)
#SBATCH --mem=128G                  # Total memory for the job
#SBATCH --qos=normal
#SBATCH --output=cellpose_out_%j.log
#SBATCH --error=cellpose_err_%j.log

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
conda activate microscopy_image_analysis

# Print the Python executable path for debugging
which python

# Run the evaluation script
python main.py

# Log completion time
echo "Job finished at $(date)"
