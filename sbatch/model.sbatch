#!/bin/bash

#SBATCH --job-name=cp-model
#SBATCH --gres=gpu:1               # Request 4 GPUs
#SBATCH --nodes=1
#SBATCH --ntasks=1                 # Number of tasks (1 task that uses multiple GPUs)
#SBATCH --cpus-per-task=32         # Adjust CPUs per task based on the total available cores
#SBATCH --mem=128G                  # Total memory for the job
#SBATCH --qos=normal
#SBATCH --output=%j_cp-model_out.log
#SBATCH --error=%j_cp-model_err.log

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
python -c "from experiments.setup import run_model_experiments; run_model_experiments()"

# Log completion time
echo "Job finished at $(date)"
