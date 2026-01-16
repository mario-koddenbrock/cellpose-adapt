#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=10-16:00:00
#SBATCH --qos=normal
#SBATCH --job-name=eval_optimization
#SBATCH --output=logs/%j_eval_optimization.log
#SBATCH --error=logs/%j_eval_optimization.log


# Log SLURM environment variables
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Job Name: $SLURM_JOB_NAME"
echo "SLURM Nodes: $SLURM_NODELIST"
echo "SLURM GPUs: $CUDA_VISIBLE_DEVICES"
echo "SLURM CPUs per task: $SLURM_CPUS_PER_TASK"
echo "SLURM Memory per node: $SLURM_MEM_PER_NODE"
echo "SLURM Partition: $SLURM_JOB_PARTITION"
echo "Config file: $1"

# Log the current date and hostname
echo "Job started on $(hostname) at $(date)"

# Load necessary modules and activate conda environment
source ~/.bashrc
conda activate cellpose-adapt

# Run the python script
python -m scripts.process_results

# Log completion time
echo "Job finished at $(date)"
