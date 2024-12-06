#!/bin/bash
#SBATCH --job-name=roberta_ray1K
#SBATCH --output=output_%A_%a.out  # Save output logs for each task
#SBATCH --error=error_%A_%a.err    # Save error logs for each task
#SBATCH --nodes=1                 # Number of nodes
#SBATCH --ntasks=1                 # One task per node
#SBATCH --cpus-per-task=24          # Number of CPUs per task
#SBATCH --mem=30G                  # Memory allocation
#SBATCH --time=02:00:00            # Max time limit
#SBATCH --array=0-31                # Array job index (for 10 chunks of tweets)
#SBATCH --mail-type=ALL            # Notify on start, end, and failure
#SBATCH --mail-user=mihai.calciu@univ-lille.fr

# Load required modules (PyTorch, Python, etc.)
#module load python/3.8
#module load pytorch

# Activate virtual environment if needed
#source activate myenv

# Run the Python script with the task array index as argument
#srun python roberta_ray_slurm.py $SLURM_ARRAY_TASK_ID
srun python roberta_ray_slurm.py
