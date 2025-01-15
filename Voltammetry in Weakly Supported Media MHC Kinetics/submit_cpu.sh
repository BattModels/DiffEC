#!/bin/bash
#SBATCH --job-name TrainJax
#SBATCH --partition= "Your Partition Name"
#SBATCH --output=./Report/%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=120G
#SBATCH --time=00:25:00
#SBATCH --array=0


module load "Your Python Library"

echo $SLURM_ARRAY_TASK_ID
export JAX_PLATFORMS=cpu
python DiffMigrationWorker.py
