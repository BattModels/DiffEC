#!/bin/bash
#SBATCH --job-name TrainJax
#SBATCH --partition=your partition name
#SBATCH --output=./Report/%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=80G
#SBATCH --time=00:25:00
#SBATCH --array=0
#####SBATCH --mail-type=BEGIN,END,FAIL
#####SBATCH --mail-user= your email

module load "your python environment"

echo $SLURM_ARRAY_TASK_ID
export JAX_PLATFORMS=cpu
python DiffCEWorker.py $1
