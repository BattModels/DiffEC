#!/bin/bash
#SBATCH --job-name worker
#SBATCH --partition=Your Partition Name
#SBATCH --output=./Report/%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=30G
#SBATCH --time=00:25:00

module load python/3.11.5
#module load tensorflow/2.7.0
source /nfs/turbo/coe-venkvis/Haotian/jax_env/bin/activate

echo $SLURM_ARRAY_TASK_ID
export JAX_PLATFORMS=cpu
python DiffECWorker.py ${1}
