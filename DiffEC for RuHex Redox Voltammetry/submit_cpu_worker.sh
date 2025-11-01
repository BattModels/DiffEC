#!/bin/bash
#SBATCH --job-name RuHexWorker
#SBATCH --partition=#Your Partion
#SBATCH --output=./Report/%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=20G
#SBATCH --time=00:25:00
#######SBATCH --array=0
######SBATCH --mail-type=BEGIN,END,FAIL
#######SBATCH --mail-user=nmerovingian@gmail.com

module load python/3.11.5
#module load tensorflow/2.7.0
source /nfs/turbo/coe-venkvis/Haotian/jax_env/bin/activate

echo $SLURM_ARRAY_TASK_ID
export JAX_PLATFORMS=cpu
python DiffECWorker.py $1
