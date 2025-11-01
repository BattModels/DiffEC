#!/bin/bash
#SBATCH --job-name master
#SBATCH --partition= # Your partition 
#SBATCH --output=./Report/%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=5G
#SBATCH --time=47:00:00
#SBATCH --array=1-30%10
#SBATCH --mail-type=FAIL


module load python/3.11.5
#module load tensorflow/2.7.0
source /nfs/turbo/coe-venkvis/Haotian/jax_env/bin/activate

echo $SLURM_ARRAY_TASK_ID
export JAX_PLATFORMS=cpu
python DiffECMaster.py $SLURM_ARRAY_TASK_ID
