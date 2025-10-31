#!/bin/bash
#SBATCH --job-name Hydroworker
#SBATCH --partition= #Your compute partition
#SBATCH --output=./Report/%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=10G
#SBATCH --time=00:15:00





########## Load your environment 
module load python/3.11.5
#module load tensorflow/2.7.0
source /nfs/turbo/coe-venkvis/Haotian/jax_env/bin/activate

echo $SLURM_ARRAY_TASK_ID
export JAX_PLATFORMS=cpu
python DiffHydrodynamicWorker.py ${1}
