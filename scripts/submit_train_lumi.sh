#!/bin/bash
#SBATCH --account=project_465002061
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=64:00:00

# Load necessary modules and activate your environment

export WANDB_API_KEY=${WANDB_API_KEY}
cd /scratch/project_465002061/DreaMS_MIMB/

module --force purge
module use /appl/local/csc/modulefiles/
module load pytorch
#export PYTHONNOUSERSITE=1
source /scratch/project_465002061/DreaMS_MIMB/dreams_mimb/bin/activate

srun python scripts/train.py
#srun python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=8 scripts/train.py