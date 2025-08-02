#!/bin/bash
#SBATCH --account=project_465002061
#SBATCH --partition=small-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --gpus-per-node=1
#SBATCH --mem=60G
#SBATCH --time=72:00:00

# Load necessary modules and activate your environment

export WANDB_API_KEY=${WANDB_API_KEY}
cd /scratch/project_465002061/DreaMS_MIMB/

module --force purge
module use /appl/local/csc/modulefiles/
module load pytorch
#export PYTHONNOUSERSITE=1
source /scratch/project_465002061/DreaMS_MIMB/dreams_mimb/bin/activate

srun python scripts/train.py -m \
  'model.hparams.train_encoder=false,true' \
  'model.hparams.gamma=0.0,0.5,1.0' \
  'model.hparams.alpha=0.25,0.5,0.90'