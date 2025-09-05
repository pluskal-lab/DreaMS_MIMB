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
module load pytorch/2.7
module load pytorch/2.6
module load pytorch/2.7
source /scratch/project_465002061/DreaMS_MIMB/dreams_mimb/bin/activate
module load pytorch/2.7
module load pytorch/2.6
module load pytorch/2.7

srun python3 scripts/train.py  \
  --config-name fluorine_config.yaml  \
  -m 'model.hparams.lr=1e-5, 1e-6' \
    'model.hparams.gamma=0.5,0.75,1.0' \
    'model.hparams.alpha=0.25,0.8,0.9,0.95'