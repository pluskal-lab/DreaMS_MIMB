#!/bin/bash
#SBATCH --account=project_465002061
#SBATCH --partition=dev-g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --job-name=test_env

cd /scratch/project_465002061/DreaMS_MIMB/

module --force purge
module use /appl/local/csc/modulefiles/
module load pytorch

echo "Activating venv..."
source dreams_mimb/bin/activate

echo "Which python am I using?"
which python
python --version

echo "PYTHONPATH: $PYTHONPATH"

echo "Where is hydra installed?"
python -c "import hydra; print('hydra:', hydra.__file__)"

echo "Where is torch installed?"
python -c "import torch; print('torch:', torch.__file__)"

echo "Where is omegaconf installed?"
python -c "import omegaconf; print('omegaconf:', omegaconf.__file__)"