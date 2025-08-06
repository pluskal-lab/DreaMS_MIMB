#!/usr/bin/env bash
set -e

eval "$(conda shell.bash hook)"

# (optional—to silence the "defaults" warning)
conda config --add channels defaults

# 1) create the env
conda env create -f environment.yml -n dreams_mimb_test

# 2) activate it
conda activate dreams_mimb_test

# 3) now do the two problem packages "no-deps"
pip install --no-deps git+https://github.com/pluskal-lab/DreaMS.git
pip install --no-deps massspecgym

# 4) done!
echo "✔️  All set — run 'jupyter lab' now."