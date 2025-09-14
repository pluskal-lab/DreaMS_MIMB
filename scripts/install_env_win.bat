@echo on
setlocal enabledelayedexpansion

REM =============================================
REM DreaMS_MIMB Windows one-click installer (CPU)
REM Run this from Anaconda Prompt in repo root.
REM =============================================

REM --- Guard: in repo root?
if not exist "environment.yml" (
  echo [ERROR] environment.yml not found. Run this in the repo root.
  exit /b 1
)

REM --- Guard: conda available?
where conda
if errorlevel 1 (
  echo [ERROR] Conda not found. Use "Anaconda Prompt (miniconda3)".
  exit /b 1
)

REM --- Pick solver: mamba if available, else conda
where mamba
if errorlevel 1 (
  set "SOLVER=conda"
  echo [INFO] mamba not found; using conda.
) else (
  set "SOLVER=mamba"
  echo [INFO] using mamba.
)

REM --- (Optional) install mamba; ignore failure
conda install -n base -y mamba

REM --- Add channels (idempotent)
conda config --add channels conda-forge
conda config --add channels bioconda
conda config --add channels defaults

REM --- Create env from YAML (name inside YAML is "dreams_mimb")
echo === Creating environment "dreams_mimb" from environment.yml ===
%SOLVER% env create -f environment.yml || (
  echo [ERROR] Could not create environment from environment.yml
  exit /b 1
)

REM --- Activate env
call conda activate dreams_mimb || (
  echo [ERROR] Could not activate environment "dreams_mimb"
  exit /b 1
)

REM --- Verify activation
echo CONDA_PREFIX=%CONDA_PREFIX%
if "%CONDA_PREFIX%"=="" (
  echo [ERROR] Environment not activated. Stop.
  exit /b 1
)

REM --- Ensure MSVC runtime (fixes fbgemm.dll on fresh Windows)
conda install -y -c conda-forge vs2015_runtime vc14_runtime

REM --- If conda’s PyTorch is present, remove ONLY that
conda list pytorch
if not errorlevel 1 (
  echo === Removing conda pytorch to avoid Windows DLL issues ===
  conda remove -y pytorch
)

REM --- If a pip torch is present, uninstall it (ok if missing)
python -c "import importlib,sys;sys.exit(0 if importlib.util.find_spec('torch') else 1"
if not errorlevel 1 (
  python -m pip uninstall -y torch
)

REM --- Install PyTorch (CPU) from the official wheel index
echo === Installing PyTorch 2.3.0 (CPU) via pip ===
python -m pip install --upgrade --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.0 || (
  echo [ERROR] Failed installing PyTorch CPU wheel.
  exit /b 1
)

REM --- Lightning stack (pip; matches your pins)
python -m pip install --upgrade pytorch-lightning==2.2.5 torchmetrics==1.4.0
python -m pip install --upgrade lightning-utilities

REM --- Project extras (no-deps to avoid pulling conflicting torch)
python -m pip install --no-deps git+https://github.com/pluskal-lab/DreaMS.git
python -m pip install --no-deps massspecgym

REM --- Missing libs you needed on Windows
python -m pip install pandarallel==1.6.5 fire==0.6.0
python -m pip install --no-deps git+https://github.com/roman-bushuiev/msml_legacy_architectures.git@main

REM --- Register Jupyter kernel for this env
python -m pip install --upgrade ipykernel
python -m ipykernel install --user --name dreams_mimb --display-name "Python (dreams_mimb)"

REM --- Sanity
python -c "import sys, torch; print('Python:', sys.executable); print('Torch:', torch.__version__, 'CUDA?', torch.cuda.is_available())" || (
  echo [ERROR] Torch sanity failed.
  exit /b 1
)

echo.
echo ✔ All set.
echo Next steps:
echo     conda activate dreams_mimb
echo     python scripts\download_assets.py
echo     jupyter lab
echo     (Kernel -> Python (dreams_mimb))
echo.
endlocal
