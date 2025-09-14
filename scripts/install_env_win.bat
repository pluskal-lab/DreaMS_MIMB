@echo off
setlocal enabledelayedexpansion

REM ---------------------------------------------
REM DreaMS_MIMB Windows one-click installer (CPU)
REM ---------------------------------------------

REM 0) Check conda
where conda >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Conda not found. Install Miniconda/Anaconda first.
  exit /b 1
)

REM 1) (Optional) faster solver; ignore if it fails
conda install -n base -y mamba >nul 2>nul

REM 2) Add channels (idempotent)
conda config --add channels conda-forge >nul 2>nul
conda config --add channels bioconda     >nul 2>nul
conda config --add channels defaults     >nul 2>nul

REM 3) Create env from YAML (name inside YAML is "dreams_mimb")
echo === Creating environment "dreams_mimb" from environment.yml ===
mamba env create -f environment.yml >nul 2>nul || conda env create -f environment.yml
if errorlevel 1 (
  echo [ERROR] Could not create environment from environment.yml
  exit /b 1
)

REM 4) Activate
call conda activate dreams_mimb || (
  echo [ERROR] Could not activate environment "dreams_mimb"
  exit /b 1
)

REM 5) Ensure MSVC runtime (fixes fbgemm.dll on fresh Windows)
conda install -y -c conda-forge vs2015_runtime vc14_runtime

REM 6) If conda's PyTorch is present, remove
conda list pytorch | findstr /R "^pytorch\s" >nul
if %errorlevel%==0 (
  echo === Removing conda pytorch to avoid DLL issues on Windows ===
  conda remove -y pytorch
)

REM 7) If a pip torch is present, uninstall it (quietly ok if missing)
python -c "import importlib,sys;sys.exit(0 if importlib.util.find_spec('torch') else 1)"
if %errorlevel%==0 (
  python -m pip uninstall -y torch >nul 2>nul
)

REM 8) Install PyTorch (CPU) from the official wheel index
echo === Installing PyTorch 2.3.0 (CPU) via pip ===
python -m pip install --upgrade --no-cache-dir ^
  --index-url https://download.pytorch.org/whl/cpu ^
  torch==2.3.0

REM 9) Lightning stack (pip; matches your pins)
python -m pip install --upgrade pytorch-lightning==2.2.5 torchmetrics==1.4.0
python -m pip install --upgrade lightning-utilities

REM 10) Project extras (no-deps to avoid pulling conflicting torch)
python -m pip install --no-deps git+https://github.com/pluskal-lab/DreaMS.git
python -m pip install --no-deps massspecgym

REM 11) Missing libs you needed on Windows
python -m pip install pandarallel==1.6.5 fire==0.6.0
python -m pip install --no-deps git+https://github.com/roman-bushuiev/msml_legacy_architectures.git@main

REM 12) Register Jupyter kernel for this env
python -m pip install --upgrade ipykernel
python -m ipykernel install --user --name dreams_mimb --display-name "Python (dreams_mimb)"

echo.
echo === Sanity ===
python - << "PY"
import sys, torch
print("Python:", sys.executable)
print("Torch:", torch.__version__, "CUDA?", torch.cuda.is_available())
PY

echo.
echo ✔ All set.
echo Next steps:
echo     conda activate dreams_mimb
echo     python scripts\download_assets.py
echo     jupyter lab
echo     (Kernel -> Python (dreams_mimb))
echo.
endlocal