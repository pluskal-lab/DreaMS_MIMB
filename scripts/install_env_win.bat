@echo on
setlocal enabledelayedexpansion

REM =============================================
REM DreaMS_MIMB Windows one-click installer (CPU)
REM Run this from Anaconda Prompt in the repo root.
REM =============================================

REM --- Guard: in repo root?
if not exist "environment.yml" (
  echo [ERROR] environment.yml not found. Run this in the repo root.
  exit /b 1
)

REM --- Conda present?
where conda
if errorlevel 1 (
  echo [ERROR] Conda not found. Use "Anaconda Prompt (miniconda3)".
  exit /b 1
)

REM --- Channels (idempotent)
echo === STEP 1: add channels ===
conda config --add channels conda-forge || goto :error
conda config --add channels bioconda     || goto :error
conda config --add channels defaults     || goto :error

REM --- Create env
echo === STEP 2: create env from environment.yml ===
conda env create -f environment.yml || goto :error

REM --- Activate
echo === STEP 3: activate env ===
call conda activate dreams_mimb || goto :error
echo CONDA_PREFIX=%CONDA_PREFIX%
if "%CONDA_PREFIX%"=="" goto :error

REM --- MSVC runtime (fixes fbgemm.dll)
echo === STEP 4: install MSVC runtime ===
conda install -y -c conda-forge vs2015_runtime vc14_runtime || goto :error

REM --- Replace conda torch with official CPU wheel
echo === STEP 5: replace torch with CPU wheel ===
conda remove -y pytorch
python -m pip uninstall -y torch
python -m pip install --upgrade --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.0 || goto :error

REM --- Lightning stack
echo === STEP 6: lightning stack ===
python -m pip install --upgrade pytorch-lightning==2.2.5 torchmetrics==1.4.0 || goto :error
python -m pip install --upgrade lightning-utilities || goto :error

REM --- Project extras
echo === STEP 7: project extras ===
python -m pip install --no-deps git+https://github.com/pluskal-lab/DreaMS.git || goto :error
python -m pip install --no-deps massspecgym || goto :error
python -m pip install pandarallel==1.6.5 fire==0.6.0 || goto :error
python -m pip install --no-deps git+https://github.com/roman-bushuiev/msml_legacy_architectures.git@main || goto :error

REM --- Register kernel
echo === STEP 8: register Jupyter kernel ===
python -m pip install --upgrade ipykernel || goto :error
python -m ipykernel install --user --name dreams_mimb --display-name "Python (dreams_mimb)" || goto :error

REM --- Sanity
echo === STEP 9: sanity ===
python -c "import sys, torch; print('Python:', sys.executable); print('Torch:', torch.__version__, 'CUDA?', torch.cuda.is_available())" || goto :error

echo.
echo ✔ All set.
echo Next:
echo   conda activate dreams_mimb
echo   python scripts\download_assets.py
echo   jupyter lab   (Kernel -> Python (dreams_mimb))
echo.
goto :eof

:error
echo.
echo *** INSTALL FAILED at the step above. ERRORLEVEL=%ERRORLEVEL% ***
exit /b %ERRORLEVEL%