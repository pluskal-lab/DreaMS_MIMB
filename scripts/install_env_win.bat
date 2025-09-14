@echo on
setlocal enabledelayedexpansion

REM =============================================
REM DreaMS_MIMB Windows one-click installer (CPU)
REM Run this from Anaconda Prompt in the repo root.
REM =============================================

REM 0) repo root check
if not exist "environment.yml" (
  echo [ERROR] environment.yml not found. Run this in the repo root.
  exit /b 1
)

REM 1) channels (use CALL with conda!)
call conda config --add channels conda-forge
call conda config --add channels bioconda
call conda config --add channels defaults

REM 2) create env
call conda env create -f environment.yml

REM 3) activate
call conda activate dreams_mimb

REM 4) MSVC runtime
call conda install -y -c conda-forge vs2015_runtime vc14_runtime

REM 5) replace torch with official CPU wheel
call conda remove -y pytorch
python -m pip uninstall -y torch
python -m pip install --upgrade --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.0

REM 6) lightning stack
python -m pip install --upgrade pytorch-lightning==2.2.5 torchmetrics==1.4.0
python -m pip install --upgrade lightning-utilities

REM 7) project extras
python -m pip install --no-deps git+https://github.com/pluskal-lab/DreaMS.git
python -m pip install --no-deps massspecgym
python -m pip install pandarallel==1.6.5 fire==0.6.0
python -m pip install --no-deps git+https://github.com/roman-bushuiev/msml_legacy_architectures.git@main

REM 8) jupyter kernel
python -m pip install --upgrade ipykernel
python -m ipykernel install --user --name dreams_mimb --display-name "Python (dreams_mimb)"

REM 9) sanity
python -c "import sys, torch; print('Python:', sys.executable); print('Torch:', torch.__version__, 'CUDA?', torch.cuda.is_available())"

echo.
echo Done. Next:
echo   conda activate dreams_mimb
echo   python scripts\download_assets.py
echo   jupyter lab   (Kernel -> Python (dreams_mimb))
echo.
endlocal