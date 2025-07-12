# DreaMS_MIMB

A step-by-step tutorial and reproducible environment for working with the DreaMS large foundation model for mass spectrometry.
This project guides you through the full workflow—from data preparation, through model fine-tuning, to advanced investigation—with annotated Jupyter Notebooks, explained as if you have never used mass spectrometry AI tools before.

🚀 What is DreaMS_MIMB?
	•	DreaMS_MIMB is a collection of stepwise tutorials (Jupyter Notebooks) to teach you:
	•	How to prepare mass spectrometry data
	•	Deduplicate and match spectral libraries
	•	Train and fine-tune the DreaMS model for your own data
	•	Investigate and interpret predictions

No prior experience with machine learning or mass spectrometry required!

🗂️ Project Structure

DreaMS_MIMB/
│
├── notebooks/
│    ├── 1_data_preparation.ipynb         # Start here: loading and understanding your data
│    ├── 2_deduplication.ipynb            # How to deduplicate spectra
│    ├── 3_library_matching.ipynb         # Match data to spectral libraries
│    ├── 4_1_DreaMS_focal_loss.ipynb      # Learn about focal loss in DreaMS
│    ├── 4_DreaMS_finetuning.ipynb        # Fine-tune the DreaMS model
│    ├── 5_prediction_investigation.ipynb # Dive into predictions and their meaning
│
├── environment.yml    # Reproducible setup for Conda
├── paths.py           # Unified project paths for all code
├── data/              # Where your data files go (see notebooks for instructions)
├── README.md          # You are here!

🛠️ Step 1: Prerequisites

Before you start, make sure you have:
	•	Anaconda / Miniconda installed
(Recommended: Miniconda for minimal install; both Windows, Mac, and Linux are supported.)

If you don’t have Conda yet:
	•	Download Miniconda from here
	•	Follow the official install instructions for your operating system

A minimal, reproducible environment for working with the [DreaMS](https://github.com/pluskal-lab/DreaMS) foundation model for mass spectrometry.

---

## 🔧 Setup

```bash
git clone git@github.com:Jozefov/DreaMS_MIMB.git
cd DreaMS_MIMB
conda env create -f environment.yml
conda activate dreams_mimb
