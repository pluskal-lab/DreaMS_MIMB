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
# TODO ADD VIDEOS 

📦 Step 2: Get the Code

Open your terminal (Anaconda Prompt on Windows!) and run:
#TO DO add image, or shortcut what you are opening, Then better to say you should navigate to folder where you want to have this project

git clone https://github.com/Jozefov/DreaMS_MIMB.git
cd DreaMS_MIMB
# TODO show this, screen shots:
If you don’t have git installed, you can also download the ZIP from the GitHub page and extract it, then open your terminal and cd into the folder.

🧪 Step 3: Setup Your Environment (Windows, macOS, Linux)
# TODO some command to show if conda is installed properly, plus before we should explain it a little.
1. Create the Conda Environment

(This ensures all dependencies work on your OS.)
conda env create -f environment.yml

2. Activate the Environment
conda activate dreams_mimb
If you see errors about “conda command not found,” make sure you have Conda installed and restart your terminal!


📓 Step 4: Start JupyterLab or Notebook
# TODO make stronger statement, like we need jupyter notebooks for running this project.
(A) Recommended: JupyterLab
# TODO show images and say for everything where you running it, like in terminal, in google where, always say where:

	•	This will open a new tab in your browser.
	•	Navigate to the notebooks folder.
	•	Start with 1_data_preparation.ipynb and proceed step by step.

(B) Classic Notebook (if you prefer)
jupyter notebook
	•	Works exactly the same way.

📚 Step 5: Follow the Tutorials
	•	The notebooks are numbered—go through them in order for best results.
	•	Each notebook includes:
	•	Explanations: What you are about to do, and why
	•	Code cells: With detailed comments
	•	Instructions: What to change, where to add your data, what to expect
Tip: If you are new to Jupyter Notebooks, you can run each cell with Shift + Enter.
# TODO Here we should also say like jupyter notebooks has cells or something and we are running just cell not that all code you see in notebook, just cell by cell and when you have run specific one you haver ensure that all that are before were run, so the one you want will work.

🧩 Troubleshooting
	•	If you run into issues:
Check the Issues tab, or open a new issue with your question in github 
# TODO provide image.
	•	Common errors:
	•	Look at NOtes section in our book chapet where are written most common pitfalls and how to solve them.
Besides that i highly recomend if you have not find answer in notes section or we have not managed to respond in issues section at github, i would recommend to use these large languge models as cloud or github, by wirting it whit what part of code you have problem and what is you error. Often it can be really simple problem we have not considered and theroefre can be solved in manners of seconds.

# TODO for all add link
🔗 More Resources
	•	DreaMS Foundation Model (Pluskal Lab)
	•	MassSpecGym

# TODO we can add gifs



A minimal, reproducible environment for working with the [DreaMS](https://github.com/pluskal-lab/DreaMS) foundation model for mass spectrometry.

---

## 🔧 Setup

```bash
git clone git@github.com:Jozefov/DreaMS_MIMB.git
cd DreaMS_MIMB
conda env create -f environment.yml
conda activate dreams_mimb
