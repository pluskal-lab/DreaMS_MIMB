# DreaMS_MIMB

TODO this text redo

A step-by-step tutorial and reproducible environment for working with the DreaMS large foundation model for mass spectrometry.  
This project guides you through the full workflow—from data preparation, through model fine-tuning, to advanced investigation—with annotated Jupyter Notebooks, explained for absolute beginners.

---

## 🚀 What is DreaMS_MIMB?

- **DreaMS_MIMB** is a collection of logical, stepwise tutorials (Jupyter Notebooks) to teach you:
  - How to prepare mass spectrometry data
  - Deduplicate and match spectral libraries
  - Train and fine-tune the DreaMS model for your own data
  - Investigate and interpret predictions

**No prior experience with machine learning or mass spectrometry required!**

---

## 🗂️ Project Structure

```
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
├── benchmark          # All main code and packages used in the tutorials
├── paths.py           # Unified project paths for all code
├── data/              # Where your data files go (see notebooks for instructions)
├── README.md          # You are here!
```

---


<h2 id="step1">🛠️ Step 1: Prerequisites</h2>

Before you start, make sure you have:
- **Anaconda / Miniconda installed**  
  (Recommended: Miniconda for minimal install; both Windows, Mac, and Linux are supported.)
- **Conda** is a tool that helps you install all the software and libraries needed for this project with a single command, and keeps them separated from other programs on your computer.

If you don’t have Conda yet:
1. Download Miniconda from [here](https://docs.conda.io/en/latest/miniconda.html)
2. Follow the official install instructions for your operating system  


**How to check if Conda is installed**

Open your terminal (or Anaconda Prompt on Windows) and type:

```bash
conda --version
```

If you see a version number (like `conda 23.3.1`), you’re ready!  
If you see an error, install Miniconda first.

---

## 📦 Step 2: Get the code

1. Open your terminal  
   - On Windows: Open the Anaconda Prompt from the Start Menu (search for “Anaconda Prompt” and click it).  
     **TODO:** Add screenshot of opening Anaconda Prompt  
   - On macOS/Linux: Open your Terminal application from the Applications folder or system menu.
2. Navigate to the folder where you want this project. For example, to use your Documents folder:

    ```bash
    cd Documents
    ```

3. Download (clone) the project code from GitHub:

    Type these commands in the terminal:
    
    ```bash
    git clone https://github.com/Jozefov/DreaMS_MIMB.git
    cd DreaMS_MIMB
    ```
   This downloads all the files and code you will need for the tutorial into a folder called DreaMS_MIMB.
   
	(You only need to do this once. You’ll always run everything from inside this folder.)

### 💡 **If you don’t have Git installed**

You can get the code just as easily! This method does the same thing as the `git clone` command, just with a few extra clicks.

1. Go to the project’s GitHub page:  
   [https://github.com/Jozefov/DreaMS_MIMB](https://github.com/Jozefov/DreaMS_MIMB)

2. Click the green **“Code”** button, then click **“Download ZIP”**.

3. Unzip (extract) the folder to the location you want on your computer
   - for example, your **Documents** folder.  
      This will create a folder named `DreaMS_MIMB` containing all the code and resources you need.

4. Open your terminal, and *change directory* (terminal command `cd`) into that folder.  
   If you put it in Documents:

    ```bash
    cd DreaMS_MIMB
    ```

   *(If your folder is in a different place, use that path instead.)*

---

**What’s happening in this step?**  
You are simply downloading all the resources and code needed for this tutorial.

*Don’t worry, using the ZIP file is just as good as using Git!*

---

<h2 id="step3">🧪 Step 3: Setup Your Environment (Windows, macOS, Linux)</h2>

**Confirm Conda is working**

**First, make sure you’re using the terminal in the same folder as the previous step (the DreaMS_MIMB folder)**. If you closed your terminal, just open a new one and use cd to enter your project folder again.

To check that Conda is installed and working, type:
```bash
conda --version
```

If you see the version number, proceed! If not, [go back to Step 1](#step1).

1. **Create the Conda Environment**  
   Now you’ll set up a dedicated workspace for this project, with all the right tools and libraries. In your terminal, type:

    ```bash
    conda env create -f environment.yml
    ```
   This command will automatically install everything needed for the tutorial.

2. **Activate the Environment**
	
	Next, activate your new workspace with:
    ```bash
    conda activate dreams_mimb
    ```
	**What does “activate” mean?**
	- By “activating” the environment, you make sure that any commands you run will use the correct versions of Python and all necessary libraries, without interfering with other projects or programs on your computer.
    - You’ll need to activate this environment every time you open a new terminal and want to work on this project.

If you see errors about “conda command not found,” make sure you have Conda installed and restart your terminal.

---

## 📓 Step 4: Start Jupyter Notebooks

To work with this project, you **must** use Jupyter Notebooks.
Jupyter lets you run small pieces of code (called “cells”), see the results instantly, and mix code with explanations, all in your web browser.

**(A) Recommended: JupyterLab**

In your terminal (make sure your environment is activated), type:

```bash
jupyter lab
```

- After a few seconds, your default web browser will open automatically.
If not, copy and paste the link from your terminal into your browser.
- You’ll see a file browser, navigate into the notebooks folder on the left side.
- Start with `1_data_preparation.ipynb` and proceed step by step.  
  **TODO:** Add screenshot of browser interface and notebooks folder!

**(B) Classic Jupyter notebook (if you prefer):**

```bash
jupyter notebook
```

- Type this command in your terminal (just like before).
- This will also open a new tab in your browser, but with a simpler interface.
- Find and open the notebooks folder, then open the first notebook.

  **TODO:** Add screenshot or note where this command is run: always in the terminal!

**Tip:**
Both JupyterLab and the classic notebook work the same way for this tutorial, choose the one you like best!
Just remember:
- You **always** run jupyter lab or jupyter notebook **in your terminal** (not in a Python console or elsewhere).
- **Always** make sure you have activated your Conda environment before starting ([see Step 3](#step3)).

---

## 📚 Step 5: Follow the Tutorials

- The notebooks are numbered—go through them in order for best results.
- Each notebook includes:
  - Explanations: What you are about to do, and why
  - Code cells: With detailed comments
  - Instructions: What to change, where to add your data, what to expect

**Tip:**
- Jupyter notebooks are organized into “cells”.
- You run one cell at a time by clicking the cell and pressing `Shift + Enter`.
- Cells must be run in order from the top:  
  Make sure all previous cells have been run before running a new one, or you may get errors!
- **TODO:** Add screenshot of notebook with cells, and highlight the Run button

---

## 🧩 Troubleshooting

- If you run into issues:
  - Check the Issues tab on GitHub, or open a new issue with your question.  
    **TODO:** Provide screenshot of creating a GitHub issue
- Common errors:
  - See the Notes section in our book chapter for the most common pitfalls and solutions.
- **Still stuck?**  
  If you don’t find an answer in the Notes section or we haven’t responded on GitHub, you can also use large language models (like ChatGPT or GitHub Copilot). Just paste your error message and the code you’re struggling with—often, these tools can help you quickly find a solution to common mistakes.

---

## 🔗 More Resources

- DreaMS Foundation Model (Pluskal Lab)
- MassSpecGym
- **TODO:** Add more links or resources as needed

---

## 📝 Contributing

Suggestions, corrections, and pull requests are welcome!  
Please open an issue or pull request to contribute.

---

## 🙌 Acknowledgements

This project was developed as a companion for the DreaMS tutorial chapter and for anyone who wants to explore modern ML for mass spectrometry—step by step and code-first.

---

## 🎉 Happy Learning and Exploring!

---

<!--
TODO:
- Add screenshots, videos, and GIFs where marked
- Add direct links to resources and documentation
- Consider adding a FAQ section or expanding Troubleshooting as new questions come in
-->

---

## Minimal Setup Commands (for quick copy-paste):

```bash
git clone https://github.com/Jozefov/DreaMS_MIMB.git
cd DreaMS_MIMB
conda env create -f environment.yml
conda activate dreams_mimb
jupyter lab
```

---

Let me know when you’re ready to add the images, videos, or GIFs!  
Or if you want custom “intro text” for the start of each notebook, just ask.

⸻

[//]: # (Let me know when you’re ready to add the images, videos, or GIFs!)

[//]: # (Or if you want custom “intro text” for the start of each notebook, just ask.)

[//]: # ()
[//]: # ()
[//]: # ()
[//]: # ()
[//]: # ()
[//]: # (If you don’t have Conda yet:)

[//]: # (	•	Download Miniconda from here)

[//]: # (	•	Follow the official install instructions for your operating system)

[//]: # (# TODO ADD VIDEOS )

[//]: # ()
[//]: # (📦 Step 2: Get the Code)

[//]: # ()
[//]: # (Open your terminal &#40;Anaconda Prompt on Windows!&#41; and run:)

[//]: # (#TO DO add image, or shortcut what you are opening, Then better to say you should navigate to folder where you want to have this project)

[//]: # ()
[//]: # (git clone https://github.com/Jozefov/DreaMS_MIMB.git)

[//]: # (cd DreaMS_MIMB)

[//]: # (# TODO show this, screen shots:)

[//]: # (If you don’t have git installed, you can also download the ZIP from the GitHub page and extract it, then open your terminal and cd into the folder.)

[//]: # ()
[//]: # (🧪 Step 3: Setup Your Environment &#40;Windows, macOS, Linux&#41;)

[//]: # (# TODO some command to show if conda is installed properly, plus before we should explain it a little.)

[//]: # (1. Create the Conda Environment)

[//]: # ()
[//]: # (&#40;This ensures all dependencies work on your OS.&#41;)

[//]: # (conda env create -f environment.yml)

[//]: # ()
[//]: # (2. Activate the Environment)

[//]: # (conda activate dreams_mimb)

[//]: # (If you see errors about “conda command not found,” make sure you have Conda installed and restart your terminal!)

[//]: # ()
[//]: # ()
[//]: # (📓 Step 4: Start JupyterLab or Notebook)

[//]: # (# TODO make stronger statement, like we need jupyter notebooks for running this project.)

[//]: # (&#40;A&#41; Recommended: JupyterLab)

[//]: # (# TODO show images and say for everything where you running it, like in terminal, in google where, always say where:)

[//]: # ()
[//]: # (	•	This will open a new tab in your browser.)

[//]: # (	•	Navigate to the notebooks folder.)

[//]: # (	•	Start with 1_data_preparation.ipynb and proceed step by step.)

[//]: # ()
[//]: # (&#40;B&#41; Classic Notebook &#40;if you prefer&#41;)

[//]: # (jupyter notebook)

[//]: # (	•	Works exactly the same way.)

[//]: # ()
[//]: # (📚 Step 5: Follow the Tutorials)

[//]: # (	•	The notebooks are numbered—go through them in order for best results.)

[//]: # (	•	Each notebook includes:)

[//]: # (	•	Explanations: What you are about to do, and why)

[//]: # (	•	Code cells: With detailed comments)

[//]: # (	•	Instructions: What to change, where to add your data, what to expect)

[//]: # (Tip: If you are new to Jupyter Notebooks, you can run each cell with Shift + Enter.)

[//]: # (# TODO Here we should also say like jupyter notebooks has cells or something and we are running just cell not that all code you see in notebook, just cell by cell and when you have run specific one you haver ensure that all that are before were run, so the one you want will work.)

[//]: # ()
[//]: # (🧩 Troubleshooting)

[//]: # (	•	If you run into issues:)

[//]: # (Check the Issues tab, or open a new issue with your question in github )

[//]: # (# TODO provide image.)

[//]: # (	•	Common errors:)

[//]: # (	•	Look at NOtes section in our book chapet where are written most common pitfalls and how to solve them.)

[//]: # (Besides that i highly recomend if you have not find answer in notes section or we have not managed to respond in issues section at github, i would recommend to use these large languge models as cloud or github, by wirting it whit what part of code you have problem and what is you error. Often it can be really simple problem we have not considered and theroefre can be solved in manners of seconds.)

[//]: # ()
[//]: # (# TODO for all add link)

[//]: # (🔗 More Resources)

[//]: # (	•	DreaMS Foundation Model &#40;Pluskal Lab&#41;)

[//]: # (	•	MassSpecGym)

[//]: # ()
[//]: # (# TODO we can add gifs)

[//]: # ()
[//]: # ()
[//]: # ()
[//]: # (A minimal, reproducible environment for working with the [DreaMS]&#40;https://github.com/pluskal-lab/DreaMS&#41; foundation model for mass spectrometry.)

[//]: # ()
[//]: # (---)

[//]: # ()
[//]: # (## 🔧 Setup)

[//]: # ()
[//]: # (```bash)

[//]: # (git clone git@github.com:Jozefov/DreaMS_MIMB.git)

[//]: # (cd DreaMS_MIMB)

[//]: # (conda env create -f environment.yml)

[//]: # (conda activate dreams_mimb)
