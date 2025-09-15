# DreaMS MIMB (Methods in Molecular Biology)

---


<p align="center">
  <a href="https://github.com/pluskal-lab/DreaMS">
    <img src="assets/dreams_background.png" alt="DreaMS" width="800">
  </a>
</p>



This repository is the hands-on companion to the book chapter *Annotating metabolomics data using DreaMS and MassSpecGym*. It includes ready-to-run Jupyter notebooks, datasets, and pretrained models for [DreaMS](https://www.nature.com/articles/s41587-025-02663-3) and [MassSpecGym](https://arxiv.org/abs/2410.23326). You can follow the complete workflow, from preparing and deduplicating spectra to matching them in spectral libraries and fine-tuning models. The tutorials match the chapter’s examples and are easy to follow for both beginners and experienced researchers. Everything comes with installation scripts so you can set up the environment quickly and adapt the methods to your own data.

---
## 🚀 What is DreaMS_MIMB?

- **DreaMS_MIMB** is a collection of logical, stepwise tutorials (Jupyter Notebooks) to teach you:
  - How to prepare mass spectrometry data
  - Deduplicate and match spectral libraries
  - Embedding and retrieval from reference library
  - Molecular networking and mass spectra annotation

**No prior experience with machine learning or mass spectrometry required!**


---
## 🧠 What is DreaMS?


<p align="center">
  <a href="https://github.com/pluskal-lab/DreaMS">
    <img src="assets/DreaMS_MIMB_11.png" alt="DreaMS - Deep Representations Empowering the Annotation of Mass Spectra" width="800">
  </a>
</p>

[DreaMS](https://www.nature.com/articles/s41587-025-02663-3) (Deep Representations Empowering the Annotation of Mass Spectra) is a transformer-based foundation model for tandem mass spectrometry. Trained in a self-supervised manner on millions of unannotated spectra from the [GNPS](https://www.nature.com/articles/nbt.3597) open data repository, DreaMS learns general-purpose embeddings, high-dimensional numerical representations of spectra that capture chemical and structural relationships of the underlying molecules. The pre-training task, analogous to masked language modeling in natural language processing, involves reconstructing masked peaks and relative retention order of compounds in chromatography. This self-supervised stage enables DreaMS to learn from the immense pool of unlabelled data, after which the model can be fine-tuned for specific downstream tasks such as elemental composition prediction, spectral similarity search, or molecular fingerprint prediction.

---

<h2 id="project-structure">🗂️ Project Structure</h2>

```
DreaMS_MIMB/
│
├── benchmark                                # All main code and packages used in the tutorials
├── data/                                    # Where your data files go (see notebooks for instructions)
│   ├── Piper_data_smiles.csv                # SMILES and metadata table for Piper fimbriulatum compounds
│   ├── Piper_sirius_all_annotated.mgf       # All spectra from Piper fimbriulatum
│   ├── Piper_sirius_matched_annotated.mgf   # Subset of only annotated spectra
│   ├── massspecgym/                    
│   │   └── MassSpecGym.mgf                  # MassSpecGym spectral library used for retrieval/matching
│   └── model_checkpoints/                   # Pretrained model weights used by DreaMS
│       ├── embedding_model.ckpt             # Finetuned DreaMS for similarity 
│       └── ssl_model.ckpt                   # Self-supervised backbone for fine-tuning/customization
├── notebooks/
│   ├── 0_notebook_tutorial.ipynb            # Start here: understanding work with notebooks and data
│   ├── 1_data_preparation.ipynb             # Loading and understanding MSData objects
│   ├── 2_deduplication.ipynb                # How to deduplicate spectra
│   ├── 3_library_matching.ipynb             # Match data to spectral libraries
│   ├── 4_molecular_networking.ipynb         # Molecular networking with DreaMS
├── LICENSE                                  # Repository license (terms for use/modification/distribution)
├── configs                               
├── README.md                                # You are here! Project overview and getting started
├── environment.yml                          # Reproducible setup for Conda
├── paths.py                                 # Unified project paths for all code 
├── scripts/                                 # Helper scripts (env install, data download, utilities)
└── setup.py                                
```

---


<h2 id="step1">🛠️ Step 1: Prerequisites</h2>

> 🪟 **Windows user?**  
> This step focuses on macOS/Linux. For Windows-specific instructions, see [💻 Windows Installation](#windows-install).

> ⚠️ **Experienced user?**  
> You can skip directly to the [Minimal setup commands](#quick-setup).


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
   - On macOS/Linux: Open your Terminal application from the Applications folder or system menu.
2. Navigate to the folder where you want this project. For example, to use your Documents folder:

    ```bash
    cd Documents
    ```

3. Download (clone) the project code from GitHub:

    Type these commands in the terminal:
    
    ```bash
    git clone https://github.com/pluskal-lab/DreaMS_MIMB.git
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

**First, make sure you’re using the terminal in the same folder as the previous step (the DreaMS_MIMB folder)**. If you closed your terminal, just open a new one and use `cd` to enter your project folder again.

To check that Conda is installed and working, type:
```bash
conda --version
```

If you see the version number, proceed! If not, [go back to Step 1](#step1).

1. **Install all packages and set up the environment**  
   Instead of manually creating an environment from `environment.yml`, we provide a script that installs all dependencies, resolves conflicts, and prepares the correct environment automatically.

    ```bash
    bash scripts/install_env.sh
    ```
   This command creates a Conda environment named `dreams_mimb`, install all required Python packages.
2. **Activate the environment**
	
	Next, activate your new workspace with:
    ```bash
    conda activate dreams_mimb
    ```
	**What does “activate” mean?**
	- By “activating” the environment, you make sure that any commands you run will use the correct versions of Python and all necessary libraries, without interfering with other projects or programs on your computer.
    - You’ll need to activate this environment every time you open a new terminal and want to work on this project.

    If you see errors about “conda command not found,” make sure you have Conda installed and restart your terminal.

3. **Download the data assets**

   We provide a script to fetch all necessary datasets and place them in the correct directories:
    ```bash
    python scripts/download_assets.py
    ```
   - After this step, your data/ folder will contain all resources needed for the tutorials.

---

<h2 id="step4">📓 Step 4: Start Jupyter Notebooks</h2>

To work with this project, you **must** use Jupyter Notebooks.
Jupyter lets you run small pieces of code (called “cells”), see the results instantly, and mix code with explanations, all in your web browser.

**(A) Recommended: JupyterLab**

In your terminal (make sure your environment is activated), type:

```bash
jupyter lab
```

- After a few seconds, your default web browser will open automatically.
If not, copy and paste the link from your terminal into your browser.


<p align="center">
  <img src="assets/DreaMS_MIMB_jupyter_setup.jpg" alt="JupyterLab setup — DreaMS_MIMB" width="500">
</p>


- You’ll see a file browser, navigate into the notebooks folder on the left side.
- Start with `1_data_preparation.ipynb` and proceed step by step.

**(B) Classic Jupyter notebook (if you prefer):**

```bash
jupyter notebook
```

- Type this command in your terminal (just like before).
- This will also open a new tab in your browser, but with a simpler interface.
- Find and open the notebooks folder, then open the first notebook.



>  **Tip:**  
> Both JupyterLab and the classic notebook work the same way for this tutorial, choose the one you like best!  
> Just remember:  
> - You **always** run `jupyter lab` or `jupyter notebook` **in your terminal** (not in a Python console or elsewhere).  
> - **Always** make sure you have activated your Conda environment before starting ([see Step 3](#step3)).

---

<h2 id="step5">📚 Step 5: Follow the Tutorials</h2>

Once your environment is set up, you're ready to explore the project through interactive Jupyter notebooks.

### 🔢 Notebook Order

- Navigate to the `notebooks` folder in the project directory.
- Start with the first notebook: [`1_data_preparation.ipynb`](#project-structure)  
(You can learn more about the project layout [here](#project-structure).)
- The notebooks are numbered in the order you should follow.
- Each notebook builds on the previous one, it's important to follow them sequentially.

### How to Use Jupyter Notebooks

Jupyter notebooks are interactive documents that combine code, text, and visualizations. Here's how to work with them:

- Notebooks are made up of “cells”, either code or explanatory text.
- Run one cell at a time:
  - Click on a cell to select it.
  - Press `Shift + Enter` to run the cell and move to the next one.
- **Always run notebooks from top to bottom**, step by step.
  - If you skip cells, later parts may not work correctly.
  - Restarting the kernel resets the notebook’s memory, you'll need to re-run all previous cells.

> **Tip:** If you're new to Jupyter, check the top menu for a ▶️ **Run** button, it's a quick way to execute the selected cell.

---
<h2 id="quick-setup"> Minimal setup commands (Mac, Linux)</h2>

```bash
git clone https://github.com/pluskal-lab/DreaMS_MIMB.git
cd DreaMS_MIMB
bash install_env.sh
conda activate dreams_mimb
python scripts/download_assets.py
jupyter lab
```

---

<h2 id="windows-install">💻 Windows Installation</h2>

## 🛠️ Prerequisites

1. **Check if Conda/Miniconda is installed**  
   On Windows, we will use the **Anaconda Prompt** for installation.  

   - Press the **Windows key**, start typing **Anaconda Prompt (miniconda3)**, then press **Enter**.  
   - You should see a black terminal window starting with something like:
     ```
     (base) C:\Users\YourName>
     ```
   - This confirms Conda is installed. To double-check, type:
     ```bash
     conda --version
     ```
     Example output:
     ```
     conda 23.3.1
     ```

   - If you don’t have Conda, install Miniconda from:  
     [https://www.anaconda.com/docs/getting-started/miniconda/main](https://www.anaconda.com/docs/getting-started/miniconda/main)  
     Download the **Windows 64-bit installer (Python 3.x)**.

2. **Check if Git is installed**  
   In the Anaconda Prompt, type:
   ```bash
   git --version
    ```
   You should now see a version like git version 2.x.x. 

---

## 📦 Step 2: Get the code
	
1.	Stay in the Anaconda Prompt.
	
2.	Navigate to the folder where you want the project. For example:    

	```bash
	 cd C:\Users\CODE 
	```  

(If the folder doesn’t exist, create it first with mkdir C:\Users\CODE.)This will be the location where the code and data for this project are saved.

3. Clone the repository:

    ```bash
    git clone https://github.com/pluskal-lab/DreaMS_MIMB.git
    cd DreaMS_MIMB
    ```
4. Verify your location with:

    ```bat
    cd
    ```
    You should see something like: `(base) C:\Users\CODE\DreaMS_MIMB>`
   The `(base)` prefix indicates Anaconda’s default environment is active.

## ⚙️ Install the Environment

From inside the repository folder (C:\Users\CODE\DreaMS_MIMB), run:
```bat
scripts\install_env_win.bat
```
- This sets up the `dreams_mimb` Conda environment automatically.  
- At the end, you’ll see a sanity check printing your Python path and PyTorch version.

---

## 🔑 Activate the Environment & Download Data

1. Activate the environment:
    ```bat
    conda activate dreams_mimb
    ```

- Your prompt will now look like:
    ```
    (dreams_mimb) C:\Users\CODE\DreaMS_MIMB>
    ```
Always activate the environment when starting a new terminal session.

2. Download the datasets:
    ```bat
    python scripts\download_assets.py
    ```

This will fill the `data/` folder with everything needed for the tutorials.  
Make sure you are inside your project folder:

    ```
    C:\Users\CODE\DreaMS_MIMB
    ```

---

## 📓 Start JupyterLab and Run Notebooks

With the environment activated and inside the project folder, type:
```bat
jupyter lab
```

- Your browser will open JupyterLab.  
- If it doesn’t, copy the URL shown in the terminal (e.g. \`http://localhost:8888/lab?...\`) and paste it into your browser.  
  For more help, see [Step 4: Start Jupyter Notebooks](#step4).

> **Tip:**  
> In JupyterLab, go to the top menu:  
> \`Kernel → Change Kernel → Python (dreams_mimb)\`  
> This ensures you’re running notebooks inside the correct environment.

When JupyterLab opens, you’re ready to continue with the tutorials:  
👉 [Go to Step 5: Follow the Tutorials](#step5)

## ⚡ Quick Setup (Windows, copy-paste)

If you’re comfortable with the terminal, here’s the complete sequence:
Open Anaconda Prompt (miniconda3) first

```bat
conda --version
git --version

cd C:\Users\PluskalLAB\CODE
git clone https://github.com/pluskal-lab/DreaMS_MIMB.git
cd DreaMS_MIMB

scripts\install_env_win.bat

conda activate dreams_mimb
python scripts\download_assets.py
jupyter lab
```

---

## 🧩 Troubleshooting

- If you run into issues:

  - **Check the Issues tab on GitHub**  
    Visit the Issues section of our GitHub page.  
    If your problem isn’t already answered there, you can create a new issue by clicking the **“New Issue”** button and describing your problem (including any error messages).

  - **Check for common problems**  
    See the **“Notes”** section in our book chapter, where we list the most common pitfalls and solutions.

- **Still stuck?**  
  If you don’t find an answer in the Notes section or we haven’t responded on GitHub, you can also use large language models (like ChatGPT or GitHub Copilot).  
  Just paste your error message and the code you’re struggling with. Often, these tools can help you quickly find a solution to common mistakes.

---

## 📖 Key Publications

The methods and tools used in this repository are based on the following peer-reviewed research:

- **Bushuiev, R. *et al.* (2025)**  
  *Self-supervised learning of molecular representations from millions of tandem mass spectra using DreaMS*  
  **Nature Biotechnology**  
  [https://www.nature.com/articles/s41587-025-02663-3](https://www.nature.com/articles/s41587-025-02663-3)

- **Bushuiev, R. *et al.* (2024)**  
  *MassSpecGym: A benchmark for the discovery and identification of molecules*  
  **NeurIPS 2024**  
  [https://arxiv.org/abs/2410.23326](https://arxiv.org/abs/2410.23326)

---

## 🤝 Contributing & Credits

Have ideas, corrections, or something to share?  
Pull requests and suggestions are always welcome!

If you spot an issue or want to improve the project, feel free to:

- Open an issue to report a bug or suggest a feature
- Submit a pull request with improvements or fixes
- Comment on existing discussions to share your input

We welcome all contributions, to help improve these tutorials and make them more useful to the community.

---

### 🙏 Acknowledgments

Special thanks to everyone who helped shape these tutorials.

This project was developed as a companion to the DreaMS book chapter, designed for researchers and students curious about **modern machine learning in mass spectrometry**, with a **hands-on, code-first approach**.






