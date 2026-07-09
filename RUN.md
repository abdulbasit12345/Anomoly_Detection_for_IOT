# 🚀 Quick Start Guide: Run Anomaly Detection on IoT Traffic

This project detects Botnet and Malware attacks in IoT network traffic using machine learning (GANs) and provides plain-English explanations using AI. 

Follow this guide to set up the project, run it on one or more datasets, and view your results.

---

## ⚡ 1. First-Time Setup (Do this once)

### On macOS or Linux:
1. Open the **Terminal** app.
2. Go to the project folder:
   ```bash
   cd /Users/apple/Downloads/ANOMOLY_DETECTION_FOR_IOT
   ```
3. Copy and paste this single command to install everything automatically:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
   ```

### On Windows:
1. Open **Command Prompt** (cmd) and go to the project folder:
   ```cmd
   cd C:\path\to\ANOMOLY_DETECTION_FOR_IOT
   ```
2. Run these commands one by one:
   ```cmd
   python -m venv .venv
   .venv\Scripts\activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 📂 2. How to Add Your Datasets
Simply copy your dataset CSV files (for example: `02-14-2018.csv`, `02-15-2018.csv`) and paste them directly into the root folder of this project (the same folder that contains this `RUN.md` file).

---

## 🏃‍♂️ 3. How to Run the Code

You can process multiple files together to get combined and separate results, or you can run on a single specific file.

### Option A: Process ALL your CSV files (Separate + Combined Results) 🌟 (Recommended)
If you have multiple CSV files in the folder and want **both** separate results for each file and a combined result of all files:

* **macOS / Linux:** Double-click or run this in Terminal:
  ```bash
  ./run_multiple.sh
  ```
* **Windows:** Open Command Prompt, activate your virtual environment, and run:
  ```cmd
  python run_multiple.py
  ```

**What this does:**
1. Processes each CSV file one by one and saves the results in separate folders named after the dataset (e.g. `results_02_14_2018/`, `results_02_15_2018/`).
2. Merges all CSV files together into a single combined dataset.
3. Processes the combined dataset and saves the unified results in a folder called `results_combined/`.

---

### Option B: Run on a Single Specific File
If you want to run the pipeline on just one file and specify a custom output folder name:

* **macOS / Linux:**
  ```bash
  ./run_training.sh --csv 02-15-2018.csv --tag my_dataset_run
  ```
* **Windows:**
  ```cmd
  python run_on_file.py --csv 02-15-2018.csv --tag my_dataset_run
  ```

This will save the results inside a new folder named `results_my_dataset_run/`.

---

## 📊 4. How to View Your Results

Once the training script completes, go to your output folder (e.g., `results_combined/`, `results_02_15_2018/`, or `results/`) and open these files in any web browser (Chrome, Safari, Firefox, Edge):

1. **`reports/explanation_report.html`**
   * *This is your main dashboard!* It displays all detections, model performance metrics, and the AI-generated natural language explanations explaining why specific traffic was flagged.
2. **`reports/comparison/gan_vs_ctgan_comparison.html`**
   * This shows a side-by-side comparison of different GAN architectures to see which performs best on your data.
3. **`plots/`**
   * This folder contains generated charts of loss curves, confusion matrices, and data distribution plots.

---

## ⚙️ Advanced: Running a Fast Test Run
By default, the training runs with full settings (200k samples, 300 epochs), which can take 3 to 5 hours on a standard computer processor (CPU). 

If you want to run a quick test (taking about 10–20 minutes) to check if everything works:
1. Open the file `config.py` in a text editor.
2. Change the values near the top of the file to:
   ```python
   SAMPLE_SIZE             = 10_000
   GAN_EPOCHS              = 10
   CTGAN_EPOCHS            = 5
   GAN_SYNTHETIC_SAMPLES   = 1_000
   CTGAN_SYNTHETIC_SAMPLES = 1_000
   CLF_EPOCHS              = 5
   BERT_EPOCHS             = 1
   ```
3. Save the file and rerun the project. Remember to change them back when you are ready for the final, high-accuracy run!

---

## ❌ Troubleshooting
* **Error: `No module named 'torch'`** -> Make sure you activated your virtual environment before running the python command.
* **Error: `FileNotFoundError`** -> Make sure your CSV files are placed in the root folder of the project.
* **Slow training** -> Deep learning can take time on CPUs. If you have a modern Mac, the code will automatically use Mac GPU acceleration (MPS) for faster speeds.
