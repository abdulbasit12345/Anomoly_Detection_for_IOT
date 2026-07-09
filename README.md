# IoT Anomaly Detection (GAN + LLM)

This project detects **Botnet** and **Malware** attacks in IoT network traffic. It combines a **GAN** (to balance the dataset) and an **OpenAI LLM** (to explain predictions in plain English).

---

## ⚡ Quick Start (Default Run)

To run the pipeline on the default dataset:

1. Open your terminal.
2. Run the launcher command:
   ```bash
   ./run_training.sh
   ```

---

## 🚀 How to Run on a New CSV File (Isolated Folders)

Follow these simple steps to run the pipeline on any other dataset file (like `02-14-2018.csv`):

### Step 1: Copy your CSV file
Put your new CSV file in the root folder of the project.

### Step 2: Run the command with flags
Run `./run_training.sh` using the `--csv` flag to point to the file, and the `--tag` flag to name your separate output folder:
```bash
./run_training.sh --csv 02-14-2018.csv --tag 02_14_2018
```

### Step 3: Find your results
The script will run the training and save everything in a separate, isolated folder. 
For the command above, your results will be located in:
*   **HTML Report:** `results_02_14_2018/reports/explanation_report.html` (Open this file in a browser to see the dashboard!)
*   **Plots & Graphs:** `results_02_14_2018/plots/`
*   **Models:** `results_02_14_2018/models/`
*   **Logs:** `logs/run_02_14_2018_<timestamp>.log`
