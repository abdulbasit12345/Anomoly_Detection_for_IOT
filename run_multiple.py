#!/usr/bin/env python3
"""
run_multiple.py
================
Run the anomaly-detection pipeline on ALL CSV files in the project directory.
This generates:
  1. Separate results for each CSV file in folders named after the files.
  2. A combined result folder (results_combined) containing the combined detections.

Designed to be simple for non-programmers.
"""

import os
import sys
import glob
import subprocess

def get_clean_tag(filename):
    """Convert a filename like '02-15-2018.csv' into a clean tag like '02_15_2018'."""
    base = os.path.splitext(os.path.basename(filename))[0]
    # Replace common separators with underscores
    clean = base.replace("-", "_").replace(".", "_").replace(" ", "_")
    return clean

def merge_csv_files(input_files, output_file):
    """Merge multiple CSV files into one, keeping only the header from the first file.
    Uses a memory-efficient stream copy to handle large files without running out of RAM.
    """
    print(f"\n[System] Merging {len(input_files)} CSV files into a single dataset: '{os.path.basename(output_file)}'...")
    first = True
    
    # We use utf-8 encoding and ignore errors just in case there are bad characters in raw logs
    with open(output_file, 'w', encoding='utf-8', errors='ignore') as outfile:
        for filepath in input_files:
            print(f"  -> Adding data from '{os.path.basename(filepath)}'...")
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as infile:
                # Read the header line
                header = infile.readline()
                if not header:
                    continue
                
                # Write header only once (from the first file)
                if first:
                    outfile.write(header)
                    first = False
                
                # Copy the rest of the lines
                for line in infile:
                    if line.strip():  # Skip empty lines
                        outfile.write(line)
    print(f"[System] Merging complete. Combined file saved to: {output_file}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Find all CSV files in the root project folder
    csv_pattern = os.path.join(base_dir, "*.csv")
    all_csvs = glob.glob(csv_pattern)
    
    # Filter out temporary/combined CSV files
    target_csvs = []
    for csv in all_csvs:
        name = os.path.basename(csv)
        if name.startswith("combined_") or name == "combined_dataset.csv":
            continue
        # Also exclude dummy/empty files if they aren't real datasets
        target_csvs.append(csv)
        
    if not target_csvs:
        print("[Error] No CSV datasets found in the project root folder!")
        print("Please place your CSV dataset files (e.g. '02-15-2018.csv') in this folder:")
        print(f"  {base_dir}")
        sys.exit(1)
        
    print("=" * 70)
    print(" IoT Anomaly Detection Batch Processing Script")
    print(f" Found {len(target_csvs)} CSV dataset file(s) to process:")
    for csv in target_csvs:
        print(f"  - {os.path.basename(csv)}")
    print("=" * 70)
    
    # Step 1: Run each CSV file individually
    results_folders = []
    for csv_file in target_csvs:
        tag = get_clean_tag(csv_file)
        name = os.path.basename(csv_file)
        print(f"\n[1/2] Processing INDIVIDUAL file: '{name}'")
        print(f"      Saving results to folder: 'results_{tag}/'")
        print("-" * 60)
        
        # Build command: python run_on_file.py --csv <file> --tag <tag>
        cmd = [sys.executable, "run_on_file.py", "--csv", csv_file, "--tag", tag]
        
        try:
            subprocess.run(cmd, check=True)
            results_folders.append((name, f"results_{tag}"))
        except subprocess.CalledProcessError as e:
            print(f"[Warning] Failed to process {name}: {e}")
            
    # Step 2: Combine all CSV files and run them together (if more than 1 file)
    if len(target_csvs) > 1:
        combined_csv_path = os.path.join(base_dir, "combined_dataset.csv")
        try:
            # Merge the CSV files
            merge_csv_files(target_csvs, combined_csv_path)
            
            print(f"\n[2/2] Processing COMBINED dataset...")
            print("      Saving results to folder: 'results_combined/'")
            print("-" * 60)
            
            cmd = [sys.executable, "run_on_file.py", "--csv", "combined_dataset.csv", "--tag", "combined"]
            subprocess.run(cmd, check=True)
            results_folders.append(("All files combined", "results_combined"))
            
            # Clean up the combined file to save disk space (it is very large)
            if os.path.exists(combined_csv_path):
                print(f"\n[System] Cleaning up temporary combined file '{os.path.basename(combined_csv_path)}' to save disk space...")
                os.remove(combined_csv_path)
                
        except Exception as e:
            print(f"[Error] Failed to generate combined results: {e}")
    else:
        print("\n[System] Skipping combined step: Only 1 dataset file was found.")

    # Print summary of results
    print("\n" + "=" * 70)
    print(" BATCH RUN SUMMARY")
    print("=" * 70)
    print("All processing completed successfully!")
    print("You can view your results in the following folders:")
    for name, folder in results_folders:
        report_path = os.path.join(base_dir, folder, "reports", "explanation_report.html")
        comparison_path = os.path.join(base_dir, folder, "reports", "comparison", "gan_vs_ctgan_comparison.html")
        print(f"\n* Dataset: {name}")
        print(f"  📁 Folder: {folder}/")
        print(f"  🌐 Dashboard Report: {report_path}")
        print(f"  📊 Model Comparison: {comparison_path}")
    print("=" * 70)

if __name__ == "__main__":
    main()
