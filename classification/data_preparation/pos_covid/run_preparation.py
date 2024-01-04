import subprocess
import sys

# List of Python scripts to run sequentially
scripts_to_run = [
    "01_feature_generation.py",
    "01_variable_encoding.py",
    "02_mv_imputation.py",
    "03_outliers.py",
    "04_scaling.py",
    "05_data_partition.py",
    "06_balancing.py",
    "07_feature_selection.py",
    "08_feature_extraction.py"
]

# Using sys.executable ensures the same Python interpreter is used
python_interpreter = sys.executable

for script in scripts_to_run:
    print(f"Running script: {script}")
    subprocess.run([python_interpreter, script])

print("All scripts executed.")
