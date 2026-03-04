import subprocess
import os

# Script to run
script = "calculateDailyRanges.py"

# Input folder (contains ALL per-day noise CSVs)
input_dir = "noise_EZIE_data"

# Output file
out_csv = "daily_EZIE_ranges_2.csv"

if not os.path.isdir(input_dir):
    raise FileNotFoundError(f"Input folder not found: {input_dir}")

cmd = [
    "python",
    script,
    input_dir,
    out_csv,
]

print("\n=== Calculating daily EZIE ranges ===")
print(" ".join(cmd))
subprocess.run(cmd, check=True)
