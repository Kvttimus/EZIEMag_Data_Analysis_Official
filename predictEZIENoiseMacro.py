import subprocess
import os

# Script to run (make sure this matches actual filename)
script = "predictEZIENoise.py"

MONTH_MAP = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

year = 2024
# month = 10

# Make sure output folder exists
os.makedirs("noise_EZIE_data", exist_ok=True)

for month in range(10,13):
    month_name = MONTH_MAP.get(month, f"{month:02d}")
    # Iterate through October 7 → 27 inclusive
    for day in range(1, 32):
        day_str = f"{day:02d}"  # 07, 08, … 27

        # Original despiked EZIE
        original_path = f"despiked_EZIE_data/EZIE_{month_name}_{day}_{year}_despiked.csv"

        # Predicted EZIE from predictEZIEData.py
        predicted_path = f"predicted_EZIE_data/EZIE_{month_name}_{day}_{year}_predictions.csv"

        # Output noise file
        out_path = f"noise_EZIE_data/EZIE_{month_name}_{day}_{year}_noise.csv"

        cmd = [
            "python",
            script,
            original_path,
            predicted_path,
            out_path,
            "--train-hours", "12",
        ]

        print(f"\n=== Computing noise for {year}-{month:02d}-{day_str} ===")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
