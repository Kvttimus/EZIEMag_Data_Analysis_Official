import subprocess

# Script to run
script = "predictEZIEData.py"

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

year = 2025
# month = 10

# How many hours of overlap to use for training
train_hours = 12.0
time_tol = 0.5  # seconds

coeff_csv = "predicted_EZIE_data/coefficients_summary.csv"

for month in range(1, 7):
    month_name = MONTH_MAP.get(month, f"{month:02d}")
    # Iterate through October 7 → 27 inclusive
    for day in range(1, 32):
        day_str = f"{day:02d}"              # 07, 08, … 27

        # Match existing naming style
        # Input EZIE (despiked)
        ezie_path = f"despiked_EZIE_data/EZIE_{month_name}_{day}_{year}_despiked.csv"

        # Input FRD (despiked)
        frd_path = f"despiked_FRD_data/frd{year}{month:02d}{day_str}psec_despiked.sec"

        # Output EZIE with predicted FRD-based Bx,By,Bz,Bh
        out_path = f"predicted_EZIE_data/EZIE_{month_name}_{day}_{year}_predictions.csv"

        log_path = f"predicted_EZIE_data/coefficient_info/EZIE_{month_name}_{day}_{year}_fit_summary.txt"

        cmd = [
            "python",
            script,
            ezie_path,
            frd_path,
            out_path,
            # log_path,
            "--train-hours", str(train_hours),
            "--time-tol", str(time_tol),
            "--coeff-csv", coeff_csv,
        ]

        print(f"\n=== Running predictions for {year}-{month:02d}-{day_str} ===")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
