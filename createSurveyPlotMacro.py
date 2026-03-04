import subprocess

script = "createSurveyPlot.py"

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
# month = 12

# columns = ["FRDH", "EZIE_Bh", "EZIE_Bh_pred", "noise_Bh", "ctemp"]
# columns = ["FRDH", "Bh", "ctemp"]
columns = ["FRDH", "EZIE_Bh", "EZIE_denoised_Bh", "ctemp"]
# columns = ["ctemp_range", "Bh_range", "noise_Bh_range"]

for month in range(12, 13):
    month_name = MONTH_MAP.get(month, f"{month:02d}")
    # Iterate through October 7 → 27 inclusive
    for day in range(5,6):
        day_str = f"{day:02d}"            # 07, 08, … 27
        date_str = f"{year}-{month:02d}-{day_str}"  # 2024-10-07, etc.

        # ezie_path = f"decompressed_EZIE_data/EZIE_Oct_{day}_{year}.csv"
        # frd_path  = f"frd_data/frd{year}{month:02d}{day_str}psec.sec"
        # ezie_path = f"despiked_EZIE_data/EZIE_{month_name}_{day}_{year}_despiked.csv"
        frd_path  = f"despiked_FRD_data/frd{year}{month:02d}{day_str}psec_despiked.sec"
        # ezie_path = f"predicted_EZIE_data/EZIE_{month_name}_{day}_{year}_predictions.csv"
        # ezie_path = f"noise_EZIE_data/EZIE_{month_name}_{day}_{year}_noise.csv"
        ezie_path = f"denoised_EZIE_data/EZIE_{month_name}_{day}_{year}_denoised.csv"

        cmd = [
            "python",
            script,
            "--combined-survey",
            "--ezie", ezie_path,
            "--frd", frd_path,
            "--date", date_str,
            "--columns", *columns
        ] 

        print(f"\n=== Running for {date_str} ===")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True)
