import subprocess
import csv
import os

# =======================
# FIXED VARS
# =======================
script = "computeColumnCorrelation.py"

MONTH_MAP = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

# =======================
# USER SETTINGS
# =======================
year = 2025

COL_X = "ctemp"
COL_Y = "noise_Bh"
METHOD = "pearson"

HOURS = 12  # only use data where hour >= HOURS

OUT_TXT = "column_correlations.txt"
OUT_CSV = "column_correlations.csv"

# =======================
# CSV SETUP (append-safe)
# =======================
write_header = not os.path.exists(OUT_CSV)

csv_file = open(OUT_CSV, "a", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)

if write_header:
    csv_writer.writerow(["date", "pearson_correlation"])

# =======================
# MAIN LOOP
# =======================
with open(OUT_TXT, "a", encoding="utf-8") as txt_file:
    for month in range(1, 7):
        month_name = MONTH_MAP[month]

        for day in range(1, 32):
            label = f"{month_name}_{day}_{year}"
            ezie_path = f"noise_EZIE_data/EZIE_{month_name}_{day}_{year}_noise.csv"

            cmd = [
                "python",
                script,
                "--csv", ezie_path,
                "--x", COL_X,
                "--y", COL_Y,
                "--method", METHOD,
                "--label", label,
                "--hours", str(HOURS),
            ]

            print(f"\n=== Running for {label} ===")
            print(" ".join(cmd))

            try:
                result = subprocess.run(
                    cmd,
                    check=True,
                    capture_output=True,
                    text=True
                )

                line = result.stdout.strip()
                print(line)

                # ---- TXT (exact string) ----
                txt_file.write(line + "\n")

                # ---- CSV (date + value) ----
                # Example line:
                # Apr_13_2025 Pearson Correlation (hour >= 12) = 0.70472
                pearson_val = float(line.split("=")[-1].strip())
                csv_writer.writerow([label, pearson_val])

            except subprocess.CalledProcessError:
                print(f"Skipping {label} (file missing or bad data)")

csv_file.close()

print(f"\nSaved → {OUT_TXT}")
print(f"Saved → {OUT_CSV}")
