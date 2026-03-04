import subprocess
import os

# =======================
# FIXED VARS
# =======================
script = "computeActualNoise.py"

MONTH_MAP = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}

# =======================
# USER SETTINGS
# =======================
year = 2025
months = [1, 6]          # e.g. [10] for Oct only, or range(1, 13) for whole year
days = range(1, 32)    # loop 1..31 (missing files will be skipped)
TIME_TOL_SEC = 0.5

EZIE_DIR = "decompressed_EZIE_data"
FRD_DIR = "frd_data"
OUT_DIR = "computedActualNoise_EZIE_data"

# Match frd filename format
FRD_NAME_FMT = "frd{yyyymmdd}psec.sec"

# =======================
# MAIN LOOP
# =======================
os.makedirs(OUT_DIR, exist_ok=True)

for month in months:
    month_name = MONTH_MAP[month]

    for day in days:
        label = f"{month_name}_{day}_{year}"

        ezie_path = f"{EZIE_DIR}/EZIE_{month_name}_{day}_{year}.csv"

        # yyyymmdd with zero padding
        yyyymmdd = f"{year}{month:02d}{day:02d}"
        frd_path = f"{FRD_DIR}/" + FRD_NAME_FMT.format(yyyymmdd=yyyymmdd)

        out_path = f"{OUT_DIR}/EZIE_{month_name}_{day}_{year}_physical_noise.csv"

        cmd = [
            "python", script,
            "--ezie", ezie_path,
            "--frd", frd_path,
            "--out", out_path,
            "--time-tol", str(TIME_TOL_SEC),
        ]

        print(f"\n=== Running for {label} ===")
        print(" ".join(cmd))

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"[OK] Wrote → {out_path}")
        except subprocess.CalledProcessError as e:
            # Common: missing file / empty merge / parse errors
            print(f"Skipping {label} (missing/bad data)")
            # Error Test
            # print(e.stdout)
            # print(e.stderr)