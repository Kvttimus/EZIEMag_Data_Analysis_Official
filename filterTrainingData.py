"""
Filter daily EZIE training files based on noise_Bh_range.

Logic:
- Read daily summary CSV (one row per day)
- If noise_Bh_range > THRESHOLD:
    -> exclude that day (do NOT copy the file)
- Else:
    -> copy the entire raw daily CSV into FILTERED_DIR

Run:
    python filterTrainingData.py
"""

from pathlib import Path
import shutil
import pandas as pd

# ============================================================
# HARD-CODED SETTINGS (EDIT THESE IF NEEDED)
# ============================================================

SUMMARY_CSV = Path("daily_EZIE_ranges_2.csv")     # daily summary file
# RAW_DATA_DIR = Path("noise_EZIE_data")          # directory with raw daily CSVs
RAW_DATA_DIR = Path("temp_data")
FILTERED_DIR = Path("temp_data2")
# FILTERED_DIR = Path("filtered_training_data_2")  # output directory

NOISE_RANGE_COLUMN = "noise_Bh_range"
FILENAME_COLUMN = "file"

THRESHOLD = 500.0   # exclude days where noise_Bh_range > THRESHOLD

# ============================================================
# MAIN LOGIC
# ============================================================

def main():
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Summary CSV not found: {SUMMARY_CSV}")

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DATA_DIR}")

    FILTERED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SUMMARY_CSV)

    # Validate required columns
    for col in [NOISE_RANGE_COLUMN, FILENAME_COLUMN]:
        if col not in df.columns:
            raise ValueError(
                f"Missing required column '{col}' in {SUMMARY_CSV}. "
                f"Found columns: {list(df.columns)}"
            )

    # Ensure numeric
    df[NOISE_RANGE_COLUMN] = pd.to_numeric(df[NOISE_RANGE_COLUMN], errors="coerce")

    # Split keep vs drop
    drop_mask = df[NOISE_RANGE_COLUMN] > THRESHOLD
    keep_df = df[~drop_mask].copy()
    drop_df = df[drop_mask].copy()

    copied = 0
    missing_files = []

    for fname in keep_df[FILENAME_COLUMN].astype(str):
        src = RAW_DATA_DIR / fname
        dst = FILTERED_DIR / fname

        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing_files.append(fname)

    # Save audit files
    keep_df.to_csv(FILTERED_DIR / "filtered_daily_EZIE_ranges_2.csv", index=False)
    drop_df.to_csv(FILTERED_DIR / "filtered_out_days_2.csv", index=False)

    if missing_files:
        pd.DataFrame({"missing_file": missing_files}).to_csv(
            FILTERED_DIR / "missing_files.csv", index=False
        )

    # Summary
    print("======================================")
    print("EZIE training data filtering complete")
    print("--------------------------------------")
    print(f"Noise threshold        : {THRESHOLD}")
    print(f"Days kept              : {len(keep_df)}")
    print(f"Days excluded          : {len(drop_df)}")
    print(f"Files copied           : {copied}")
    print(f"Missing files          : {len(missing_files)}")
    print(f"Output directory       : {FILTERED_DIR.resolve()}")
    print("======================================")

if __name__ == "__main__":
    main()
