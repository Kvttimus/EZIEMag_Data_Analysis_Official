import pandas as pd
import numpy as np
from pathlib import Path

# =======================
# USER SETTINGS (EDIT ME)
# =======================
IN_CSV = "daily_EZIE_ranges_2.csv"
OUT_CSV = "daily_EZIE_ranges_filtered_2.csv"

# Toggle behavior
OVERWRITE_ORIGINAL = True   # False = create new file (default), True = overwrite original

TARGET_COL = "noise_Bh_range"  # column to threshold
THRESHOLD = 250               # values > this become NaN

# If True: try to coerce non-numeric values to numeric (bad parses -> NaN)
COERCE_TO_NUMERIC = True

# =======================
# CLEANING LOGIC
# =======================
df = pd.read_csv(IN_CSV)

if TARGET_COL not in df.columns:
    raise ValueError(f"Column '{TARGET_COL}' not found. Columns are: {list(df.columns)}")

# Ensure we can compare numerically
if COERCE_TO_NUMERIC:
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

# Replace values above threshold with NaN
mask = df[TARGET_COL] > THRESHOLD
df.loc[mask, TARGET_COL] = np.nan

# Decide output path
if OVERWRITE_ORIGINAL:
    out_path = IN_CSV
else:
    out_path = OUT_CSV

# Save CSV
df.to_csv(out_path, index=False)

print(f"Done. Replaced {mask.sum()} values in '{TARGET_COL}' > {THRESHOLD} with NaN.")
print(f"Saved: {Path(OUT_CSV).resolve()}")
