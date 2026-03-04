import pandas as pd
import numpy as np

# =========================
# USER SETTINGS
# =========================
CSV_PATH = "daily_EZIE_ranges_2.csv"

TARGET_COL = "noise_Bh_range"   # always correlate against this
COMPARE_COL = "ctemp_max"              # set to a column name OR leave None to compute all

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(CSV_PATH)

if TARGET_COL not in df.columns:
    raise ValueError(f"{TARGET_COL} not found in {CSV_PATH}")

# Convert everything possible to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")

# =========================
# FUNCTION
# =========================
def pearson(x, y):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    if mask.sum() < 2:
        return np.nan
    return float(np.corrcoef(x[mask], y[mask])[0, 1])

# =========================
# SINGLE COLUMN MODE
# =========================
if COMPARE_COL is not None:
    if COMPARE_COL not in df.columns:
        raise ValueError(f"{COMPARE_COL} not found in CSV")

    r = pearson(df[TARGET_COL], df[COMPARE_COL])
    print(f"\nPearson r between {TARGET_COL} and {COMPARE_COL}: {r:.6f}")

# =========================
# ALL NUMERIC COLUMNS MODE
# =========================
else:
    results = []

    for col in df.columns:
        if col == TARGET_COL:
            continue
        if not np.issubdtype(df[col].dtype, np.number):
            continue

        r = pearson(df[TARGET_COL], df[col])
        results.append((col, r))

    results = sorted(results, key=lambda x: abs(x[1]) if not np.isnan(x[1]) else -1, reverse=True)

    print(f"\nPearson correlations vs {TARGET_COL}:\n")
    for col, r in results:
        print(f"{col:20s}  r = {r:.6f}")