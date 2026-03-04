import os
import glob
import pandas as pd

# =========================
# SETTINGS (EDIT THESE ONLY)
# =========================
SOURCE_DIR = "noise_EZIE_data"              # has Bh_pred (lots of rows)
TARGET_DIR = "denoised_EZIE_data"           # has 288-row resampled outputs
OUT_DIR = "merged_denoised_EZIE_data"

FILE_PATTERN = "*.csv"

COLUMN_TO_COPY = "Bh_pred"

SOURCE_TIME_COL = "timeString"
TARGET_TIME_COL = "timeString"

SOURCE_NAME_REPLACE = "_noise.csv"
TARGET_NAME_REPLACE = "_denoised.csv"

# If target timestamps are exactly on 5-min boundaries and source isn't,
# turn this ON to snap both to the nearest 5 minutes before matching.
SNAP_TO_5MIN = True
# =========================

os.makedirs(OUT_DIR, exist_ok=True)
target_files = sorted(glob.glob(os.path.join(TARGET_DIR, FILE_PATTERN)))

def to_utc_time(s: pd.Series) -> pd.Series:
    t = pd.to_datetime(s.astype(str).str.strip(), utc=True, errors="coerce", format="mixed")
    if t.isna().any():
        mask = t.isna()
        t.loc[mask] = pd.to_datetime(s[mask].astype(str).str.strip(), utc=True, errors="coerce", format="ISO8601")
    return t

for tpath in target_files:
    tname = os.path.basename(tpath)

    # map denoised filename -> noise filename
    sname = tname.replace(TARGET_NAME_REPLACE, SOURCE_NAME_REPLACE)
    spath = os.path.join(SOURCE_DIR, sname)

    if not os.path.exists(spath):
        print(f"Skipping {tname}: source file missing ({sname})")
        continue

    try:
        src = pd.read_csv(spath)
        tgt = pd.read_csv(tpath)
    except Exception as e:
        print(f"Skipping {tname}: read error: {e}")
        continue

    if COLUMN_TO_COPY not in src.columns:
        print(f"Skipping {tname}: '{COLUMN_TO_COPY}' not in source")
        continue
    if TARGET_TIME_COL not in tgt.columns or SOURCE_TIME_COL not in src.columns:
        print(f"Skipping {tname}: missing timeString column in source/target")
        continue

    # parse times
    src["_time"] = to_utc_time(src[SOURCE_TIME_COL])
    tgt["_time"] = to_utc_time(tgt[TARGET_TIME_COL])

    # optionally snap to 5-min grid so 1Hz source matches 5-min target
    if SNAP_TO_5MIN:
        src["_time_key"] = src["_time"].dt.floor("5min")
        tgt["_time_key"] = tgt["_time"].dt.floor("5min")
    else:
        src["_time_key"] = src["_time"]
        tgt["_time_key"] = tgt["_time"]

    # reduce source to one row per time_key (mean is fine; could also use last())
    src[COLUMN_TO_COPY] = pd.to_numeric(src[COLUMN_TO_COPY], errors="coerce")
    src_small = (
        src.dropna(subset=["_time_key"])[["_time_key", COLUMN_TO_COPY]]
        .groupby("_time_key", as_index=False)
        .mean(numeric_only=True)
    )

    # left-merge into target (keeps target row count)
    out = tgt.merge(src_small, on="_time_key", how="left", suffixes=("", "_src"))

    matched = out[COLUMN_TO_COPY].notna().sum()
    total = len(out)

    # drop helper cols, keep original timeString untouched
    out = out.drop(columns=["_time", "_time_key"], errors="ignore")

    out_path = os.path.join(OUT_DIR, tname)
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path} | matched {matched}/{total} rows")

print("Done.")