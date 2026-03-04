# BAD FILE DO NOT USE

import os
import glob
import pandas as pd

# =========================
# USER SETTINGS (EDIT ONCE)
# =========================
DESPIKED_DIR = "despiked_EZIE_data"
DESPIKED_GLOB = "*_despiked.csv"

PRED_PATH = "out_complex/complex_predictions.csv"

OUT_DIR = "denoised_EZIE_data"

# IMPORTANT: must match the resample used when generating predictions (usually 5min)
PRED_FREQ = "5min"          # e.g. "1min" or "5min"
AGG = "mean"                # "mean" to match model training

# Columns in despiked EZIE
TIME_COL = "timeString"
BH_COL = "Bh"
CTEMP_COL = "ctemp"

# Columns in predictions
PRED_TIME_COL = "time"
YTRUE_COL = "y_true"
YPRED_COL = "y_pred"

# =========================
# LOAD PREDICTIONS (ONCE)
# =========================
pred = pd.read_csv(PRED_PATH)

need = [PRED_TIME_COL, YTRUE_COL, YPRED_COL]
missing = [c for c in need if c not in pred.columns]
if missing:
    raise ValueError(f"{PRED_PATH} missing columns: {missing}")

# Predictions timestamps are like "2025-03-29 19:10:00"
# Treat them as UTC to match EZIE timeString (+00:00)
pred["_time"] = pd.to_datetime(pred[PRED_TIME_COL], utc=True, errors="coerce")

pred[YTRUE_COL] = pd.to_numeric(pred[YTRUE_COL], errors="coerce")
pred[YPRED_COL] = pd.to_numeric(pred[YPRED_COL], errors="coerce")
pred = pred.dropna(subset=["_time", YTRUE_COL, YPRED_COL])

# ensure unique times (safe)
pred = pred.groupby("_time", as_index=False).mean(numeric_only=True)

# =========================
# PROCESS EACH DESPIKED FILE
# =========================
paths = sorted(glob.glob(os.path.join(DESPIKED_DIR, DESPIKED_GLOB)))
if not paths:
    raise FileNotFoundError(f"No files matched {DESPIKED_GLOB} in {DESPIKED_DIR}")

os.makedirs(OUT_DIR, exist_ok=True)

for p in paths:
    fname = os.path.basename(p)
    out_name = fname.replace("_despiked.csv", "_denoised.csv")
    out_path = os.path.join(OUT_DIR, out_name)

    try:
        df = pd.read_csv(p)
    except pd.errors.ParserError as e:
        print(f"Skipping {os.path.basename(p)}: CSV parse error: {e}")
        continue

    missing = [c for c in [TIME_COL, BH_COL, CTEMP_COL] if c not in df.columns]
    if missing:
        print(f"Skipping {fname}: missing columns {missing}")
        continue

    df["_time"] = pd.to_datetime(df[TIME_COL], utc=True, errors="coerce")
    df[BH_COL] = pd.to_numeric(df[BH_COL], errors="coerce")
    df[CTEMP_COL] = pd.to_numeric(df[CTEMP_COL], errors="coerce")
    df = df.dropna(subset=["_time", BH_COL, CTEMP_COL])

    # --- RESAMPLE EZIE to the same grid as predictions ---
    d = df.set_index("_time")[[BH_COL, CTEMP_COL]]

    if AGG == "mean":
        d = d.resample(PRED_FREQ).mean()
    elif AGG == "median":
        d = d.resample(PRED_FREQ).median()
    else:
        raise ValueError("AGG must be 'mean' or 'median'")

    d = d.dropna().reset_index()

    # Add a clean timeString column matching the resampled timestamps
    # (keeps +00:00 style)
    d[TIME_COL] = d["_time"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    # --- MERGE with predictions on _time ---
    merged = d.merge(pred, on="_time", how="inner")

    if merged.empty:
        # This will happen if the day doesn't overlap the prediction times
        print(f"Skipping {fname}: no overlapping timestamps with predictions")
        continue

    merged = merged.rename(
        columns={
            YTRUE_COL: "noise_Bh_actual",
            YPRED_COL: "noise_Bh_pred",
        }
    )

    merged["denoised_Bh"] = merged[BH_COL] - merged["noise_Bh_pred"]

    merged = merged[
        [
            TIME_COL,
            CTEMP_COL,
            BH_COL,
            "noise_Bh_actual",
            "noise_Bh_pred",
            "denoised_Bh",
        ]
    ]

    merged.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(merged)} rows)")

print("Done.")