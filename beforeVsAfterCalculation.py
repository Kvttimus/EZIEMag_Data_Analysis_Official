import os
import glob
import re
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# USER SETTINGS
# =========================
DENOISED_DIR = "merged_denoised_EZIE_data"
DENOISED_GLOB = "*_merged.csv"   # e.g. EZIE_Apr_1_2025_denoised.csv

# Must match denoised file timestamps
FREQ = "5min"
AGG = "mean"

# Columns
TIME_COL = "timeString"
BH_COL = "Bh"
DENOISED_COL = "denoised_Bh"
TRUE_COL = "Bh_pred" 

# Evaluate only last N hours of each day
LAST_HOURS = 12

# =========================
# HELPERS
# =========================
def resample_df(df: pd.DataFrame, freq: str, agg: str) -> pd.DataFrame:
    if agg == "mean":
        return df.resample(freq).mean()
    if agg == "median":
        return df.resample(freq).median()
    raise ValueError("AGG must be 'mean' or 'median'")

def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    nrmse = rmse / np.std(y_true)

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "nrmse": float(nrmse)
    }

# =========================
# MAIN
# =========================
paths = sorted(glob.glob(os.path.join(DENOISED_DIR, DENOISED_GLOB)))
if not paths:
    raise FileNotFoundError(f"No files matched {DENOISED_GLOB} in {DENOISED_DIR}")

# accumulate overall arrays
all_true = []
all_raw = []
all_denoised = []

# store per-file metrics
before_list = []
after_list = []

for p in paths:
    fname = os.path.basename(p)

    df = pd.read_csv(p)
    missing = [c for c in [TIME_COL, BH_COL, DENOISED_COL, TRUE_COL] if c not in df.columns]
    if missing:
        print(f"Skipping {fname}: missing columns {missing}")
        continue

    df["_time"] = pd.to_datetime(df[TIME_COL], utc=True, errors="coerce")
    df[BH_COL] = pd.to_numeric(df[BH_COL], errors="coerce")
    df[DENOISED_COL] = pd.to_numeric(df[DENOISED_COL], errors="coerce")
    df[TRUE_COL] = pd.to_numeric(df[TRUE_COL], errors="coerce")

    # 1) Require valid time, Bh, denoised; allow Bh_pred to be NaN for now
    df = df.dropna(subset=["_time", BH_COL, DENOISED_COL]).set_index("_time").sort_index()

    # 2) Use original data without resampling
    ezie = df[[BH_COL, DENOISED_COL, TRUE_COL]].copy().reset_index()

    if ezie.empty:
        print(f"Skipping {fname}: no rows after filtering (Bh / denoised)")
        continue

    # 3) Keep only last N hours
    ezie = ezie.dropna(subset=[TRUE_COL])

    if len(ezie) < 2:
        print(f"Skipping {fname}: not enough points with Bh_pred in this file")
        continue

    # 4) Now require Bh_pred to be present inside that window
    ezie = ezie.dropna(subset=[TRUE_COL])

    if len(ezie) < 2:
        print(f"Skipping {fname}: not enough points with Bh_pred in last {LAST_HOURS} hours")
        continue


    # Treat Bh_pred as the reference ("y_true")
    y_true = ezie[TRUE_COL].to_numpy()
    y_raw = ezie[BH_COL].to_numpy()
    y_den = ezie[DENOISED_COL].to_numpy()

    # normal
    m_before = metrics(y_true, y_raw)
    m_after = metrics(y_true, y_den)

    # ---- HIGH-FREQUENCY (DELTA) METRICS ----
    y_true_d = np.diff(y_true)
    y_raw_d  = np.diff(y_raw)
    y_den_d  = np.diff(y_den)

    m_before_d = metrics(y_true_d, y_raw_d)
    m_after_d  = metrics(y_true_d, y_den_d)

    print(f"\n=== {fname} | rows(last {LAST_HOURS}h)={len(ezie)} ===")
    print("ABSOLUTE")
    print(f"BEFORE  MAE={m_before['mae']:.3f}  RMSE={m_before['rmse']:.3f}  R^2={m_before['r2']:.4f}")
    print(f"AFTER   MAE={m_after['mae']:.3f}  RMSE={m_after['rmse']:.3f}  R^2={m_after['r2']:.4f}")

    print("DELTA (high-frequency)")
    print(f"BEFORE  MAE={m_before_d['mae']:.3f}  RMSE={m_before_d['rmse']:.3f}  R^2={m_before_d['r2']:.4f}")
    print(f"AFTER   MAE={m_after_d['mae']:.3f}  RMSE={m_after_d['rmse']:.3f}  R^2={m_after_d['r2']:.4f}")

    before_list.append(m_before)
    after_list.append(m_after)

    all_true.append(y_true)
    all_raw.append(y_raw)
    all_denoised.append(y_den)

# ===============================
# FINAL SUMMARY
# ===============================
if all_true:
    YT = np.concatenate(all_true)
    YR = np.concatenate(all_raw)
    YD = np.concatenate(all_denoised)

    # ---- NORMAL GLOBAL METRICS ----
    overall_before = metrics(YT, YR)
    overall_after  = metrics(YT, YD)

    # ---- DELTA GLOBAL METRICS ----
    YT_d = np.diff(YT)
    YR_d = np.diff(YR)
    YD_d = np.diff(YD)

    overall_before_d = metrics(YT_d, YR_d)
    overall_after_d  = metrics(YT_d, YD_d)

    print("\n==============================")
    print(f"GLOBAL (all rows combined, last {LAST_HOURS}h each day)")
    print("==============================")

    print("\nABSOLUTE")
    print(f"BEFORE  MAE={overall_before['mae']:.3f}  RMSE={overall_before['rmse']:.3f}  R^2={overall_before['r2']:.4f}")
    print(f"AFTER   MAE={overall_after['mae']:.3f}  RMSE={overall_after['rmse']:.3f}  R^2={overall_after['r2']:.4f}")

    print("\nDELTA (high-frequency)")
    print(f"BEFORE  MAE={overall_before_d['mae']:.3f}  RMSE={overall_before_d['rmse']:.3f}  R^2={overall_before_d['r2']:.4f}")
    print(f"AFTER   MAE={overall_after_d['mae']:.3f}  RMSE={overall_after_d['rmse']:.3f}  R^2={overall_after_d['r2']:.4f}")

    print(f"NRMSE={overall_before['nrmse']:.4f}")

    

    # ---- MEAN PER-FILE (absolute) ----
    avg_before = {
        "mae": float(np.mean([m["mae"] for m in before_list])),
        "rmse": float(np.mean([m["rmse"] for m in before_list])),
        "r2": float(np.mean([m["r2"] for m in before_list])),
    }

    avg_after = {
        "mae": float(np.mean([m["mae"] for m in after_list])),
        "rmse": float(np.mean([m["rmse"] for m in after_list])),
        "r2": float(np.mean([m["r2"] for m in after_list])),
    }

    print("\n==============================")
    print("MEAN PER-FILE (each day equal weight)")
    print("==============================")
    print(f"BEFORE  MAE={avg_before['mae']:.3f}  RMSE={avg_before['rmse']:.3f}  R^2={avg_before['r2']:.4f}")
    print(f"AFTER   MAE={avg_after['mae']:.3f}  RMSE={avg_after['rmse']:.3f}  R^2={avg_after['r2']:.4f}")