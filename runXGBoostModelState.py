import os
import re
import glob
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# =========================
# USER SETTINGS
# =========================
MODEL_PATH = "model_states_filtered/xgb_both_5min_lesslag.joblib"
DESPIKED_DIR = "noise_EZIE_data"        # input .sec -> *_noise.csv
DESPIKED_GLOB = "*_noise.csv"
OUT_DIR = "denoised4_EZIE_data"

TIME_COL = "timeString"
BH_COL = "Bh"
CTEMP_COL = "ctemp"
TARGET_COL = "noise_Bh"                 # what model learned to predict
RESAMPLE_FREQ = "5min"                  # must match training
# =========================

# -------------------------
# Helpers
# -------------------------
def add_lags_concat(df: pd.DataFrame, col: str, lags: int) -> pd.DataFrame:
    lag_df = pd.DataFrame(
        {f"{col}_lag{k}": df[col].shift(k) for k in range(1, lags + 1)},
        index=df.index,
    )
    return pd.concat([df, lag_df], axis=1)

def add_rolling_stats_concat(df: pd.DataFrame, col: str, window: int) -> pd.DataFrame:
    s = df[col].shift(1)
    stats_df = pd.DataFrame(
        {
            f"{col}_rollmean{window}": s.rolling(window).mean(),
            f"{col}_rollstd{window}": s.rolling(window).std(),
            f"{col}_rollmax{window}": s.rolling(window).max(),
            f"{col}_rollmin{window}": s.rolling(window).min(),
            f"{col}_slope{window}": (s - s.shift(window - 1)) / max(window - 1, 1),
        },
        index=df.index,
    )
    return pd.concat([df, stats_df], axis=1)

def build_features(df: pd.DataFrame, cols: list[str], lags: int, add_stats: bool):
    for c in cols:
        df = add_lags_concat(df, c, lags)
        if add_stats:
            df = add_rolling_stats_concat(df, c, window=lags)

    feat = []
    for c in cols:
        feat += [f"{c}_lag{k}" for k in range(1, lags + 1)]
        if add_stats:
            feat += [
                f"{c}_rollmean{lags}",
                f"{c}_rollstd{lags}",
                f"{c}_rollmax{lags}",
                f"{c}_rollmin{lags}",
                f"{c}_slope{lags}",
            ]
    return df, feat

def resample_mean(df: pd.DataFrame, freq: str, cols: list[str]) -> pd.DataFrame:
    d = df[["_time"] + cols].copy()
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["_time"])
    d = d.set_index("_time").resample(freq).mean().dropna().reset_index()
    return d

MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12
}

def parse_date_from_filename(fname: str) -> datetime | None:
    m = re.search(r"EZIE_([A-Za-z]{3})_(\d{1,2})_(\d{4})_noise\.csv$", fname)
    if not m:
        return None
    mon = MONTHS.get(m.group(1))
    day = int(m.group(2))
    year = int(m.group(3))
    if not mon:
        return None
    return datetime(year, mon, day)

# -------------------------
# Load model
# -------------------------
state = joblib.load(MODEL_PATH)
model = state["model"]
feature_cols = state["feature_cols"]
args = state.get("args", {})

resample = str(state.get("resample", args.get("resample", RESAMPLE_FREQ)))
lags = int(state.get("lags", args.get("lags", 10)))   # default to 10 if missing
use_inputs = args.get("use_inputs", "both")
add_stats = bool(args.get("add_stats", False))
add_interactions = bool(args.get("add_interactions", False))
bh_col_train = args.get("bh_col", "Bh")
ct_col_train = args.get("ctemp_col", "ctemp")
target_col_train = args.get("target_col", TARGET_COL)

input_cols = [target_col_train]
if use_inputs in ("both", "bh"):
    input_cols.append(bh_col_train)
if use_inputs in ("both", "ctemp"):
    input_cols.append(ct_col_train)

print("Loaded model:", MODEL_PATH)
print("resample:", resample, "| lags:", lags, "| use_inputs:", use_inputs)

# -------------------------
# Main loop per day
# -------------------------
os.makedirs(OUT_DIR, exist_ok=True)

paths = sorted(glob.glob(os.path.join(DESPIKED_DIR, DESPIKED_GLOB)))
if not paths:
    raise FileNotFoundError(f"No files matched {DESPIKED_GLOB} in {DESPIKED_DIR}")

for p in paths:
    fname = os.path.basename(p)
    day_dt = parse_date_from_filename(fname)
    if day_dt is None:
        print(f"Skipping {fname}: couldn't parse date from filename")
        continue

    out_name = fname.replace("_noise.csv", "_denoised.csv")
    out_path = os.path.join(OUT_DIR, out_name)

    try:
        cur = pd.read_csv(p)
    except pd.errors.ParserError as e:
        print(f"Skipping {fname}: CSV parse error: {e}")
        continue

    # basic columns
    need = [TIME_COL, BH_COL, CTEMP_COL, target_col_train]
    miss = [c for c in need if c not in cur.columns]
    if miss:
        print(f"Skipping {fname}: missing {miss}")
        continue

    cur["_time"] = pd.to_datetime(cur[TIME_COL], utc=True, errors="coerce")
    cur[BH_COL] = pd.to_numeric(cur[BH_COL], errors="coerce")
    cur[CTEMP_COL] = pd.to_numeric(cur[CTEMP_COL], errors="coerce")
    cur[target_col_train] = pd.to_numeric(cur[target_col_train], errors="coerce")
    cur = cur.dropna(subset=["_time", BH_COL, CTEMP_COL, target_col_train]).sort_values("_time")

    # restrict to this day
    day_start = pd.Timestamp(day_dt, tz="UTC")
    day_end = day_start + pd.Timedelta(days=1)
    cur_day = cur[(cur["_time"] >= day_start) & (cur["_time"] < day_end)].copy()
    if cur_day.empty:
        print(f"Skipping {fname}: no rows in this day after filtering")
        continue

    # resample to 5min within the day
    cols_to_keep = [target_col_train]
    if use_inputs in ("both", "bh"):
        cols_to_keep.append(bh_col_train)
    if use_inputs in ("both", "ctemp"):
        cols_to_keep.append(ct_col_train)

    cur_rs = resample_mean(cur_day[["_time"] + cols_to_keep], resample, cols_to_keep)

    if cur_rs.empty:
        print(f"Skipping {fname}: no rows after resample")
        continue

    # build lag features on resampled grid
    cur_rs, _ = build_features(cur_rs, input_cols, lags, add_stats=add_stats)

    # interactions (must match training)
    if add_interactions and use_inputs == "both":
        cur_rs["bh_ctemp_lag1"] = cur_rs[f"{bh_col_train}_lag1"] * cur_rs[f"{ct_col_train}_lag1"]
        cur_rs["abs_dBh_lag1"] = (cur_rs[f"{bh_col_train}_lag1"] - cur_rs[f"{bh_col_train}_lag2"]).abs()
        cur_rs["abs_dBh_ctemp_lag1"] = cur_rs["abs_dBh_lag1"] * cur_rs[f"{ct_col_train}_lag1"]

    # predict only where features are complete
    valid_rows = cur_rs.dropna(subset=feature_cols).copy()
    print(fname, "resampled rows:", len(cur_rs), "valid rows:", len(valid_rows))

    noise_pred_full = np.full(len(cur_rs), np.nan, dtype=float)

    if not valid_rows.empty:
        X = valid_rows[feature_cols].values
        noise_pred_valid = model.predict(X)
        noise_pred_full[valid_rows.index.to_numpy()] = noise_pred_valid

    # build output aligned to resampled times
    out = pd.DataFrame({
        "timeString": cur_rs["_time"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "ctemp": cur_rs.get(ct_col_train, np.nan),
        "Bh": cur_rs.get(bh_col_train, np.nan),
        "noise_Bh_pred": noise_pred_full,
    })
    out["denoised_Bh"] = out["Bh"] - out["noise_Bh_pred"]

    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(out)} rows)")

print("Done.")
