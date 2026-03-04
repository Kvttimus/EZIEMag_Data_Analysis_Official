"""
findEZIEQuietDays.py

Use noise-annotated EZIE files + coefficient summary to:
  - compute per-day quality metrics (noise_std, RMSE, R^2, Bh_std, ctemp stats)
  - merge with Bh_a, Bh_b from coefficients_summary.csv
  - label days as quiet vs disturbed
  - rank days by data quality

Assumes:
  - noise files: noise_EZIE_data/EZIE_Oct_<day>_2024_noise.csv
  - coeff file: predicted_EZIE_data/coefficients_summary.csv

Usage:
  python findEZIEQuietDays.py
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------
# Per-file metrics from noise_EZIE_data/EZIE_Oct_*_2024_noise.csv
# ---------------------------------------------------------

def compute_daily_metrics_from_noise(noise_path: Path) -> dict:
    """
    Given one noise-annotated EZIE CSV, compute daily metrics:

      - date
      - n_points (post-training, where Bh_pred and noise_Bh are valid)
      - rmse (Bh vs Bh_pred)
      - r2   (Bh vs Bh_pred)
      - noise_std, noise_range
      - Bh_std  (overall variability for the day)
      - ctemp_mean, ctemp_std
    """
    df = pd.read_csv(noise_path)

    # --- Time & date ---
    # predictEZIENoise.py already creates a 'time' column
    if "time" in df.columns:
        t = pd.to_datetime(df["time"], utc=True, errors="coerce")
    elif "timeString" in df.columns:
        t = pd.to_datetime(df["timeString"], utc=True, errors="coerce")
    else:
        raise ValueError(f"{noise_path} has no 'time' or 'timeString' column.")

    df["time"] = t
    df = df.dropna(subset=["time"])
    df = df.sort_values("time")

    if df.empty:
        raise ValueError(f"{noise_path} produced an empty DataFrame after time parsing.")

    date = df["time"].dt.date.iloc[0]

    # --- Core columns we need ---
    for col in ["Bh", "Bh_pred", "noise_Bh"]:
        if col not in df.columns:
            raise ValueError(f"{noise_path} is missing required column '{col}'")

    # --- Use only rows where we actually have predictions & noise ---
    mask_valid = np.isfinite(df["Bh"]) & np.isfinite(df["Bh_pred"]) & np.isfinite(df["noise_Bh"])
    df_valid = df[mask_valid].copy()

    n_points = int(df_valid.shape[0])

    if n_points == 0:
        # No valid predicted region (maybe training window only)
        return {
            "date": date,
            "n_points": 0,
            "rmse": np.nan,
            "r2": np.nan,
            "noise_std": np.nan,
            "noise_range": np.nan,
            "Bh_std": float(np.nanstd(df["Bh"].to_numpy(dtype=float))),
            "ctemp_mean": float(np.nanmean(df["ctemp"].to_numpy(dtype=float))) if "ctemp" in df.columns else np.nan,
            "ctemp_std": float(np.nanstd(df["ctemp"].to_numpy(dtype=float))) if "ctemp" in df.columns else np.nan,
        }

    # --- RMSE & R^2 for Bh vs Bh_pred (post-training, valid region only) ---
    y = df_valid["Bh"].to_numpy(dtype=float)
    y_hat = df_valid["Bh_pred"].to_numpy(dtype=float)

    residuals = y - y_hat
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # --- Noise metrics (post-training only) ---
    noise = df_valid["noise_Bh"].to_numpy(dtype=float)
    noise_std = float(np.nanstd(noise))
    noise_range = float(np.nanmax(noise) - np.nanmin(noise)) if noise.size > 0 else np.nan

    # --- Overall Bh variability (storminess proxy, whole day) ---
    Bh_all = df["Bh"].to_numpy(dtype=float)
    Bh_std = float(np.nanstd(Bh_all))

    # --- Temperature stats (whole day) ---
    if "ctemp" in df.columns:
        ctemp = df["ctemp"].to_numpy(dtype=float)
        ctemp_mean = float(np.nanmean(ctemp))
        ctemp_std  = float(np.nanstd(ctemp))
        ctemp_range = float(np.nanmax(ctemp) - np.nanmin(ctemp))
    else:
        ctemp_mean = np.nan
        ctemp_std = np.nan
        ctemp_range = np.nan

    return {
        "date": date,
        "n_points": n_points,
        "rmse": rmse,
        "r2": r2,
        "noise_std": noise_std,
        "noise_range": noise_range,
        "Bh_std": Bh_std,
        "ctemp_mean": ctemp_mean,
        "ctemp_std": ctemp_std,
        "ctemp_range": ctemp_range,
    }


# ---------------------------------------------------------
# Main aggregation + merge with coefficients
# ---------------------------------------------------------

def main():
    noise_dir = Path("noise_EZIE_data")
    coeff_path = Path("predicted_EZIE_data/coefficients_summary.csv")

    if not noise_dir.exists():
        raise SystemExit(f"Directory not found: {noise_dir}")

    if not coeff_path.exists():
        raise SystemExit(f"Coefficient summary not found: {coeff_path}")

    # Find all daily noise files (adjust the pattern if needed)
    noise_files = sorted(noise_dir.glob("EZIE_Oct_*_2024_noise.csv"))

    if not noise_files:
        raise SystemExit(f"No noise files found in {noise_dir} matching pattern.")

    print(f"[INFO] Found {len(noise_files)} noise files.")

    # Compute per-day metrics
    records = []
    for path in noise_files:
        print(f"[INFO] Processing {path} ...")
        rec = compute_daily_metrics_from_noise(path)
        records.append(rec)

    quality_df = pd.DataFrame.from_records(records)

    # Normalize date type to string (YYYY-MM-DD) for safe merging
    quality_df["date_str"] = quality_df["date"].astype(str)

    # Load coefficients summary
    coeff_df = pd.read_csv(coeff_path)

    # Try to infer its date column name (‘date’ from our earlier design)
    if "date" not in coeff_df.columns:
        raise SystemExit(f"{coeff_path} must contain a 'date' column.")

    coeff_df["date_str"] = pd.to_datetime(coeff_df["date"]).dt.date.astype(str)

    # Merge
    merged = quality_df.merge(coeff_df, on="date_str", how="inner", suffixes=("_quality", "_coeff"))

    # For convenience, keep a single 'date' column as datetime.date
    merged["date"] = pd.to_datetime(merged["date_str"]).dt.date

    # --------------------------------------------------
    # Label quiet vs disturbed by noise_std median
    # --------------------------------------------------
    noise_median = merged["noise_std"].median()
    merged["class"] = np.where(merged["noise_std"] <= noise_median, "quiet", "disturbed")
    merged["noise_std_thresh"] = noise_median

    # Sort by noise_std ascending: quietest at top
    merged_sorted = merged.sort_values("noise_std")

    # Output directory
    out_dir = Path("analysis_FRD_EZIE")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save master CSV
    out_master = out_dir / "daily_quality_with_coeffs.csv"
    merged_sorted.to_csv(out_master, index=False)
    print(f"\n[INFO] Saved merged daily quality + coefficients to: {out_master}")

    # Print a quick summary
    print("\n[INFO] Top 5 quietest days (by noise_std):")
    cols_quick = [
        "date",
        "class",
        "noise_std",
        "rmse",
        "r2",
        "Bh_std",
        "Bh_a",
        "Bh_b",
    ]
    for col in cols_quick:
        if col not in merged_sorted.columns:
            pass

    # Be robust: only print columns that exist
    cols_quick = [c for c in cols_quick if c in merged_sorted.columns]
    print(merged_sorted[cols_quick].head(5))

    print("\n[INFO] Quiet vs disturbed counts:")
    print(merged["class"].value_counts())


if __name__ == "__main__":
    main()
