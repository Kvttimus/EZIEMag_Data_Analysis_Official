"""
analyzeFixedBhModel.py

Goal:
  1. Use the 5 quietest days (lowest noise_std) to build a fixed Bh model:
         Bh_fixed ≈ a* * FRDH + b*
  2. For ALL days:
         - load the noise_EZIE_data file
         - align with despiked FRD
         - apply the fixed model
         - compute RMSE_fixed and R2_fixed (post-training region only)
  3. Save a new CSV with both:
         - daily model metrics (rmse, r2, Bh_a, Bh_b, ...)
         - fixed model metrics (rmse_fixed, r2_fixed)
  4. Print:
         - Quiet vs disturbed RMSE (daily vs fixed)
         - Quiet vs disturbed Bh_a mean/std (slope stability)

Input:
  - analysis_FRD_EZIE/daily_quality_with_coeffs.csv  (from previous script)
  - noise_EZIE_data/EZIE_Oct_<day>_2024_noise.csv
  - despiked_FRD_data/frdYYYYMMDDpsec_despiked.sec

Usage:
  python analyzeFixedBhModel.py
"""

from pathlib import Path
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Helpers to load EZIE + FRD
# ------------------------------------------------------------

def load_noise_ezie(path: Path) -> pd.DataFrame:
    """
    Load a noise_EZIE_data CSV and get a proper 'time' column (tz-aware UTC).

    Expected columns (from predictEZIENoise.py):
      - time (preferred) OR timeString OR tval
      - Bh, noise_Bh
    """
    df = pd.read_csv(path)

    # Time parsing
    if "time" in df.columns:
        t = pd.to_datetime(df["time"], utc=True, errors="coerce")
    elif "timeString" in df.columns:
        t = pd.to_datetime(df["timeString"], utc=True, errors="coerce")
    elif "tval" in df.columns:
        t = pd.to_datetime(df["tval"], unit="s", origin="unix", utc=True, errors="coerce")
    else:
        raise ValueError(f"{path} has no 'time', 'timeString', or 'tval' column.")

    df["time"] = t
    df = df.dropna(subset=["time"])
    df = df.sort_values("time")

    # Basic sanity
    for col in ["Bh", "noise_Bh"]:
        if col not in df.columns:
            raise ValueError(f"{path} is missing column '{col}'")

    return df


def load_frd_sec(path: Path) -> pd.DataFrame:
    """
    Load a despiked FRD IAGA-2002 .sec file.

    Looks for the header line starting with 'DATE', then parses whitespace columns:
      DATE TIME DOY FRDX FRDY FRDZ FRDF FRDH
    """
    if not path.exists():
        raise FileNotFoundError(f"FRD file not found: {path}")

    with open(path, "r") as f:
        lines = f.readlines()

    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("DATE"):
            start_idx = i
            break

    if start_idx is None:
        raise ValueError(f"Could not find 'DATE' header in FRD file: {path}")

    data_str = "".join(lines[start_idx:])

    # Use io.StringIO in a way that works regardless of pandas version
    import io
    df = pd.read_csv(
        io.StringIO(data_str),
        sep=r"\s+",
        na_values=["NaN", "99999.00", "99999.99"],
    )

    # Expect DATE, TIME, FRDH
    for col in ["DATE", "TIME", "FRDH"]:
        if col not in df.columns:
            raise ValueError(f"{path} is missing required column '{col}'")

    df["time"] = pd.to_datetime(
        df["DATE"].astype(str) + " " + df["TIME"].astype(str),
        utc=True,
        errors="coerce",
    )
    df = df.dropna(subset=["time"])
    df = df.sort_values("time")
    return df[["time", "FRDH"]]


# ------------------------------------------------------------
# Fixed-model evaluation
# ------------------------------------------------------------

def compute_fixed_metrics_for_day(
    date, a_star, b_star,
    noise_dir: Path,
    frd_dir: Path,
    time_tol_sec: float = 0.5,
) -> dict:
    """
    For a given date, apply the fixed model:
        Bh_fixed = a_star * FRDH + b_star
    and compute RMSE_fixed, R2_fixed on the evaluation region
    (rows where noise_Bh is finite, i.e., post-training).

    Returns a dict with:
      - date
      - rmse_fixed
      - r2_fixed
    """
    year = date.year
    month = date.month
    day = date.day

    # Only have October in this dataset; naming follows EZIE_Oct_<day>_2024_noise.csv
    month_str_map = {10: "Oct"}  # extend for when more months are added later
    if month not in month_str_map:
        raise ValueError(f"No filename mapping defined for month={month}")

    month_tag = month_str_map[month]

    noise_path = noise_dir / f"EZIE_{month_tag}_{day}_{year}_noise.csv"
    frd_path = frd_dir / f"frd{year}{month:02d}{day:02d}psec_despiked.sec"

    # Load data
    ezie = load_noise_ezie(noise_path)
    frd = load_frd_sec(frd_path)

    # Align FRDH with EZIE times
    merged = pd.merge_asof(
        ezie.sort_values("time"),
        frd.sort_values("time"),
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=time_tol_sec),
    )

    # Need Bh, noise_Bh, FRDH all finite
    for col in ["Bh", "noise_Bh", "FRDH"]:
        if col not in merged.columns:
            raise ValueError(f"{noise_path} / {frd_path} merge missing '{col}'")

    mask_eval = (
        np.isfinite(merged["Bh"]) &
        np.isfinite(merged["noise_Bh"]) &
        np.isfinite(merged["FRDH"])
    )
    eval_df = merged[mask_eval].copy()

    if eval_df.empty:
        return {
            "date": date,
            "rmse_fixed": np.nan,
            "r2_fixed": np.nan,
        }

    Bh = eval_df["Bh"].to_numpy(dtype=float)
    FRDH = eval_df["FRDH"].to_numpy(dtype=float)

    Bh_fixed = a_star * FRDH + b_star
    residual = Bh - Bh_fixed

    rmse_fixed = float(np.sqrt(np.mean(residual ** 2)))

    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((Bh - np.mean(Bh)) ** 2))
    r2_fixed = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "date": date,
        "rmse_fixed": rmse_fixed,
        "r2_fixed": r2_fixed,
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    # Input master file from previous script
    base_dir = Path("analysis_FRD_EZIE")
    master_path = base_dir / "daily_quality_with_coeffs.csv"

    if not master_path.exists():
        raise SystemExit(f"Master daily file not found: {master_path}")

    df = pd.read_csv(master_path)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    # de-duplicate dates (keep best-noise row per date)
    df = df.sort_values(["date", "noise_std"])
    df = df.groupby("date", as_index=False).first()

    # Pick the 5 quietest days by noise_std
    df_sorted_by_noise = df.sort_values("noise_std")
    quiet5 = df_sorted_by_noise.head(5).copy()

    print("[INFO] Using these 5 quietest days to build fixed Bh model:")
    print(quiet5[["date", "noise_std", "rmse", "r2", "Bh_a", "Bh_b"]])

    # Compute fixed model parameters (median over quiet days)
    a_star = float(quiet5["Bh_a"].median())
    b_star = float(quiet5["Bh_b"].median())

    print(f"\n[INFO] Fixed model parameters from quiet days:")
    print(f"  a* (slope)  = {a_star:.6f}")
    print(f"  b* (offset) = {b_star:.3f}")
    print("  Bh_fixed ≈ a* * FRDH + b*")

    # Evaluate fixed model on all days
    noise_dir = Path("noise_EZIE_data")
    frd_dir = Path("despiked_FRD_data")

    fixed_records = []
    for _, row in df.iterrows():
        date = row["date"]
        print(f"\n[INFO] Evaluating fixed model for {date} ...")
        rec = compute_fixed_metrics_for_day(date, a_star, b_star, noise_dir, frd_dir)
        fixed_records.append(rec)

    fixed_df = pd.DataFrame(fixed_records)
    fixed_df["date"] = pd.to_datetime(fixed_df["date"]).dt.date

    # Merge back into original df on date
    df_merged = df.merge(fixed_df, on="date", how="left")

    # Save
    out_path = base_dir / "daily_quality_with_coeffs_and_fixed.csv"
    df_merged.to_csv(out_path, index=False)
    print(f"\n[INFO] Saved daily metrics + fixed model metrics to: {out_path}")

    # --------------------------------------------------------
    # Step B: Quiet vs Disturbed - RMSE (daily vs fixed)
    # --------------------------------------------------------
    print("\n[INFO] Quiet vs Disturbed – RMSE (daily vs fixed):")
    if "class" in df_merged.columns:
        for cls in ["quiet", "disturbed"]:
            sub = df_merged[df_merged["class"] == cls]
            if sub.empty:
                continue
            print(f"\n  Class: {cls}")
            print(f"    n_days = {len(sub)}")
            print(f"    mean rmse (daily) = {sub['rmse'].mean():.3f}")
            print(f"    mean rmse (fixed) = {sub['rmse_fixed'].mean():.3f}")
    else:
        print("  (No 'class' column found; skipping class-based RMSE summary.)")

    # --------------------------------------------------------
    # Step A: slope stability - Bh_a stats for quiet vs disturbed
    # --------------------------------------------------------
    if "class" in df_merged.columns and "Bh_a" in df_merged.columns:
        quiet = df_merged[df_merged["class"] == "quiet"]
        dist = df_merged[df_merged["class"] == "disturbed"]

        print("\n[INFO] Bh_a (slope) statistics by class:")

        if not quiet.empty:
            print("\n  Quiet days:")
            print(f"    n_days       = {len(quiet)}")
            print(f"    Bh_a mean    = {quiet['Bh_a'].mean():.6f}")
            print(f"    Bh_a std     = {quiet['Bh_a'].std():.6f}")
        if not dist.empty:
            print("\n  Disturbed days:")
            print(f"    n_days       = {len(dist)}")
            print(f"    Bh_a mean    = {dist['Bh_a'].mean():.6f}")
            print(f"    Bh_a std     = {dist['Bh_a'].std():.6f}")
    else:
        print("\n[INFO] Skipping Bh_a slope stats (missing 'class' or 'Bh_a').")


if __name__ == "__main__":
    main()
