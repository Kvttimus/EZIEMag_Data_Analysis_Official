"""
predictEZIEData.py

Predict EZIE magnetometer data from FRD SEC data using simple per-axis
linear mappings of the form:

    Bx ≈ a1 * FRDX + b1
    By ≈ a2 * FRDY + b2
    Bz ≈ a3 * FRDZ + b3
    Bh ≈ a4 * FRDH + b4

Workflow:
    1. Load EZIE CSV (raw)
    2. Load FRD SEC (raw, IAGA-2002 format)
    3. Use first X hours of overlapping time to "train" linear models
    4. Output a new EZIE-like CSV:
           - First X hours: real EZIE data (Bx,By,Bz,Bh unchanged)
           - Remaining times: predicted Bx,By,Bz,Bh from FRD

Notes:
    - FRD data may contain NaNs; these are skipped during training.
    - Time alignment is done with nearest-neighbor join (merge_asof)
      with a small tolerance (default 0.5 seconds).

"""

import argparse
import sys
from io import StringIO
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Loading / parsing helpers
# ---------------------------------------------------------------------

def load_ezie(ezie_path: str) -> pd.DataFrame:
    """
    Load EZIE CSV.

    Expects a CSV with at least these columns:
        tval,timeString,latitude,longitude,altitude,tres,ctemp,ccr,
        Bx,By,Bz,Bh,Ax,Ay,Az,Gx,Gy,Gz,imu_ctemp
    """
    df = pd.read_csv(ezie_path)
    return df


def load_frd(frd_path: str) -> pd.DataFrame:
    """
    Load FRD IAGA-2002 .sec file.

    Skips header lines until the line starting with "DATE",
    then parses whitespace-delimited columns:
        DATE TIME DOY FRDX FRDY FRDZ FRDF FRDH
    """
    with open(frd_path, "r") as f:
        lines = f.readlines()

    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("DATE"):
            start_idx = i
            break

    if start_idx is None:
        raise ValueError("Could not find 'DATE' header line in FRD file.")

    data_str = "".join(lines[start_idx:])
    df = pd.read_csv(
        StringIO(data_str),
        sep=r"\s+",
        na_values=["NaN", "99999.00", "99999.99"],
    )

    return df

def _get_time_series(df: pd.DataFrame, label: str) -> pd.Series:
    """
    Extract a datetime-like Series from a DataFrame and force UTC tz-aware.

    For EZIE:
        - Prefer 'timeString'
        - Fallback to 'tval' as POSIX seconds (if present)

    For FRD:
        - Expect 'DATE' and 'TIME' columns.
    """
    if {"DATE", "TIME"}.issubset(df.columns):
        # FRD case
        t = pd.to_datetime(
            df["DATE"].astype(str) + " " + df["TIME"].astype(str),
            utc=True,
        )
        return t

    if "timeString" in df.columns:
        # Mixed ISO8601 formats (with/without fractional seconds)
        t = pd.to_datetime(df["timeString"], format="ISO8601", utc=True)
        return t

    if "tval" in df.columns:
        # If tval is unix timestamp in seconds
        t = pd.to_datetime(df["tval"], unit="s", origin="unix", utc=True)
        return t

    raise ValueError(
        f"Could not infer time column for {label}; "
        f"expected either DATE/TIME or timeString/tval."
    )

# ---------------------------------------------------------------------
# Linear model helper
# ---------------------------------------------------------------------

def _fit_linear(x: np.ndarray, y: np.ndarray, name: str) -> Tuple[float, float]:
    """
    Fit a simple linear model y ≈ a * x + b via least squares, ignoring NaNs.

    Returns:
        (a, b)
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x_valid = x[mask]
    y_valid = y[mask]

    if x_valid.size < 2:
        raise RuntimeError(
            f"Not enough valid (non-NaN) data points to fit linear model for {name}."
        )

    # polyfit degree 1: y = a * x + b
    a, b = np.polyfit(x_valid, y_valid, 1)
    return float(a), float(b)


# ---------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------

def predict_ezie_from_frd(
    ezie_path: str,
    frd_path: str,
    out_path: str,
    train_hours: float = 6.0,
    time_tol_sec: float = 0.5,
    verbose: bool = True,
    coeff_csv: str | None = None,
) -> None:
    

    """
    Predict EZIE Bx,By,Bz,Bh from FRD using per-axis linear mapping.

    Args:
        ezie_path:   Path to EZIE CSV (raw).
        frd_path:    Path to FRD SEC (raw, IAGA-2002).
        out_path:    Path to write new EZIE-like CSV.
        train_hours: Number of hours of overlapping data to use for training.
        time_tol_sec: Nearest-neighbor match tolerance in seconds when
                      aligning EZIE and FRD samples.
        verbose:     If True, print fitted coefficients and basic info.
    """

    # 1. Load data
    try:
        ezie_df = load_ezie(ezie_path)
    except FileNotFoundError:
        print(f"[WARN] EZIE file not found: {ezie_path}", file=sys.stderr)
        return

    try:
        frd_df = load_frd(frd_path)
    except FileNotFoundError:
        print(f"[WARN] FRD file not found: {frd_path}", file=sys.stderr)
        return


    # Keep original EZIE column order to preserve it in output
    ezie_cols = list(ezie_df.columns)

    # 2. Construct time columns (tz-aware UTC)
    ezie_df = ezie_df.copy()
    frd_df = frd_df.copy()

    ezie_df["time"] = _get_time_series(ezie_df, "EZIE")
    frd_df["time"] = _get_time_series(frd_df, "FRD")

    # Sort by time
    ezie_df.sort_values("time", inplace=True)
    frd_df.sort_values("time", inplace=True)

    # 3. Determine overlapping window
    overlap_start = max(ezie_df["time"].min(), frd_df["time"].min())
    overlap_end = min(ezie_df["time"].max(), frd_df["time"].max())
    if overlap_end <= overlap_start:
        raise RuntimeError("No overlapping time between EZIE and FRD data.")

    # Training end time is first X hours of overlap
    train_end = overlap_start + pd.Timedelta(hours=train_hours)
    if train_end > overlap_end:
        if verbose:
            print(
                f"[WARN] Requested {train_hours} hours for training, "
                f"but only {(overlap_end - overlap_start) / pd.Timedelta(hours=1):.2f} "
                f"hours are available. Using full overlap instead.",
                file=sys.stderr,
            )
        train_end = overlap_end

    if verbose:
        print(f"Overlap start: {overlap_start}")
        print(f"Overlap end:   {overlap_end}")
        print(f"Train end:     {train_end}")

    # 4. Merge EZIE with FRD (nearest time within tolerance)
    tol = pd.Timedelta(seconds=time_tol_sec)
    frd_for_merge = frd_df[["time", "FRDX", "FRDY", "FRDZ", "FRDH"]].copy()

    merged = pd.merge_asof(
        ezie_df.sort_values("time"),
        frd_for_merge.sort_values("time"),
        on="time",
        direction="nearest",
        tolerance=tol,
    )

    # 5. Select training subset: first X hours of overlap
    train_mask = (merged["time"] >= overlap_start) & (merged["time"] < train_end)

    # Ensure EZIE magnetometer columns exist
    for col in ["Bx", "By", "Bz", "Bh"]:
        if col not in merged.columns:
            raise ValueError(f"Column '{col}' not found in EZIE data.")

    # Ensure FRD columns exist
    for col in ["FRDX", "FRDY", "FRDZ", "FRDH"]:
        if col not in merged.columns:
            raise ValueError(f"Column '{col}' not found in FRD data after merge.")

    train = merged[train_mask].copy()
    if train.empty:
        raise RuntimeError("No data in training window after alignment.")

    models: Dict[str, Tuple[float, float]] = {}

    # Fit linear models, skipping NaNs in FRD and EZIE data
    models["Bx"] = _fit_linear(
        train["FRDX"].to_numpy(), train["Bx"].to_numpy(), "Bx ~ FRDX"
    )
    models["By"] = _fit_linear(
        train["FRDY"].to_numpy(), train["By"].to_numpy(), "By ~ FRDY"
    )
    models["Bz"] = _fit_linear(
        train["FRDZ"].to_numpy(), train["Bz"].to_numpy(), "Bz ~ FRDZ"
    )
    models["Bh"] = _fit_linear(
        train["FRDH"].to_numpy(), train["Bh"].to_numpy(), "Bh ~ FRDH"
    )

    if verbose:
        print("\nFitted linear models (y ≈ a * x + b):")
        for comp in ["Bx", "By", "Bz", "Bh"]:
            a, b = models[comp]
            src = {
                "Bx": "FRDX",
                "By": "FRDY",
                "Bz": "FRDZ",
                "Bh": "FRDH",
            }[comp]
            print(f"  {comp} ≈ {a:.6f} * {src} + {b:.6f}")

    if coeff_csv is not None:
        from pathlib import Path
        import csv

        # Use the start date as a label for this run
        train_date = overlap_start.date()

        # How many points were actually used in training?
        n_train = int(train.shape[0])

        row = {
            "date": train_date,
            "train_start_utc": overlap_start.isoformat(),
            "train_end_utc": train_end.isoformat(),
            "train_hours": train_hours,
            "n_train": n_train,
            "Bx_a": models["Bx"][0],
            "Bx_b": models["Bx"][1],
            "By_a": models["By"][0],
            "By_b": models["By"][1],
            "Bz_a": models["Bz"][0],
            "Bz_b": models["Bz"][1],
            "Bh_a": models["Bh"][0],
            "Bh_b": models["Bh"][1],
        }

        coeff_path = Path(coeff_csv)
        write_header = not coeff_path.exists()

        with coeff_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        if verbose:
            print(f"\n[INFO] Appended coefficients to {coeff_path}")

    # 6. Prepare output DataFrame: start from original EZIE, add FRD for prediction
    out_df = merged.copy()

    # Prediction region: times >= train_end
    pred_mask = out_df["time"] >= train_end

    # For each component, compute predictions for pred_mask rows.
    # If FRD value is NaN at a given time, prediction is set to NaN.
    for comp, (a, b) in models.items():
        frd_col = {
            "Bx": "FRDX",
            "By": "FRDY",
            "Bz": "FRDZ",
            "Bh": "FRDH",
        }[comp]

        # Current FRD values only in the prediction window
        x = out_df.loc[pred_mask, frd_col].to_numpy(dtype=float)

        # Initialize predictions as NaN (for FRD NaNs)
        y_pred = np.full_like(x, np.nan, dtype=float)
        finite_mask = np.isfinite(x)
        y_pred[finite_mask] = a * x[finite_mask] + b

        # Overwrite Bx/By/Bz/Bh ONLY in the prediction window
        out_df.loc[pred_mask, comp] = y_pred

    # Drop helper FRD columns so the output has the same columns as original EZIE
    out_df.drop(columns=["FRDX", "FRDY", "FRDZ", "FRDH"], inplace=True)

    # Restore original column order exactly
    out_df = out_df[ezie_cols]

    # 7. Write CSV
    out_df.to_csv(out_path, index=False)

    if verbose:
        print(f"\nWrote predicted EZIE-like data to: {out_path}")
        print("Only Bx, By, Bz, Bh were modified (post-training window).")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict EZIE magnetometer data from FRD using per-axis linear mapping.\n\n"
            "Example:\n"
            "  python predictEZIEData.py EZIE_raw.csv FRD_1s.sec EZIE_predicted.csv "
            "--train-hours 6"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("ezie_path", help="Path to EZIE CSV (raw)")
    parser.add_argument("frd_path", help="Path to FRD SEC (raw IAGA-2002)")
    parser.add_argument("out_path", help="Path to output EZIE-like CSV")
    parser.add_argument(
        "--train-hours",
        "-X",
        type=float,
        default=6.0,
        help="Number of hours of overlapping data used for training (default: 6.0)",
    )
    parser.add_argument(
        "--time-tol",
        type=float,
        default=0.5,
        help="Time alignment tolerance in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose logging",
    )
    parser.add_argument(
        "--coeff-csv",
        help="Optional path to append fitted coefficients as a CSV row.",
    )

    args = parser.parse_args()

    predict_ezie_from_frd(
        ezie_path=args.ezie_path,
        frd_path=args.frd_path,
        out_path=args.out_path,
        train_hours=args.train_hours,
        time_tol_sec=args.time_tol,
        verbose=not args.quiet,
        coeff_csv=args.coeff_csv, 
    )


if __name__ == "__main__":
    main()