import pandas as pd
import numpy as np
import argparse
import sys

# ------------------------------------------------------------
# Helper: load EZIE-style CSV and construct a 'time' column
# ------------------------------------------------------------
def load_with_time(path: str) -> pd.DataFrame:
    """
    Load an EZIE-like CSV and parse 'timeString' into a tz-aware UTC
    'time' column. Keeps all original columns.
    """
    df = pd.read_csv(path)

    if "timeString" not in df.columns:
        raise ValueError(f"{path} must contain a 'timeString' column.")

    ts = df["timeString"].astype(str).str.strip()

    # Try mixed ISO (with/without fractional seconds)
    df["time"] = pd.to_datetime(ts, utc=True, errors="coerce", format="mixed")

    # Fallback to strict ISO8601 where needed
    if df["time"].isna().any():
        mask = df["time"].isna()
        df.loc[mask, "time"] = pd.to_datetime(
            ts[mask],
            utc=True,
            errors="coerce",
            format="ISO8601",
        )

    # Optional last fallback: if tval exists and time is still NaN
    if df["time"].isna().any() and "tval" in df.columns:
        mask = df["time"].isna()
        df.loc[mask, "time"] = pd.to_datetime(
            df.loc[mask, "tval"],
            unit="s",
            origin="unix",
            utc=True,
            errors="coerce",
        )

    df = df.dropna(subset=["time"])
    df = df.sort_values("time")
    return df


# ------------------------------------------------------------
# Core noise computation
# ------------------------------------------------------------
def compute_noise(
    original_path: str,
    predicted_path: str,
    out_path: str,
    train_hours: float = 12.0,
    time_tol_sec: float = 0.5,
) -> None:
    """
    Combine original despiked EZIE and FRD-predicted EZIE into one file,
    adding *_pred and noise_* columns.

    Behavior:
      - Original Bx,By,Bz,Bh are NEVER modified.
      - After merge, we get columns like Bx (orig) and Bx_pred (from predicted file).
      - For the first `train_hours`:
            Bx_pred,By_pred,Bz_pred,Bh_pred are set to NaN
            noise_B* are NaN as well
      - For times >= train_end:
            B*_pred contain the merged predicted values
            noise_B* = B* - B*_pred
    """
    # Load both files with a consistent 'time' column
    try:
        orig = load_with_time(original_path)
    except FileNotFoundError:
        print(f"[WARN] original file not found: {original_path}", file=sys.stderr)
        return

    try:
        pred = load_with_time(predicted_path)
    except FileNotFoundError:
        print(f"[WARN] predicted file not found: {predicted_path}", file=sys.stderr)
        return


    # Merge on nearest time within tolerance
    merged = pd.merge_asof(
        orig.sort_values("time"),
        pred.sort_values("time"),
        on="time",
        suffixes=("", "_pred"),
        direction="nearest",
        tolerance=pd.Timedelta(seconds=time_tol_sec),
    )

    if merged.empty:
        raise RuntimeError("Merged DataFrame is empty; check time overlap and tolerance.")

    # Determine training window
    start_time = merged["time"].min()
    train_end = start_time + pd.Timedelta(hours=train_hours)
    train_mask = merged["time"] < train_end

    # Magnetometer components we care about
    components = ["Bx", "By", "Bz", "Bh"]

    for comp in components:
        pred_col = f"{comp}_pred"
        noise_col = f"noise_{comp}"

        if pred_col not in merged.columns:
            raise ValueError(
                f"Expected column '{pred_col}' in merged data but did not find it. "
                f"Does the predicted file contain '{comp}' so it can become '{pred_col}'?"
            )

        # For training hours: *_pred should be NaN
        merged.loc[train_mask, pred_col] = np.nan

        # Compute noise only when we have both original and predicted (post-training)
        diff = merged[comp] - merged[pred_col]
        # noise is NaN in training window, diff elsewhere (may still be NaN if pred is NaN)
        merged[noise_col] = np.where(train_mask, np.nan, diff)

    # # ------------------------------------------------------------
    # # Daily range calculations
    # # ------------------------------------------------------------

    # # Create a UTC day column
    # merged["day"] = merged["time"].dt.floor("D")

    # # ctemp daily range
    # if "ctemp" in merged.columns:
    #     merged["ctemp_range_day"] = (
    #         merged.groupby("day")["ctemp"]
    #         .transform(lambda x: x.max() - x.min())
    #     )
    # else:
    #     merged["ctemp_range_day"] = np.nan
    #     print("[WARN] 'ctemp' column not found; ctemp_range_day set to NaN.")

    # # EZIE Bh noise daily range (post-training window only)
    # if "noise_Bh" in merged.columns:
    #     merged["eziebhnoise_range_day"] = (
    #         merged.groupby("day")["noise_Bh"]
    #         .transform(lambda x: x.max() - x.min())
    #     )
    # else:
    #     merged["eziebhnoise_range_day"] = np.nan
    #     print("[WARN] 'noise_Bh' column not found; eziebhnoise_range_day set to NaN.")

    # # Optional: drop helper column
    # merged = merged.drop(columns=["day"])


    # Write out the combined file
    merged.to_csv(out_path, index=False)
    print(f"[OK] Wrote noise-annotated EZIE file to: {out_path}")
    print(f"  - Original Bx,By,Bz,Bh were not modified.")
    print(f"  - *_pred and noise_* valid only after {train_hours} hours from {start_time}.")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute EZIE noise by comparing original vs FRD-predicted EZIE files."
    )
    parser.add_argument("original_path", help="Path to original despiked EZIE CSV")
    parser.add_argument("predicted_path", help="Path to FRD-predicted EZIE CSV")
    parser.add_argument("out_path", help="Path to output noise-annotated EZIE CSV")
    parser.add_argument(
        "--train-hours",
        type=float,
        default=12.0,
        help="Number of initial hours treated as training window (default: 12.0). "
             "In this window, *_pred and noise_* are set to NaN.",
    )
    parser.add_argument(
        "--time-tol",
        type=float,
        default=0.5,
        help="Time alignment tolerance in seconds for merge_asof (default: 0.5).",
    )
    args = parser.parse_args()

    compute_noise(
        original_path=args.original_path,
        predicted_path=args.predicted_path,
        out_path=args.out_path,
        train_hours=args.train_hours,
        time_tol_sec=args.time_tol,
    )


if __name__ == "__main__":
    main()
