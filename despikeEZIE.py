import numpy as np
import pandas as pd
from pathlib import Path


def _detect_spikes(series, window=101, z_thresh=8.0):
    """
    Detect spikes in a 1D Series using a rolling median + MAD robust z-score.
    Returns a boolean mask (True = spike).
    """
    # Rolling median (local baseline)
    med = series.rolling(window, center=True, min_periods=1).median()

    # Absolute deviation from local median
    abs_dev = (series - med).abs()

    # Rolling MAD
    mad = abs_dev.rolling(window, center=True, min_periods=1).median()
    eps = 1e-9

    # Robust z-score (scale invariant)
    robust_z = abs_dev / (1.4826 * mad + eps)

    vals = series.values

    # Local peak / trough condition so we only catch sharp blips
    if len(vals) >= 3:
        is_peak = np.r_[False, (vals[1:-1] > vals[:-2]) & (vals[1:-1] > vals[2:]), False]
        is_trough = np.r_[False, (vals[1:-1] < vals[:-2]) & (vals[1:-1] < vals[2:]), False]
    else:
        is_peak = np.zeros_like(vals, dtype=bool)
        is_trough = np.zeros_like(vals, dtype=bool)

    is_extreme = is_peak | is_trough

    spikes = (robust_z > z_thresh) & is_extreme
    return spikes


def _despike_and_interpolate(df, cols, time_col="tval",
                             window=101, z_thresh=8.0):
    """
    For each column in `cols`, detect spikes, replace them with NaN,
    then interpolate in time to fill the gaps.

    Returns a new DataFrame.
    """
    df = df.sort_values(time_col).copy()
    df = df.set_index(time_col)

    for col in cols:
        if col not in df.columns:
            print(f"[WARN] Column '{col}' not found in DataFrame, skipping.")
            continue

        s = df[col].astype(float)

        # 1. detect spikes
        spikes = _detect_spikes(s, window=window, z_thresh=z_thresh)

        # 2. replace spikes with NaN
        s_clean = s.copy()
        s_clean[spikes] = np.nan

        # 3. interpolate over NaNs using time-based or index-based interpolation
        try:
            s_interp = s_clean.interpolate(method="time")
        except ValueError:
            s_interp = s_clean.interpolate(method="index")

        # Fill any remaining NaNs at edges
        s_interp = s_interp.bfill().ffill()

        df[col] = s_interp

    return df.reset_index()


def despikeEZIE(
    input_path,
    output_path,
    cols,
    time_col="tval",
    window=101,
    z_thresh=8.0,
    suffix="_despiked",
):
    """
    Despike EZIE-style CSV file(s) and write new CSVs with a suffix.

    Parameters
    ----------
    input_path : str or Path
        Path to a single CSV file OR a directory containing CSV files.
    output_path : str or Path
        REQUIRED. Directory where cleaned files are written.
    cols : list of str
        Column names to despike (e.g., ["Bx", "By", "Bz", "Ax", "Ay", "Az"]).
    time_col : str
        Name of the time column (default "tval").
    window : int
        Rolling window size for median/MAD (default 101).
    z_thresh : float
        Robust z-score threshold for spikes (default 8.0).
    suffix : str
        Suffix added before file extension for output (default "_despiked").

    Returns
    -------
    list of Path
        Paths of the created output files.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} does not exist")

    if cols is None or len(cols) == 0:
        raise ValueError("Must specify a non-empty list: cols=[...]")

    output_files = []

    # Helper to process a single CSV
    def _process_file(csv_path: Path) -> Path:
        print(f"[INFO] Despiking {csv_path}")
        df = pd.read_csv(csv_path)

        df_clean = _despike_and_interpolate(
            df,
            cols=cols,
            time_col=time_col,
            window=window,
            z_thresh=z_thresh,
        )
        
        # Add horizontal magnetic field magnitude
        if "Bx" in df_clean.columns and "By" in df_clean.columns:
            df_clean["Bh"] = np.sqrt(df_clean["Bx"]**2 + df_clean["By"]**2)
    
        out_path = output_path / f"{csv_path.stem}_despiked.csv"
        df_clean.to_csv(out_path, index=False)
        print(f"[INFO] Saved despiked file to: {out_path}")
        return out_path

    # Single file or all CSVs in directory
    if input_path.is_file():
        output_files.append(_process_file(input_path))
        return output_files
    else:
        csvs = sorted(input_path.glob("*.csv"))
        if not csvs:
            print(f"[WARN] No CSV files found in {input_path}")
            return []
        for f in csvs:
            output_files.append(_process_file(f))

        return output_files