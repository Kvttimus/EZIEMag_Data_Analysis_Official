"""
correlate_any.py

Compute Pearson R between any two columns from:
- EZIE CSV files (with tval)
- FRD SEC files (IAGA-style X/Y/Z/F)
- Any combo: CSV–CSV, CSV–SEC, SEC–SEC.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from io import StringIO


def load_ezie_or_frd(path):
    """
    Load an EZIE .csv or FRD .sec file into a DataFrame with:

        tval      : float seconds since epoch
        tval_sec  : integer seconds since epoch

    CSV:
        - must already have 'tval'.
    SEC:
        - assumes columns: date, time, doy, X, Y, Z, F
    """
    path = Path(path)
    suf = path.suffix.lower()

    if suf == ".csv":
        df = pd.read_csv(path)
        if "tval" not in df.columns:
            raise KeyError(f"{path} has no 'tval' column")
        df["tval"] = pd.to_numeric(df["tval"], errors="coerce")
        df["tval_sec"] = df["tval"].round().astype("Int64")
        return df

    if suf == ".sec":
        text = path.read_text()
        lines = text.splitlines()

        header = []
        data = []
        reading_data = False
        for ln in lines:
            if not reading_data and ln[:4].isdigit():
                reading_data = True
            if reading_data and ln.strip():
                data.append(ln)
            else:
                header.append(ln)

        if not data:
            raise ValueError(f"No data lines in {path}")

        df = pd.read_csv(
            StringIO("\n".join(data)),
            delim_whitespace=True,
            header=None,
            names=["date", "time", "doy", "X", "Y", "Z", "F"],
        )

        dt = pd.to_datetime(df["date"] + " " + df["time"], utc=True, errors="coerce")
        if dt.isna().all():
            raise ValueError(f"Failed to parse date/time in {path}")

        tval_sec = (dt.view("int64") // 1_000_000_000).astype("Int64")
        df["tval_sec"] = tval_sec
        df["tval"] = df["tval_sec"].astype(float)

        return df

    raise ValueError(f"Unsupported file type {suf}, must be .csv or .sec")


def align_two_signals(file1, col1, file2, col2, hours_min=None, hours_max=None):
    """
    Load file1 / file2 (csv or sec), align on tval_sec, optionally restrict
    to a time window (in hours from the start of the overlapping data),
    and return x, y arrays.

    hours_min, hours_max:
        - measured in hours from the earliest overlapping timestamp
        - example: hours_min=0, hours_max=12 -> first 12 hours
        - default (None) = use all data
    """
    df1 = load_ezie_or_frd(file1)
    df2 = load_ezie_or_frd(file2)

    if "tval_sec" not in df1.columns or "tval_sec" not in df2.columns:
        raise KeyError("Missing tval_sec after loading files")

    if col1 not in df1.columns:
        raise KeyError(f"{col1} not in {file1}")
    if col2 not in df2.columns:
        raise KeyError(f"{col2} not in {file2}")

    df1 = df1[["tval_sec", col1]].copy()
    df2 = df2[["tval_sec", col2]].copy()

    df1[col1] = pd.to_numeric(df1[col1], errors="coerce")
    df2[col2] = pd.to_numeric(df2[col2], errors="coerce")

    merged = pd.merge(df1, df2, on="tval_sec", how="inner")
    merged = merged.dropna(subset=[col1, col2])

    if merged.empty:
        raise ValueError("No overlapping timestamps after alignment")

    # ---- Optional time window filtering (in hours from first overlapping point) ----
    if hours_min is not None or hours_max is not None:
        t0 = merged["tval_sec"].min()
        rel_hours = (merged["tval_sec"] - t0) / 3600.0

        mask = pd.Series(True, index=merged.index)
        if hours_min is not None:
            mask &= rel_hours >= hours_min
        if hours_max is not None:
            mask &= rel_hours < hours_max

        merged = merged[mask]
        if merged.empty:
            raise ValueError(
                "No data left after applying hours_min/hours_max filter"
            )

    x = merged[col1].to_numpy()
    y = merged[col2].to_numpy()
    return x, y


def compute_R(file1, col1, file2=None, col2=None, hours_min=None, hours_max=None):
    """
    Compute Pearson R between col1 in file1 and col2 in file2.

    If file2 is None, uses file1 for both (two columns in same file).

    Optional:
        hours_min, hours_max (float, hours from start of overlapping data)
        Example:
            hours_min=0, hours_max=12   -> use first 12 hours
            hours_min=12, hours_max=24  -> use second half of the day
        Default: use all available data.
    """
    if file2 is None:
        file2 = file1
    if col2 is None:
        raise ValueError("You must pass col2")

    x, y = align_two_signals(
        file1, col1, file2, col2,
        hours_min=hours_min,
        hours_max=hours_max,
    )

    if x.size < 2:
        raise ValueError("Not enough points for correlation")

    R = np.corrcoef(x, y)[0, 1]
    return float(R)
