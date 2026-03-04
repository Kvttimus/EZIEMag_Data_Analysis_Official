# Removes the 999,999 auto fill missing value from FRD data
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd


def _read_frd_sec(path: Path):
    """
    Read an IAGA-2002 .sec file.

    Returns
    -------
    header_lines : list[str]
        All non-data lines (metadata & comments).
    df : pandas.DataFrame
        Columns: date, time, doy, X, Y, Z, F
    """
    text = path.read_text()
    lines = text.splitlines()

    header_lines = []
    data_lines = []

    data_started = False
    for ln in lines:
        # Data lines start with a year like "2024-..."
        if not data_started and ln and ln[0].isdigit():
            data_started = True

        if data_started:
            if ln.strip():  # non-empty
                data_lines.append(ln)
        else:
            header_lines.append(ln)

    if not data_lines:
        raise ValueError(f"No data lines found in {path}")

    # Update the column header line to include FRDH
    for i, hl in enumerate(header_lines):
        if hl.strip().startswith("DATE") and "FRDH" not in hl:
            header_lines[i] = hl.replace("FRDF", "FRDF      FRDH")
            break

    # whitespace-delimited: date time doy X Y Z F
    df = pd.read_csv(
        StringIO("\n".join(data_lines)),
        delim_whitespace=True,
        header=None,
        names=["date", "time", "doy", "X", "Y", "Z", "F"],
    )

    return header_lines, df


def _write_frd_sec(path: Path, header_lines, df: pd.DataFrame):
    """
    Write header + data back to an IAGA-style .sec file.
    """
    # Build data lines
    data_lines = []
    for row in df.itertuples(index=False):
        # keep date/time strings exactly, format the numeric fields
        frdh = getattr(row, "FRDH", np.nan)
        line = (
            f"{row.date} {row.time} "
            f"{int(row.doy):3d} "
            f"{row.X:11.2f} "
            f"{row.Y:11.2f} "
            f"{row.Z:11.2f} "
            f"{row.F:11.2f} "
            f"{frdh:11.2f}"
        )
        data_lines.append(line)

    text = "\n".join(header_lines + data_lines) + "\n"
    path.write_text(text)


def _despike_frd_dataframe(
    df: pd.DataFrame,
    cols=("X", "Y", "Z", "F"),
    spike_min_value=90000.0,
):
    """
    Replace 'spike' values in FRD data (e.g. 99999) by interpolation.

    Any value >= spike_min_value in the selected columns is treated as a spike.
    """
    df = df.copy()

    # Use integer index since the data are 1-second cadence
    df = df.reset_index(drop=True)

    for col in cols:
        if col not in df.columns:
            print(f"[WARN] Column '{col}' not found, skipping.")
            continue

        s = df[col].astype(float)

        # Mask spike sentinels, e.g. 99999.00
        spike_mask = s >= spike_min_value

        if spike_mask.any():
            print(f"[INFO] Column {col}: {spike_mask.sum()} spike values found")

        s_clean = s.copy()
        s_clean[spike_mask] = np.nan

        # Linear interpolation across spikes
        s_interp = s_clean.interpolate(limit_direction="both")
        df[col] = s_interp

    # Add horizontal magnitude column (after despiking)
    if "X" in df.columns and "Y" in df.columns:
        df["FRDH"] = np.sqrt(df["X"].astype(float)**2 + df["Y"].astype(float)**2)

    return df


def despikeFRD(
    input_path,
    output_dir,
    spike_min_value=90000.0,
):
    """
    Despike FRD .sec file(s) by replacing 99999-style values via interpolation.

    Parameters
    ----------
    input_path : str or Path
        A single .sec file OR a directory containing .sec files.
    output_dir : str or Path
        Directory where despiked files will be written.
        Created if it does not exist.
    spike_min_value : float
        Any value >= this in X/Y/Z/F is treated as a spike (default 90000).

    Returns
    -------
    list[Path]
        Paths of the newly written despiked files.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} does not exist")

    out_paths = []

    def _process_one(sec_path: Path) -> Path:
        print(f"[INFO] Processing {sec_path.name}")
        header_lines, df = _read_frd_sec(sec_path)
        df_clean = _despike_frd_dataframe(df, spike_min_value=spike_min_value)

        out_name = f"{sec_path.stem}_despiked{sec_path.suffix}"
        out_path = output_dir / out_name

        _write_frd_sec(out_path, header_lines, df_clean)
        print(f"[INFO] Saved despiked file to {out_path}")
        return out_path

    if input_path.is_file():
        # Single file
        out_paths.append(_process_one(input_path))
    elif input_path.is_dir():
        # All .sec files in directory
        sec_files = sorted(input_path.glob("*.sec"))
        if not sec_files:
            print(f"[WARN] No .sec files found in {input_path}")
            return []
        for sec in sec_files:
            out_paths.append(_process_one(sec))
    else:
        raise FileNotFoundError(f"{input_path} is not a file or directory")

    return out_paths
