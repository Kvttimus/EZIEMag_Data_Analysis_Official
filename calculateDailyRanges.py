#ctemp range is currently from all 24 hours, not the last 24 hours

import pandas as pd
import numpy as np
import argparse
import sys
from pathlib import Path

def load_with_time(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "timeString" not in df.columns and "time" not in df.columns:
        raise ValueError(f"{path} must contain 'timeString' or 'time'.")

    # If already has parsed time column, use it; else parse timeString
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    else:
        ts = df["timeString"].astype(str).str.strip()
        df["time"] = pd.to_datetime(ts, utc=True, errors="coerce", format="mixed")
        if df["time"].isna().any():
            mask = df["time"].isna()
            df.loc[mask, "time"] = pd.to_datetime(
                ts[mask], utc=True, errors="coerce", format="ISO8601"
            )

    # Last fallback using tval if still needed
    if df["time"].isna().any() and "tval" in df.columns:
        mask = df["time"].isna()
        df.loc[mask, "time"] = pd.to_datetime(
            df.loc[mask, "tval"],
            unit="s",
            origin="unix",
            utc=True,
            errors="coerce",
        )

    df = df.dropna(subset=["time"]).sort_values("time")
    return df

def numeric_range(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna()
    if len(x) < 2:
        return np.nan
    return float(x.max() - x.min())

def day_from_df_or_filename(df: pd.DataFrame, filename_stem: str) -> str:
    # Use first UTC day present in the file
    if len(df) > 0:
        day = df["time"].dt.floor("D").min()
        if pd.notna(day):
            return day.strftime("%Y-%m-%d")
    # fallback: keep filename stem if time is missing
    return filename_stem

def summarize_file(path: Path) -> dict:
    df = load_with_time(str(path))
    day = day_from_df_or_filename(df, path.stem)

    out = {
        "day": day,
        "file": path.name,
        "n_rows": int(len(df)),
    }

    # ctemp range
    out["ctemp_range"] = numeric_range(df["ctemp"]) if "ctemp" in df.columns else np.nan

    # ctemp average
    out["ctemp_avg"] = np.mean(df["ctemp"]) if "ctemp" in df.columns else np.nan

    # Bh range (raw)
    out["Bh_range"] = numeric_range(df["Bh"]) if "Bh" in df.columns else np.nan

    # noise_Bh range (model residual), if available
    out["noise_Bh_range"] = numeric_range(df["noise_Bh"]) if "noise_Bh" in df.columns else np.nan

    # noise_Bh median
    out["noise_Bh_median"] = np.median(df["noise_Bh"]) if "noise_Bh" in df.columns else np.nan

    # Ax range
    out["Ax_range"] = numeric_range(df["Ax"]) if "Ax" in df.columns else np.nan

    # Ay range
    out["Ay_range"] = numeric_range(df["Ay"]) if "Ay" in df.columns else np.nan

    # Az range
    out["Az_range"] = numeric_range(df["Az"]) if "Az" in df.columns else np.nan

    # Gx range
    out["Gx_range"] = numeric_range(df["Gx"]) if "Gx" in df.columns else np.nan

    # Gy range
    out["Gy_range"] = numeric_range(df["Gy"]) if "Gy" in df.columns else np.nan

    # Gz range
    out["Gz_range"] = numeric_range(df["Gz"]) if "Gz" in df.columns else np.nan

    # # noise_Bh std deviation
    # out["noise_Bh_stdDeviation"] = np.std(df["noise_Bh"]) if "noise_Bh" in df.columns else np.nan

    # useful sanity stats
    out["ctemp_min"] = float(pd.to_numeric(df["ctemp"], errors="coerce").min()) if "ctemp" in df.columns else np.nan
    out["ctemp_max"] = float(pd.to_numeric(df["ctemp"], errors="coerce").max()) if "ctemp" in df.columns else np.nan

    out["Bh_min"] = float(pd.to_numeric(df["Bh"], errors="coerce").min()) if "Bh" in df.columns else np.nan
    out["Bh_max"] = float(pd.to_numeric(df["Bh"], errors="coerce").max()) if "Bh" in df.columns else np.nan

    return out

def main():
    ap = argparse.ArgumentParser(description="Summarize daily ranges (1 CSV file = 1 day).")
    ap.add_argument("input_dir", help="Folder containing per-day CSV files")
    ap.add_argument("out_csv", help="Output CSV path (one row per day)")
    ap.add_argument("--glob", default="*.csv", help="File glob pattern (default: *.csv)")
    args = ap.parse_args()

    folder = Path(args.input_dir)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    files = sorted(folder.glob(args.glob))
    if not files:
        raise RuntimeError(f"No files matched {args.glob} in {folder}")

    rows = []
    for f in files:
        try:
            rows.append(summarize_file(f))
        except Exception as e:
            print(f"[WARN] Skipping {f.name}: {e}", file=sys.stderr)

    out_df = pd.DataFrame(rows)

    # Sort by day if it looks like YYYY-MM-DD; otherwise sort by filename
    if "day" in out_df.columns:
        out_df = out_df.sort_values(["day", "file"])

    out_df.to_csv(args.out_csv, index=False)
    print(f"[OK] Wrote {len(out_df)} daily rows to {args.out_csv}")

if __name__ == "__main__":
    main()
