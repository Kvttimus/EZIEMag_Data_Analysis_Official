"""
3 Stacked Plots
python createSurveyPlot.py \
  --combined-survey \
  --ezie decompressed_EZIE_data/EZIE_Oct_12_2024.csv \
  --frd frd_data/frd20241012psec.sec \
  --date 2024-10-12

Original (3 Separate Plots) 
python createSurveyPlot.py \
  --ezie decompressed_EZIE_data/EZIE_Oct_12_2024.csv \
  --frd frd_data/frd20241012psec.sec \
  --date 2024-10-12
"""

import pandas as pd, numpy as np, matplotlib.pyplot as plt
from io import StringIO
import argparse

# ---------------- EZIE LOADER ----------------

def load_ezie_csv(path_or_text):
    # Load file normally or from text blob
    if "\n" in path_or_text or path_or_text.strip().startswith("timeString"):
        df = pd.read_csv(StringIO(path_or_text))
    else:
        df = pd.read_csv(path_or_text)

    # --- Parse time from timeString ---
    if "timeString" not in df.columns:
        raise ValueError("EZIE CSV must contain 'timeString' column.")

    ts = df["timeString"].astype(str).str.strip()

    # mixed ISO8601: with/without fractional seconds
    df["time"] = pd.to_datetime(ts, utc=True, errors="coerce", format="mixed")
    if df["time"].isna().any():
        mask = df["time"].isna()
        df.loc[mask, "time"] = pd.to_datetime(
            ts[mask], utc=True, errors="coerce", format="ISO8601"
        )

    df = df.dropna(subset=["time"])
    df = df.set_index("time").sort_index()

    # If Bh is missing but Bx, By exist, compute Bh
    if "Bh" not in df.columns and {"Bx", "By"}.issubset(df.columns):
        df["Bh"] = np.sqrt(df["Bx"]**2 + df["By"]**2)

    return df

# ---------------- FRD LOADER ----------------

def load_frd_text(path_or_text):
    """
    Load an FRD IAGA-2002 SEC file (original or despiked) and return a DataFrame
    indexed by UTC time.

    Handles:
      - Original: DATE TIME DOY FRDX FRDY FRDZ FRDF
      - Despiked: DATE TIME DOY FRDX FRDY FRDZ FRDF FRDH
    """
    lines = []
    if "\n" in path_or_text:
        src_lines = path_or_text.splitlines()
    else:
        with open(path_or_text, "r", encoding="utf-8", errors="ignore") as f:
            src_lines = f.readlines()

    for ln in src_lines:
        # Data lines start with YYYY-
        if ln[:4].isdigit() and ln[4] == "-":
            lines.append(ln.strip())

    if not lines:
        raise ValueError("No FRD data lines found in input")

    # Allow 7 or 8 columns (FRDH optional)
    df = pd.read_csv(
        StringIO("\n".join(lines)),
        sep=r"\s+",
        engine="python",
        names=["DATE", "TIME", "DOY", "FRDX", "FRDY", "FRDZ", "FRDF", "FRDH"],
        header=None,
        parse_dates=[["DATE", "TIME"]],
    )

    df = df.rename(columns={"DATE_TIME": "time"})
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"])
    df = df.set_index("time").sort_index()

    # Horizontal magnetic field
    if {"FRDX", "FRDY"}.issubset(df.columns):
        df["FRD_Bh"] = np.sqrt(df["FRDX"]**2 + df["FRDY"]**2)

    return df

# ---------------- PLOTTING HELPERS ----------------

def plot_day_series(df, date_utc, column, title_prefix="", resample=None, reducer="median", ax=None):
    day = pd.to_datetime(date_utc).date()
    start = pd.Timestamp(day, tz="UTC")
    end = start + pd.Timedelta(days=1)

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in provided DataFrame.")

    sub = df.loc[(df.index >= start) & (df.index < end), [column]].copy()
    if sub.empty:
        raise ValueError(f"No {column} data on {date_utc}.")

    # --- Auto-resample rule ---
    # If resample is None AND the column is NOT an FRD column,
    # automatically treat it as EZIE → smooth with 1-min median.
    if resample is None:
        if not column.startswith("FRD"):
            # resample = "1min"
            # reducer = "median"   # default
            resample = None

    # --- Apply resampling ---
    if resample:
        # default reducer = median if none provided
        if reducer is None:
            reducer = "median"

        if reducer == "median":
            sub = sub.resample(resample).median()
        else:
            sub = sub.resample(resample).mean()

    sub["hour"] = (sub.index - start).total_seconds() / 3600.0

    if ax is not None:
        ax.plot(sub["hour"], sub[column], linewidth=0.7)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        # units heuristics
        if column.lower().endswith("temp"):
            ylabel = f"{column} (°C)"
        else:
            ylabel = f"{column} (nT)"
        ax.set_ylabel(ylabel, fontsize=18)
        ax.set_title(f"{title_prefix or column} {date_utc.replace('-','')}", fontsize=18)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True, linestyle="--", alpha=0.3)
        return

    # Standalone plot mode
    fig, ax_local = plt.subplots(figsize=(9, 2.6))
    ax_local.plot(sub["hour"], sub[column], linewidth=1.0)
    ax_local.set_xlim(0, 24)
    ax_local.set_xticks(range(0, 25, 3))
    ax_local.set_xlabel("UT hour (0–24)")
    if column.lower().endswith("temp"):
        ylabel = f"{column} (°C)"
    else:
        ylabel = f"{column} (nT)"
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(f"{title_prefix or column} {date_utc.replace('-','')}", fontsize=18)
    fig.tight_layout()
    fig.savefig(f"{column}_{date_utc.replace('-','')}.png", dpi=160, bbox_inches="tight")
    # plt.show()

# ---------------- ARGPARSE + MAIN ----------------

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--combined-survey",
    action="store_true",
    help="Render Bh, ctemp, and FRD_Bh together as stacked subplots (one image)."
)
_parser.add_argument(
    "--ezie",
    type=str,
    default="decompressed_EZIE_data/EZIE_Oct_7_2024.csv",
    help="Path to EZIE CSV file."
)
_parser.add_argument(
    "--frd",
    type=str,
    default="frd_data/frd20241007psec.sec",
    help="Path to FRD SEC file."
)
_parser.add_argument(
    "--date",
    type=str,
    default="2024-10-07",
    help="UTC date (YYYY-MM-DD) to plot."
)
_parser.add_argument(
    "--columns", "-c",
    nargs="+",
    help=(
        "Custom columns to plot as stacked subplots (raw names).\n"
        "Example: --columns Bh ctemp Ay FRD_Bh noise_Bh"
    ),
)

_args, _unknown = _parser.parse_known_args()

ezie = load_ezie_csv(_args.ezie)
frd  = load_frd_text(_args.frd)
_date_for_plots = _args.date

# ---------------------------------------------------------
# MODE 1: Custom columns (new, most flexible)
# ---------------------------------------------------------
if _args.columns:
    cols = _args.columns
    if len(cols) == 0:
        raise ValueError("At least one column name must be provided with --columns/-c.")

    fig, axes = plt.subplots(len(cols), 1, sharex=True,
                             figsize=(12, 3 * len(cols)),
                             constrained_layout=True)

    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        # First try EZIE, then FRD
        if col in ezie.columns:
            df_src = ezie
        elif col in frd.columns:
            df_src = frd
        else:
            raise ValueError(f"Column '{col}' not found in EZIE or FRD data.")

        plot_day_series(df_src, _date_for_plots, col, title_prefix=col, ax=ax)

    axes[-1].set_xlabel("UT hour (0–24)", fontsize=18)
    out_name = f"survey_multi_{_date_for_plots.replace('-','')}.png"
    # plt.suptitle("FRDH vs EZIE_Bh_original vs EZIE_Bh_predicted vs noise_Bh vs ctemp", fontsize=22)
    plt.savefig(out_name, dpi=160, bbox_inches="tight")
    # plt.show()

# ---------------------------------------------------------
# MODE 2: Old combined-survey (3 specific panels)
# ---------------------------------------------------------
elif _args.combined_survey:
    # Ensure needed columns exist
    for need in ["Bh", "ctemp"]:
        if need not in ezie.columns:
            raise ValueError(f"EZIE column '{need}' not found.")
    if "FRD_Bh" not in frd.columns:
        raise ValueError("FRD column 'FRD_Bh' not found.")

    fig, axes = plt.subplots(3, 1, sharex=True,
                             figsize=(12, 8),
                             constrained_layout=True)

    # 1) Bh from EZIE
    plot_day_series(ezie, _date_for_plots, "Bh", "Bh", resample="1min", reducer="median", ax=axes[0])
    # 2) ctemp from EZIE
    plot_day_series(ezie, _date_for_plots, "ctemp", "ctemp", ax=axes[1])
    # 3) FRD_Bh from FRD
    plot_day_series(frd,  _date_for_plots, "FRD_Bh", "FRD_Bh", ax=axes[2])

    axes[-1].set_xlabel("UT hour (0–24)")
    # fig.suptitle("", fontsize=22)

    out_name = f"survey_{_date_for_plots.replace('-','')}.png"
    fig.savefig(out_name, dpi=160, bbox_inches="tight")
    # plt.show()

# ---------------------------------------------------------
# MODE 3: Original separate plots (no --combined-survey, no --columns)
# ---------------------------------------------------------
else:
    plot_day_series(ezie, _date_for_plots, "Bh", "Bh", resample="1min", reducer="median")
    plot_day_series(ezie, _date_for_plots, "ctemp", "ctemp")
    plot_day_series(frd,  _date_for_plots, "FRD_Bh", "FRD_Bh")
