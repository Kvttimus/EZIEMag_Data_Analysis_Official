import pandas as pd
import numpy as np
import argparse
from io import StringIO
import os

# ---------------- EZIE LOADER ----------------
def load_ezie_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if "timeString" not in df.columns:
        raise ValueError("EZIE CSV must contain 'timeString'.")

    ts = df["timeString"].astype(str).str.strip()
    df["time"] = pd.to_datetime(ts, utc=True, errors="coerce", format="mixed")

    if df["time"].isna().any():
        mask = df["time"].isna()
        df.loc[mask, "time"] = pd.to_datetime(
            ts[mask], utc=True, errors="coerce", format="ISO8601"
        )

    df = df.dropna(subset=["time"]).sort_values("time")

    # Compute EZIE_Bh
    if "Bh" in df.columns:
        df["EZIE_Bh"] = pd.to_numeric(df["Bh"], errors="coerce")
    elif {"Bx", "By"}.issubset(df.columns):
        bx = pd.to_numeric(df["Bx"], errors="coerce")
        by = pd.to_numeric(df["By"], errors="coerce")
        df["EZIE_Bh"] = np.sqrt(bx**2 + by**2)
    else:
        raise ValueError("EZIE must contain 'Bh' or both 'Bx' and 'By'.")

    return df

# ---------------- FRD LOADER ----------------
def load_frd_sec(path: str) -> pd.DataFrame:
    lines = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if ln[:4].isdigit() and ln[4] == "-":
                lines.append(ln.strip())

    if not lines:
        raise ValueError("No FRD data lines found.")

    df = pd.read_csv(
        StringIO("\n".join(lines)),
        sep=r"\s+",
        engine="python",
        names=["DATE", "TIME", "DOY", "FRDX", "FRDY", "FRDZ", "FRDF", "FRDH"],
        header=None,
    )

    df["time"] = pd.to_datetime(
        df["DATE"].astype(str) + " " + df["TIME"].astype(str),
        utc=True,
        errors="coerce",
    )

    df = df.dropna(subset=["time"]).sort_values("time")

    # Prefer FRDH if present; else compute it
    if "FRDH" in df.columns and df["FRDH"].notna().any():
        df["FRD_Bh"] = pd.to_numeric(df["FRDH"], errors="coerce")
    else:
        frdx = pd.to_numeric(df["FRDX"], errors="coerce")
        frdy = pd.to_numeric(df["FRDY"], errors="coerce")
        df["FRD_Bh"] = np.sqrt(frdx**2 + frdy**2)

    return df[["time", "FRD_Bh"]]

# ---------------- Main Stuff ----------------
def compute_actual_noise(ezie_path: str, frd_path: str, out_path: str, time_tol_sec: float) -> None:
    ezie = load_ezie_csv(ezie_path)
    frd = load_frd_sec(frd_path)

    merged = pd.merge_asof(
        ezie.sort_values("time"),
        frd.sort_values("time"),
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=time_tol_sec),
    )

    merged["actual_noise_Bh"] = merged["FRD_Bh"] - merged["EZIE_Bh"]

    out_cols = ["timeString", "time", "EZIE_Bh", "FRD_Bh", "actual_noise_Bh"]
    if "ctemp" in merged.columns:
        out_cols.append("ctemp")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    merged[out_cols].to_csv(out_path, index=False)

    print(f"[OK] {os.path.basename(out_path)}")

def main():
    parser = argparse.ArgumentParser(description="Compute actual noise: FRD_Bh - EZIE_Bh")
    parser.add_argument("--ezie", required=True, help="Path to EZIE CSV")
    parser.add_argument("--frd", required=True, help="Path to FRD .sec file")
    parser.add_argument("--out", required=True, help="Path to output CSV")
    parser.add_argument("--time-tol", type=float, default=0.5, help="merge tolerance in seconds (default 0.5)")
    args = parser.parse_args()

    compute_actual_noise(args.ezie, args.frd, args.out, args.time_tol)

if __name__ == "__main__":
    main()