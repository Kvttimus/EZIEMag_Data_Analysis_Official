"""
applyCalibration.py

PURPOSE: Apply a previously saved calibration (A, b) to an EZIE file to produce calibrated EZIE columsn in FRD's NED format (X, Y, Z, F)
"""

"""
Usage:
  python applyCalibration.py \
    --in  og_data/eziemag_decompressed.1s/eziemag.2024101000.AAAAAFPTJDYA.1s.selected.csv \
    --cal og_data/frd_decompressed.1s/frd20241010psec.calibration.json \
    --out og_data/eziemag_decompressed.1s/eziemag.2024101000.calibrated_XYZF.csv \
    [--filter-win 5]

'--filter-win 5' means apply a 5-second rolling mean to the calibrated output after applying the transformation.
    - so for each second, it would take the mean of +- 2 seconds and set it as that value
"""

import argparse, json, numpy as np, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp",  required=True, help="EZIE selected.csv (Bx,By,Bz)")
    ap.add_argument("--cal", dest="cal",  required=True, help="Calibration JSON with keys A (3x3), b (3,)")
    ap.add_argument("--out", dest="outp", required=True, help="Output calibrated XYZF CSV")
    ap.add_argument("--filter-win", type=int, default=0,
                    help="Optional rolling-mean window (seconds) AFTER calibration")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    if "timeString" in df.columns:
        tcol = "timeString"
    elif "tval" in df.columns:
        df["timeString"] = pd.to_datetime(df["tval"], unit="s", utc=True).astype(str)
        tcol = "timeString"
    else:
        raise SystemExit("Need timeString or tval in input CSV.")

    for c in ["Bx","By","Bz"]:
        if c not in df.columns:
            raise SystemExit(f"Missing {c} in input.")

    # Load calibration from JSON
    with open(args.cal, "r") as f:
        cal = json.load(f)
    A = np.array(cal["A"], dtype=float)  # 3x3
    b = np.array(cal["b"], dtype=float)  # 3,

    # Apply calibration
    E = df[["Bx","By","Bz"]].to_numpy(float)   # Nx3
    I = (E @ A.T) + b                          # Nx3 -> X,Y,Z
    X, Y, Z = I[:,0], I[:,1], I[:,2]
    F = np.sqrt(X**2 + Y**2 + Z**2)

    out = pd.DataFrame({
        "timeString": df[tcol],
        "X": X, "Y": Y, "Z": Z, "F": F
    })

    # Optional smoothing AFTER calibration
    if args.filter_win and args.filter_win > 1:
        for c in ["X","Y","Z","F"]:
            out[c] = out[c].rolling(args.filter_win, center=True, min_periods=1).mean()

    out.to_csv(args.outp, index=False)
    print(f"Wrote {len(out)} rows -> {args.outp}")

if __name__ == "__main__":
    main()
