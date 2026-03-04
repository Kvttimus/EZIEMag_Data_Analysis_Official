"""
PURPOSE: Decide if 1 static affine transform (A, b) can map EZIE [Bx, By, Bz] to FRD [X, Y, Z]
"""

"""
Usage:
   python checkStaticRotation.py \
       --sec og_data/frd_decompressed.1s/frd20241010psec.sec \
       --ezie og_data/eziemag_decompressed.1s/eziemag.2024101000.AAAAAFPTJDYA.1s.selected.csv \
       --win 60 
'--win 60' --> uses 60 second windows when checking for stability
"""

"""
HOW TO READ OUTPUT
- Global RMS residual: after fitting one (A,b), how far EZIE→FRD is on average
    - Tens of nT is great; a few hundred can still be okay depending on environment
- Window RMS residuals (fixed A,b): if these are flat over time (and small), data is consistent with one static R,b
    - If they jump around (and the IMU shows motion), a single R,b is not valid for the whole period
- IMU stability:
    - Gyro RMS ~ 0 → likely no rotation.
    - Accel direction std ~ 0-1° → gravity direction stable → orientation static.
"""

import argparse, re
import numpy as np
import pandas as pd
import json, os

def read_iaga_sec(path):
    # Find where data starts
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("#DATA"):
            start = i + 1; break
        if re.match(r"\s*DATE\s+TIME\s+DOY", line):
            start = i + 1; break
    if start is None:
        start = next((i for i,l in enumerate(lines) if not l.strip().startswith('#')), 0)

    # Use regex sep (pandas deprecates delim_whitespace)
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None, skiprows=start, engine="python")

    if df.shape[1] >= 7:
        df = df.iloc[:, :7]
        df.columns = ["DATE","TIME","DOY","X","Y","Z","F"]
    elif df.shape[1] == 6:
        df.columns = ["DATE","TIME","DOY","X","Y","Z"]
        df["F"] = np.sqrt(df["X"]**2 + df["Y"]**2 + df["Z"]**2)
    else:
        raise ValueError(f"Unexpected .sec column count: {df.shape[1]}")

    df["time"] = pd.to_datetime(df["DATE"] + " " + df["TIME"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    # Optional: denoise a bit (helps conditioning)
    for c in ["X","Y","Z"]:
        df[c] = df[c].rolling(5, center=True, min_periods=1).mean()

    return df[["time","X","Y","Z","F"]]

def read_ezie_csv(path):
    df = pd.read_csv(path)
    # Build time
    if "timeString" in df.columns:
        df["time"] = pd.to_datetime(df["timeString"], utc=True, errors="coerce")
    elif "tval" in df.columns:
        df["time"] = pd.to_datetime(df["tval"], unit="s", utc=True, errors="coerce")
    else:
        raise ValueError("EZIE CSV needs timeString or tval.")
    for c in ["Bx","By","Bz"]:
        if c not in df.columns:
            raise ValueError(f"Missing {c} in EZIE CSV")

    # Keep IMU if present (optional stability checks)
    has_imu = all(c in df.columns for c in ["Gx","Gy","Gz","Ax","Ay","Az"])

    # Denoise magnetic channels
    for c in ["Bx","By","Bz"]:
        df[c] = pd.Series(df[c]).rolling(5, center=True, min_periods=1).mean()

    cols = ["time","Bx","By","Bz"]
    if has_imu:
        cols += ["Gx","Gy","Gz","Ax","Ay","Az"]
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return df[cols]

def align_by_second(ezie, frd):
    e = ezie.copy(); f = frd.copy()
    e["t_sec"] = e["time"].dt.round("s")
    f["t_sec"] = f["time"].dt.floor("s")
    m = pd.merge(f, e, on="t_sec", how="inner", suffixes=("_frd","_ezie")).sort_values("t_sec")
    m = m.rename(columns={"t_sec":"time"})
    return m

def fit_affine(E, I):
    """
    Solve I ≈ A E + b  (least squares).
    E: Nx3 (Bx,By,Bz), I: Nx3 (X,Y,Z)
    Returns A (3x3), b (3,)
    """
    N = E.shape[0]
    M = np.hstack([E, np.ones((N,1))])   # [Bx By Bz 1]
    X, *_ = np.linalg.lstsq(M, I, rcond=None)  # 4x3
    A = X[:3,:].T
    b = X[3,:]
    return A, b

def window_rms_residuals(E, I, A, b, win):
    """
    Apply fixed (A,b) and compute residual RMS per window.
    Returns list of (t0, rmsX, rmsY, rmsZ, rmsVec)
    """
    N = E.shape[0]
    out = []
    for start in range(0, N - win + 1, win):
        Ei = E[start:start+win]
        Ii = I[start:start+win]
        Ihat = (Ei @ A.T) + b
        R = Ii - Ihat
        rms = np.sqrt((R**2).mean(axis=0))
        rms_vec = np.sqrt((np.linalg.norm(R, axis=1)**2).mean())
        out.append((start, rms[0], rms[1], rms[2], rms_vec))
    return out

def imu_stability(df):
    """Return simple IMU stability metrics if IMU present: gyro RMS (deg/s) and accel direction std (deg)."""
    if not all(c in df.columns for c in ["Gx","Gy","Gz","Ax","Ay","Az"]):
        return None
    gyro = df[["Gx","Gy","Gz"]].to_numpy(float)
    acc  = df[["Ax","Ay","Az"]].to_numpy(float)
    gyro_rms = np.sqrt((gyro**2).mean(axis=0))
    # Accel direction stability: angle of each sample to the mean gravity vector
    g_mean = acc.mean(axis=0)
    g_mean /= np.linalg.norm(g_mean) + 1e-12
    dots = (acc @ g_mean) / (np.linalg.norm(acc, axis=1) + 1e-12)
    dots = np.clip(dots, -1, 1)
    ang = np.degrees(np.arccos(dots))
    return dict(gyro_rms=gyro_rms, accel_dir_std=np.std(ang), accel_dir_max=np.max(ang))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sec", required=True, help="Path to FRD .sec (IAGA-2002 1s)")
    ap.add_argument("--ezie", required=True, help="Path to EZIE CSV (with Bx,By,Bz and optionally IMU)")
    ap.add_argument("--win", type=int, default=60, help="Window length in seconds for stability check")
    args = ap.parse_args()

    frd  = read_iaga_sec(args.sec)
    ezie = read_ezie_csv(args.ezie)

    merged = align_by_second(ezie, frd)
    if merged.empty:
        raise SystemExit("No overlap after second alignment.")

    # Build matrices
    I = merged[["X","Y","Z"]].to_numpy(float)          # FRD
    E = merged[["Bx","By","Bz"]].to_numpy(float)       # EZIE

    # Fit a single affine mapping on the WHOLE overlap
    A, b = fit_affine(E, I)
    Ihat = (E @ A.T) + b
    R = I - Ihat
    rms_global = np.sqrt((R**2).mean(axis=0))
    rms_vec_global = np.sqrt((np.linalg.norm(R, axis=1)**2).mean())

    print("=== Global fixed calibration (A,b) ===")
    print("A ="); 
    with np.printoptions(precision=6, suppress=True):
        print(A)
    print("b =", np.array2string(b, precision=2, suppress_small=True))
    print("Global RMS residual [nT]: X={:.1f} Y={:.1f} Z={:.1f} | vector={:.1f}".format(
        rms_global[0], rms_global[1], rms_global[2], rms_vec_global
    ))

    # Save EZIE Mag A & b calibration matrices next to the .sec file 
    out_cal = os.path.splitext(os.path.basename(args.sec))[0] + ".calibration.json"
    cal_path = os.path.join(os.path.dirname(args.sec), out_cal)
    cal = {"A": A.tolist(), "b": b.tolist()}
    with open(cal_path, "w") as f:
        json.dump(cal, f, indent=2)
    print(f"\nSaved calibration -> {cal_path}")

    # Per-window residuals using the *same* A,b (this is the rotation/bias stability test)
    win = max(30, int(args.win))
    per = window_rms_residuals(E, I, A, b, win)
    vec_rms = [p[4] for p in per]
    print("\n=== Window RMS residuals with fixed (A,b) ===")
    for k, (start, rx, ry, rz, rv) in enumerate(per):
        t0 = merged["time"].iloc[start]
        print(f"Window {k:02d} @ {t0}: RMS_vec={rv:6.1f} nT  (X={rx:6.1f}, Y={ry:6.1f}, Z={rz:6.1f})")

    if vec_rms:
        print("\nVector RMS (fixed A,b): min={:.1f}  median={:.1f}  max={:.1f} nT".format(
            float(np.min(vec_rms)), float(np.median(vec_rms)), float(np.max(vec_rms))
        ))

    # IMU stability (if available)
    imu = imu_stability(merged.rename(columns={"Gx":"Gx","Gy":"Gy","Gz":"Gz","Ax":"Ax","Ay":"Ay","Az":"Az"}))
    if imu:
        gx, gy, gz = imu["gyro_rms"]
        print("\n=== IMU stability ===")
        print("Gyro RMS [deg/s]: Gx={:.3f}, Gy={:.3f}, Gz={:.3f}".format(gx, gy, gz))
        print("Accel direction std [deg]: {:.3f}   max [deg]: {:.3f}".format(imu["accel_dir_std"], imu["accel_dir_max"]))

    # Simple decision rule
    # (adjust these based on needed tolerance and site noise)
    ok_vec_rms = (np.median(vec_rms) < 100.0) if vec_rms else False
    ok_spread  = ((np.max(vec_rms) - np.min(vec_rms)) < 50.0) if vec_rms else False

    if ok_vec_rms and ok_spread:
        print("\n=> Looks consistent with a single static orientation (R) and bias (b).")
    else:
        print("\n=> Residuals vary notably over time. Consider time-varying orientation/bias (use IMU) or revisit denoising/overlap.")
    
if __name__ == "__main__":
    main()
