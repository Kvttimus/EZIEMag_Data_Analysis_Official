"""
EZIE Magnetometer RAW/RAWZ → CSV parser & folder merger

- Accepts a single `.raw`/`.rawz` file OR a folder path
- Auto-decompresses gzip/zlib `.rawz`
- Parses fixed 160-byte records (NASA EZIE schema offsets below)
- Writes either one CSV per file (file input) OR one merged CSV (folder input)

Examples:
  # single file → CSV
  python processRawzToCSV.py path/to/eziemag.2024101000...1s.rawz

  # folder → merged CSV (non-recursive)
  python processRawzToCSV.py path/to/folder

  # folder → merged CSV (recursive) with custom outdir/name
  python processRawzToCSV.py path/to/folder --recursive \
      --outdir og_data/eziemag_decompressed.1s \
      --outfile my_merged.selected.csv


$ python processRawzToCSV.py og_data/EZIE_Oct_7_to_13_2024/home/ezie/raw.1s/20241008 --recursive --outdir decompressed_EZIE_data --outfile EZIE_Oct_8_2024.csv
"""

import os
import sys
import struct
import gzip
import zlib
import argparse
from datetime import datetime, timezone

try:
    import pandas as pd
except ImportError:
    print("This script requires pandas. Install with: pip install pandas", file=sys.stderr)
    sys.exit(1)

REC_SIZE = 160  # bytes per record (fixed)

COLS_IN_CSV = [
    "timeString", "tval",   # time
    "latitude", "longitude", "altitude",
    "tres", "ctemp", "ccr",    # temps/meta
    "Bx", "By", "Bz",          # magnetic field
    "Ax", "Ay", "Az",          # accelerometer
    "Gx", "Gy", "Gz",          # gyroscope
    "imu_ctemp",               # IMU temp
]

# ---------- IO helpers ----------
    
def _read_all_bytes(path: str) -> bytes | None:
    try:
        with open(path, "rb") as f:
            return f.read()
    except FileNotFoundError:
        print(f"[warn] {path} file not found", file=sys.stderr)
        return None


def _maybe_decompress(raw_bytes: bytes) -> bytes:
    # gzip by magic 1F 8B
    if len(raw_bytes) >= 2 and raw_bytes[:2] == b"\x1f\x8b":
        return gzip.decompress(raw_bytes)
    # zlib (common headers 78 9C / 78 DA)
    if len(raw_bytes) >= 2 and raw_bytes[:2] in (b"\x78\x9c", b"\x78\xda"):
        return zlib.decompress(raw_bytes)
    # already decompressed .raw
    return raw_bytes

# ---------- parsing ----------

def _parse_record(rec: bytes):
    """Parse one 160-byte record into a dict. Assumes little-endian floats/doubles."""
    if len(rec) != REC_SIZE:
        return None

    # Core time (Unix seconds, float64)
    tval = struct.unpack_from("<d", rec, 0)[0]  # 0..7

    # GPS & environment
    lat  = struct.unpack_from("<f", rec, 36)[0]   # deg
    lon  = struct.unpack_from("<f", rec, 40)[0]   # deg
    alt  = struct.unpack_from("<f", rec, 44)[0]   # meters
    tres = struct.unpack_from("<f", rec, 56)[0]   # °C step
    ctemp = struct.unpack_from("<f", rec, 60)[0]  # °C
    ccr_f = struct.unpack_from("<f", rec, 68)[0]  # RM3100 cycles (float in stream)

    # Magnetometer (nT)
    bx = struct.unpack_from("<f", rec, 72)[0]
    by = struct.unpack_from("<f", rec, 76)[0]
    bz = struct.unpack_from("<f", rec, 80)[0]

    # IMU: accel (m/s^2)
    ax = struct.unpack_from("<f", rec, 120)[0]
    ay = struct.unpack_from("<f", rec, 124)[0]
    az = struct.unpack_from("<f", rec, 128)[0]

    # IMU: gyro (deg/s)
    gx = struct.unpack_from("<f", rec, 132)[0]
    gy = struct.unpack_from("<f", rec, 136)[0]
    gz = struct.unpack_from("<f", rec, 140)[0]

    # IMU internal temp (°C)
    imu_ctemp = struct.unpack_from("<f", rec, 152)[0]

    # ISO8601 UTC
    try:
        time_iso = datetime.fromtimestamp(tval, tz=timezone.utc).isoformat()
    except Exception:
        time_iso = ""

    # round cycle count if finite
    ccr = int(round(ccr_f)) if ccr_f == ccr_f else None  # NaN check

    return dict(
        timeString=time_iso,
        tval=tval,
        latitude=lat, longitude=lon, altitude=alt,
        tres=tres, ctemp=ctemp, ccr=ccr,
        Bx=bx, By=by, Bz=bz,
        Ax=ax, Ay=ay, Az=az,
        Gx=gx, Gy=gy, Gz=gz,
        imu_ctemp=imu_ctemp,
    )

def parse_bytes_to_dataframe(raw_bytes: bytes) -> pd.DataFrame:
    n = len(raw_bytes) // REC_SIZE
    rows = []
    if len(raw_bytes) % REC_SIZE != 0:
        # ignore trailing partial record, but warn
        print(f"[warn] byte length {len(raw_bytes)} not multiple of {REC_SIZE}; "
              f"ignoring last {len(raw_bytes) % REC_SIZE} bytes", file=sys.stderr)
    for i in range(n):
        rec = raw_bytes[i*REC_SIZE:(i+1)*REC_SIZE]
        row = _parse_record(rec)
        if row is not None:
            rows.append(row)
    return pd.DataFrame(rows)

def write_selected_columns(df: pd.DataFrame, out_path: str, selected_cols=None):
    cols = selected_cols or COLS_IN_CSV
    cols_available = [c for c in cols if c in df.columns]
    df[cols_available].to_csv(out_path, index=False)

# ---------- batch logic (folder) ----------

def list_raw_files(root: str, recursive: bool = False):
    exts = {".raw", ".rawz"}
    out = []
    if recursive:
        for d, _, files in os.walk(root):
            for f in files:
                if os.path.splitext(f)[1].lower() in exts:
                    out.append(os.path.join(d, f))
    else:
        for f in os.listdir(root):
            p = os.path.join(root, f)
            if os.path.isfile(p) and os.path.splitext(f)[1].lower() in exts:
                out.append(p)
    # deterministic order
    return sorted(out)

def process_file_to_df(path: str) -> pd.DataFrame:
    try:
        raw_bytes = _read_all_bytes(path)
        if raw_bytes is None:
            return pd.DataFrame(columns=COLS_IN_CSV)

        data = _maybe_decompress(raw_bytes)
        df = parse_bytes_to_dataframe(data)

        if df.empty:
            print(f"[warn] produced 0 rows for {path}", file=sys.stderr)

        df["_source_file"] = os.path.basename(path)
        return df

    except FileNotFoundError:
        print(f"[warn] {path} file not found", file=sys.stderr)
        return pd.DataFrame(columns=COLS_IN_CSV)

    except Exception as e:
        print(f"[warn] failed processing {path}: {e}", file=sys.stderr)
        return pd.DataFrame(columns=COLS_IN_CSV)

def merge_folder(root: str, recursive: bool, dedupe: bool) -> pd.DataFrame:
    files = list_raw_files(root, recursive=recursive)
    if not files:
        print(f"No .raw/.rawz files found in: {root}", file=sys.stderr)
        return pd.DataFrame(columns=COLS_IN_CSV)

    frames = []
    for i, p in enumerate(files, 1):
        print(f"[{i}/{len(files)}] parsing {p} ...")
        # frames.append(process_file_to_df(p))
        df = process_file_to_df(p)
        if df is None:
            print(f"[warn] got None for {p}; skipping", file=sys.stderr)
            continue
        frames.append(df)


    merged = pd.concat(frames, ignore_index=True)
    # sort by tval first (float), then timeString for tie-breakers
    if "tval" in merged.columns:
        merged = merged.sort_values(["tval", "timeString"], kind="stable")
    elif "timeString" in merged.columns:
        merged = merged.sort_values("timeString", kind="stable")

    if dedupe and "tval" in merged.columns:
        before = len(merged)
        merged = merged.drop_duplicates(subset=["tval"], keep="first")
        after = len(merged)
        if after < before:
            print(f"[info] deduped {before - after} rows on tval", file=sys.stderr)

    merged = merged.reset_index(drop=True)
    return merged

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="EZIE RAW/RAWZ to CSV (single file or folder merge)")
    ap.add_argument("input", help="Path to .raw/.rawz file OR a folder containing them")
    ap.add_argument("--outdir", default=os.path.join("og_data", "eziemag_decompressed.1s"),
                    help="Output directory (default: og_data/eziemag_decompressed.1s)")
    ap.add_argument("--outfile", default=None,
                    help="Output CSV filename when merging a folder (default: <foldername>_merged.selected.csv)")
    ap.add_argument("--recursive", action="store_true",
                    help="Recurse into subfolders when input is a folder")
    ap.add_argument("--no-dedupe", action="store_true",
                    help="Do not drop duplicate tval rows in merged output")
    args = ap.parse_args()

    in_path = args.input
    out_dir = args.outdir if args.outdir else "."
    os.makedirs(out_dir, exist_ok=True)

    # File vs Folder
    if os.path.isdir(in_path):
        merged_df = merge_folder(in_path, recursive=args.recursive, dedupe=not args.no_dedupe)
        if merged_df.empty:
            print("No rows parsed; nothing written.", file=sys.stderr)
            sys.exit(2)

        # choose output name
        if args.outfile:
            out_csv = os.path.join(out_dir, args.outfile)
        else:
            folder_base = os.path.basename(os.path.abspath(in_path)) or "eziemag"
            out_csv = os.path.join(out_dir, f"{folder_base}_merged.selected.csv")

        cols_available = [c for c in COLS_IN_CSV if c in merged_df.columns]
        merged_df[cols_available].to_csv(out_csv, index=False)
        print(f"[OK] merged CSV → {out_csv}")
        print(f"[info] rows: {len(merged_df):,}")
        return

    # Single file path
    if not os.path.isfile(in_path):
        print(f"Input not found: {in_path}", file=sys.stderr)
        # sys.exit(1)
        return

    base = os.path.splitext(os.path.basename(in_path))[0]
    out_csv = os.path.join(out_dir, base + ".selected.csv")

    print(f"[1/1] parsing {in_path} ...")
    df = process_file_to_df(in_path)
    if df.empty:
        print(f"[warn] 0 rows parsed from {in_path}", file=sys.stderr)

    cols_available = [c for c in COLS_IN_CSV if c in df.columns]
    df[cols_available].to_csv(out_csv, index=False)
    print(f"[OK] CSV → {out_csv}")
    print(f"[info] rows: {len(df):,}")

if __name__ == "__main__":
    main()
