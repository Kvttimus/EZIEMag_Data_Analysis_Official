import os
import glob
import pandas as pd

# ===== USER SETTINGS =====
DIR1 = "denoised4_EZIE_data"      # 5-min data (Bh, denoised_Bh, etc.)
DIR2 = "noise_EZIE_data"          # 1-sec data (Bh_pred or noise_Bh_pred)
GLOB1 = "*_denoised.csv"
GLOB2 = "*_noise.csv"

TIME_COL = "timeString"
PRED_COL = "Bh_pred"              # column in DIR2 to aggregate & append
OUT_DIR = "merged_denoised4_EZIE_data"
# ==========================

def base_key(path, suffix):
    name = os.path.basename(path)
    if name.endswith(suffix):
        return name[:-len(suffix)]
    return os.path.splitext(name)[0]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    paths1 = sorted(glob.glob(os.path.join(DIR1, GLOB1)))
    paths2 = sorted(glob.glob(os.path.join(DIR2, GLOB2)))

    files1 = {base_key(p, "_denoised.csv"): p for p in paths1}
    files2 = {base_key(p, "_noise.csv"): p for p in paths2}

    common_keys = sorted(set(files1.keys()) & set(files2.keys()))
    if not common_keys:
        raise RuntimeError("No matching base names between DIR1 and DIR2")

    print("Common base names:", common_keys)

    for key in common_keys:
        path1 = files1[key]
        path2 = files2[key]
        name1 = os.path.basename(path1)
        name2 = os.path.basename(path2)

        print(f"\nMerging '{key}':")
        print(f"  DIR1 (5min): {name1}")
        print(f"  DIR2 (sec) : {name2}")

        df1 = pd.read_csv(path1)
        df2 = pd.read_csv(path2)

        # basic checks
        if TIME_COL not in df1.columns:
            print(f"  Skipping {name1}: missing {TIME_COL}")
            continue
        if TIME_COL not in df2.columns:
            print(f"  Skipping {name2}: missing {TIME_COL}")
            continue
        if PRED_COL not in df2.columns:
            print(f"  Skipping {name2}: missing {PRED_COL}")
            continue

        # parse time
        df1["_t"] = pd.to_datetime(df1[TIME_COL], utc=True, errors="coerce")
        df2["_t"] = pd.to_datetime(df2[TIME_COL], utc=True, errors="coerce")

        df1 = df1.dropna(subset=["_t"])
        df2 = df2.dropna(subset=["_t"])

        # --- downsample DIR2 (sec) to 5min to match DIR1 ---
        df2[PRED_COL] = pd.to_numeric(df2[PRED_COL], errors="coerce")
        df2 = df2.dropna(subset=[PRED_COL])

        df2_5min = (
            df2.set_index("_t")
               .resample("5min")[PRED_COL]
               .mean()
               .dropna()
               .reset_index()
        )

        # merge on 5min timestamps
        df2_sub = df2_5min.rename(columns={PRED_COL: PRED_COL})

        merged = pd.merge(df1, df2_sub, left_on="_t", right_on="_t", how="left")

        print("  non-NaN Bh_pred after merge:", merged[PRED_COL].notna().sum())

        merged = merged.drop(columns=["_t"])
        merged = merged.loc[:, ~merged.columns.duplicated()]

        out_path = os.path.join(OUT_DIR, f"{key}_merged.csv")
        merged.to_csv(out_path, index=False)
        print(f"  -> saved {out_path} ({len(merged)} rows)")

    print("\nDone.")

if __name__ == "__main__":
    main()
