"""
COMPLEX MODEL (XGBoost):
- Loads ALL daily CSVs in a folder
- Uses timeString/time for timestamps (no filename date parsing)
- Toggled resampling 
- Builds lag features for: Bh raw history + ctemp history
- Saves model + predictions + feature importances

IMPORTANT:
- 'lags' means last N timesteps, where timestep duration depends on --resample.
  Example: --resample 5min and --lags 10 => last 10 time series
"""

import os
import glob
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------------
# Metrics / split
# -------------------------
def metrics(y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def time_split(df: pd.DataFrame, test_frac: float):
    n = len(df)
    split = int(np.floor(n * (1 - test_frac)))
    return df.iloc[:split].copy(), df.iloc[split:].copy()


# -------------------------
# Feature engineering (NO fragmentation)
# -------------------------
def add_lags_concat(df: pd.DataFrame, col: str, lags: int) -> pd.DataFrame:
    # Sparse lags: dense near present, coarse further back
    dense = list(range(1, min(31, lags + 1)))          # every minute up to 30
    coarse = list(range(60, lags + 1, 60))             # every hour after that
    lag_steps = sorted(set(dense + coarse))
    lag_df = pd.DataFrame(
        {f"{col}_lag{k}": df[col].shift(k) for k in lag_steps},
        index=df.index,
    )
    return pd.concat([df, lag_df], axis=1)


def add_rolling_stats_concat(df: pd.DataFrame, col: str, window: int) -> pd.DataFrame:
    s = df[col].shift(1)  # prior-only

    stats_df = pd.DataFrame(
        {
            f"{col}_rollmean{window}": s.rolling(window).mean(),
            f"{col}_rollstd{window}": s.rolling(window).std(),
            f"{col}_rollmax{window}": s.rolling(window).max(),
            f"{col}_rollmin{window}": s.rolling(window).min(),
            f"{col}_slope{window}": (s - s.shift(window - 1)) / max(window - 1, 1),
        },
        index=df.index,
    )
    return pd.concat([df, stats_df], axis=1)


def build_features(df: pd.DataFrame, cols: list[str], lags: int, add_stats: bool) -> tuple[pd.DataFrame, list[str]]:
    # Add features column-group-by-column-group with concat (fast, no fragmentation)
    for c in cols:
        df = add_lags_concat(df, c, lags)
        if add_stats:
            df = add_rolling_stats_concat(df, c, window=lags)

    feat = []
    for c in cols:
        dense = list(range(1, min(31, lags + 1)))
        coarse = list(range(60, lags + 1, 60))
        lag_steps = sorted(set(dense + coarse))
        feat += [f"{c}_lag{k}" for k in lag_steps]
        if add_stats:
            feat += [
                f"{c}_rollmean{lags}",
                f"{c}_rollstd{lags}",
                f"{c}_rollmax{lags}",
                f"{c}_rollmin{lags}",
                f"{c}_slope{lags}",
            ]
    return df, feat


def apply_screening(df: pd.DataFrame, method: str, cols: list[str]) -> pd.DataFrame:
    method = (method or "none").lower()
    if method == "none":
        return df

    if method == "drop_extreme":
        out = df.copy()

        def robust_z(x):
            x = x.astype(float)
            med = np.nanmedian(x)
            mad = np.nanmedian(np.abs(x - med))
            if mad == 0 or np.isnan(mad):
                return np.zeros_like(x)
            return 0.6745 * (x - med) / mad

        for c in cols:
            z = robust_z(out[c].values)
            out = out[np.abs(z) < 8]
        return out

    raise ValueError(f"Unknown screening: {method}")


# -------------------------
# Loading + resampling
# -------------------------
def load_folder_concat(data_dir: str, glob_pat: str, timestring_col: str, time_col: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(data_dir, glob_pat)))
    if not paths:
        raise FileNotFoundError(f"No files matched {glob_pat} in {data_dir}")

    frames = []
    for p in paths:
        df = pd.read_csv(p)

        if timestring_col in df.columns:
            df["_time"] = pd.to_datetime(df[timestring_col], errors="coerce", utc=True)
        elif time_col in df.columns:
            df["_time"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        else:
            raise ValueError(f"{p} missing both '{timestring_col}' and '{time_col}'.")

        df["_source_file"] = os.path.basename(p)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["_time"]).sort_values("_time").reset_index(drop=True)
    return out


def resample_df(df: pd.DataFrame, freq: str, keep_cols: list[str]) -> pd.DataFrame:
    keep_cols = list(dict.fromkeys(keep_cols))  # preserves order, removes duplicates

    d = df[["_time"] + keep_cols].copy()
    for c in keep_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.dropna(subset=["_time"]).set_index("_time")
    d = d.resample(freq).mean()
    d = d.dropna().reset_index()
    return d


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--glob", default="*_noise.csv")

    ap.add_argument("--timestring_col", default="timeString")
    ap.add_argument("--time_col", default="time")

    ap.add_argument("--target_col", default="noise_Bh")
    ap.add_argument("--bh_col", default="Bh")
    ap.add_argument("--ctemp_col", default="ctemp")

    ap.add_argument("--resample", default="5min", help="e.g., 1min, 5min, or 'none'")
    ap.add_argument("--lags", type=int, default=144, help="Lag steps (depends on --resample)")
    ap.add_argument("--add_stats", action="store_true")
    ap.add_argument("--add_interactions", action="store_true")

    ap.add_argument("--screening", default="none", choices=["none", "drop_extreme"])
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--out_dir", default="out_complex")

    ap.add_argument(
        "--use_inputs",
        default="both",
        choices=["both", "bh", "ctemp"],
        help="Which inputs to use for features: both, bh-only, or ctemp-only",
    )

    ap.add_argument(
        "--model_state_name",
        default="complex_xgb",
        help="Base filename (without .joblib) for saving/loading model state",
    )

    # XGB params
    ap.add_argument("--xgb_n_estimators", type=int, default=1500)
    ap.add_argument("--xgb_max_depth", type=int, default=6)
    ap.add_argument("--xgb_learning_rate", type=float, default=0.03)
    ap.add_argument("--xgb_subsample", type=float, default=0.9)
    ap.add_argument("--xgb_colsample_bytree", type=float, default=0.9)
    ap.add_argument("--xgb_reg_lambda", type=float, default=1.0)
    ap.add_argument("--xgb_reg_alpha", type=float, default=1.0)
    ap.add_argument("--xgb_min_child_weight", type=int, default=10)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(f"{args.model_state_name}.joblib")

    raw = load_folder_concat(args.data_dir, args.glob, args.timestring_col, args.time_col)

    required = [args.target_col]
    if args.use_inputs in ("both", "bh"):
        required.append(args.bh_col)
    if args.use_inputs in ("both", "ctemp"):
        required.append(args.ctemp_col)

    for c in required:
        if c not in raw.columns:
            raise ValueError(f"Missing required column '{c}' in CSVs.")

    keep_cols = [args.target_col]
    if args.use_inputs in ("both", "bh"):
        keep_cols.append(args.bh_col)
    if args.use_inputs in ("both", "ctemp"):
        keep_cols.append(args.ctemp_col)

    if args.resample.lower() != "none":
        df = resample_df(raw, args.resample, keep_cols=keep_cols)
        timestep_desc = f"{args.resample} per row"
    else:
        df = raw[["_time"] + keep_cols].copy()
        for c in keep_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna().sort_values("_time").reset_index(drop=True)
        timestep_desc = "raw rows (likely ~1 second)"

    screen_cols = [args.target_col]
    if args.use_inputs in ("both", "bh"):
        screen_cols.append(args.bh_col)
    if args.use_inputs in ("both", "ctemp"):
        screen_cols.append(args.ctemp_col)

    df = apply_screening(df, args.screening, cols=screen_cols)

    print("======================================")
    print("COMPLEX MODEL (XGBoost)")
    print(f"Rows: {len(df):,}")
    print(f"Timestep: {timestep_desc}")
    print(f"Lags: {args.lags} timesteps (depends on --resample)")
    print("======================================")

    # diagnostics
    print("ctemp summary:")
    print(df[args.ctemp_col].describe())

    print("Bh summary:")
    print(df[args.bh_col].describe())

    df["ctemp_bin"] = pd.qcut(df[args.ctemp_col], 4, duplicates="drop")
    print("noise_Bh by ctemp quartile:")
    print(
        df.groupby("ctemp_bin")[args.target_col]
        .agg(["mean", "std", "count"])
    )

    # inputs for lags
    input_cols = [args.target_col]
    if args.use_inputs in ("both", "bh"):
        input_cols.append(args.bh_col)
    if args.use_inputs in ("both", "ctemp"):
        input_cols.append(args.ctemp_col)

    df, feature_cols = build_features(df, input_cols, args.lags, add_stats=args.add_stats)

    # no is_hot anymore
    model_df = df.dropna(subset=feature_cols + [args.target_col]).copy()

    if args.add_interactions:
        if args.use_inputs != "both":
            print("NOTE: --add_interactions ignored unless --use_inputs both")
        else:
            df["bh_ctemp_lag1"] = df[f"{args.bh_col}_lag1"] * df[f"{args.ctemp_col}_lag1"]
            df["abs_dBh_lag1"] = (df[f"{args.bh_col}_lag1"] - df[f"{args.bh_col}_lag2"]).abs()
            df["abs_dBh_ctemp_lag1"] = df["abs_dBh_lag1"] * df[f"{args.ctemp_col}_lag1"]
            feature_cols += ["bh_ctemp_lag1", "abs_dBh_lag1", "abs_dBh_ctemp_lag1"]

    model_df = df.dropna(subset=feature_cols + [args.target_col]).copy()

    if len(model_df) < 200:
        raise ValueError(
            f"Not enough rows after lagging ({len(model_df)}). "
            f"Try smaller --lags or coarser --resample (e.g., 10min)."
        )

    print("Overall time range:", model_df["_time"].min(), "→", model_df["_time"].max())
    print("Overall target mean/std:",
          model_df[args.target_col].mean(),
          model_df[args.target_col].std())

    train_df, test_df = time_split(model_df, args.test_frac)

    print("Train time range:", train_df["_time"].min(), "→", train_df["_time"].max())
    print("Test  time range:", test_df["_time"].min(), "→", test_df["_time"].max())
    print("Train target mean/std:",
          train_df[args.target_col].mean(),
          train_df[args.target_col].std())
    print("Test  target mean/std:",
          test_df[args.target_col].mean(),
          test_df[args.target_col].std())

    X_train = train_df[feature_cols].values
    y_train = train_df[args.target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[args.target_col].values

    # persistence baseline
    model_df["y_persist"] = model_df[args.target_col].shift(1)
    baseline_df = model_df.dropna(subset=[args.target_col, "y_persist"])

    train_b, test_b = time_split(baseline_df, args.test_frac)
    y_train_b = train_b[args.target_col].values
    yhat_train_b = train_b["y_persist"].values
    y_test_b = test_b[args.target_col].values
    yhat_test_b = test_b["y_persist"].values

    print("Baseline persistence Train:", metrics(y_train_b, yhat_train_b))
    print("Baseline persistence Test: ", metrics(y_test_b,  yhat_test_b))

    model_type = "xgboost"
    try:
        from xgboost import XGBRegressor  # type: ignore

        model = XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=5.0,
            reg_alpha=5.0,
            min_child_weight=20,
            random_state=args.seed,
            n_jobs=-1,
            objective="reg:squarederror",
        )
    except Exception as e:
        model_type = "sklearn_hist_gbdt"
        from sklearn.ensemble import HistGradientBoostingRegressor

        print("NOTE: xgboost not available; falling back to HistGradientBoostingRegressor.")
        print(f"Reason: {e}")

        model = HistGradientBoostingRegressor(
            random_state=args.seed,
            learning_rate=0.06,
            max_depth=6,
            max_iter=900,
        )

    model.fit(X_train, y_train)
    yhat_train = model.predict(X_train)
    yhat_test = model.predict(X_test)

    print("Model:", model_type)
    print("Train:", metrics(y_train, yhat_train))
    print("Test: ", metrics(y_test, yhat_test))

    joblib.dump(
        {
            "model": model,
            "model_type": model_type,
            "feature_cols": feature_cols,
            "input_cols": input_cols,
            "target_col": args.target_col,
            "resample": args.resample,
            "lags": args.lags,
            "args": vars(args),
        },
        model_path,
    )

    pred = pd.DataFrame({"time": test_df["_time"].values, "y_true": y_test, "y_pred": yhat_test})
    pred.to_csv(os.path.join(args.out_dir, "complex_predictions.csv"), index=False)

    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame({"feature": feature_cols, "importance": model.feature_importances_}).sort_values(
            "importance", ascending=False
        )
        fi.to_csv(os.path.join(args.out_dir, "complex_feature_importances.csv"), index=False)

    print(f"Saved to: {args.out_dir}")
    print(f"Saved model to: {model_path}")


if __name__ == "__main__":
    main()
