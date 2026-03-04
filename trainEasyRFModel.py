"""
EASY MODEL (Random Forest):
- Loads ALL daily CSVs in a folder
- Uses timeString/time for timestamps (no filename date parsing)
- Optional resampling (recommended)
- Builds lag features from the last N timesteps (depends on resample)
- Optional rolling stats
- Optional interaction features (only when using both Bh+ctemp)
- Trains RandomForestRegressor
- Saves model + predictions + feature importances

IMPORTANT:
- 'lags' means last N rows.
  If --resample 5min and --lags 144 -> last 12 hours.
  If --resample none, lags are in raw rows (likely seconds).
"""

import os
import glob
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
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
# Feature engineering
# -------------------------
def add_lags(df: pd.DataFrame, col: str, lags: int) -> pd.DataFrame:
    lag_df = pd.DataFrame(
        {f"{col}_lag{k}": df[col].shift(k) for k in range(1, lags + 1)},
        index=df.index,
    )
    return pd.concat([df, lag_df], axis=1)


def add_rolling_stats(df: pd.DataFrame, col: str, window: int) -> pd.DataFrame:
    s = df[col].shift(1)
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


def build_features(df: pd.DataFrame, input_cols: list[str], lags: int, add_stats: bool):
    for c in input_cols:
        df = add_lags(df, c, lags)
        if add_stats:
            df = add_rolling_stats(df, c, window=lags)

    feature_cols = []
    for c in input_cols:
        feature_cols += [f"{c}_lag{k}" for k in range(1, lags + 1)]
        if add_stats:
            feature_cols += [
                f"{c}_rollmean{lags}",
                f"{c}_rollstd{lags}",
                f"{c}_rollmax{lags}",
                f"{c}_rollmin{lags}",
                f"{c}_slope{lags}",
            ]
    return df, feature_cols


# -------------------------
# Loading + resampling
# -------------------------
def load_folder_concat(data_dir: str, glob_pat: str, timestring_col: str, time_col: str):
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

        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["_time"]).sort_values("_time").reset_index(drop=True)
    return out


def resample_df(df: pd.DataFrame, freq: str, keep_cols: list[str]):
    d = df[["_time"] + keep_cols].copy()
    for c in keep_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d = d.set_index("_time").resample(freq).mean().dropna().reset_index()
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

    ap.add_argument("--use_inputs", default="ctemp", choices=["both", "bh", "ctemp"])
    ap.add_argument("--add_interactions", action="store_true")

    ap.add_argument("--resample", default="5min")
    ap.add_argument("--lags", type=int, default=144)
    ap.add_argument("--add_stats", action="store_true")

    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--n_estimators", type=int, default=600)
    ap.add_argument("--max_depth", type=int, default=None)
    ap.add_argument("--min_samples_leaf", type=int, default=2)

    ap.add_argument("--out_dir", default="out_easy")

    # MODEL STATE ARGS
    ap.add_argument("--model_state_name", default="model_states/rf_model")
    ap.add_argument("--model_dir", default=".")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    # allow nested model_state_name paths
    parent = os.path.dirname(args.model_state_name)
    if parent:
        os.makedirs(os.path.join(args.model_dir, parent), exist_ok=True)

    model_path = os.path.join(args.model_dir, f"{args.model_state_name}.joblib")

    raw = load_folder_concat(args.data_dir, args.glob, args.timestring_col, args.time_col)

    input_cols = []
    if args.use_inputs in ("both", "bh"):
        input_cols.append(args.bh_col)
    if args.use_inputs in ("both", "ctemp"):
        input_cols.append(args.ctemp_col)

    keep_cols = [args.target_col] + input_cols

    if args.resample.lower() != "none":
        df = resample_df(raw, args.resample, keep_cols)
    else:
        df = raw[["_time"] + keep_cols].dropna().reset_index(drop=True)

    df, feature_cols = build_features(df, input_cols, args.lags, args.add_stats)

    if args.add_interactions and args.use_inputs == "both":
        df["bh_ctemp_lag1"] = df[f"{args.bh_col}_lag1"] * df[f"{args.ctemp_col}_lag1"]
        df["abs_dBh_lag1"] = (df[f"{args.bh_col}_lag1"] - df[f"{args.bh_col}_lag2"]).abs()
        df["abs_dBh_ctemp_lag1"] = df["abs_dBh_lag1"] * df[f"{args.ctemp_col}_lag1"]
        feature_cols += ["bh_ctemp_lag1", "abs_dBh_lag1", "abs_dBh_ctemp_lag1"]

    model_df = df.dropna(subset=feature_cols + [args.target_col])

    train_df, test_df = time_split(model_df, args.test_frac)

    X_train = train_df[feature_cols].values
    y_train = train_df[args.target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[args.target_col].values

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        random_state=args.seed,
        n_jobs=-1,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
    )
    model.fit(X_train, y_train)

    print("Train:", metrics(y_train, model.predict(X_train)))
    print("Test: ", metrics(y_test, model.predict(X_test)))

    # SAVE MODEL STATE
    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "input_cols": input_cols,
            "target_col": args.target_col,
            "args": vars(args),
        },
        model_path,
    )

    print(f"Saved model state to: {model_path}")


if __name__ == "__main__":
    main()
