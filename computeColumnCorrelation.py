import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--csv", required=True)
parser.add_argument("--x", required=True)
parser.add_argument("--y", required=True)
parser.add_argument("--label", required=True)
parser.add_argument("--hours", type=int, default=12)
parser.add_argument("--method", default="pearson", choices=["pearson", "spearman", "kendall"])

args = parser.parse_args()

df = pd.read_csv(args.csv)

# --- Parse timeString ---
if "timeString" not in df.columns:
    raise ValueError("Expected column 'timeString' not found.")

t = pd.to_datetime(df["timeString"], utc=True, errors="coerce")
df = df[t.notna()].copy()
t = t[t.notna()]

# --- Filter: keep only data where hour >= HOURS ---
df = df[t.dt.hour >= args.hours]

# --- Correlation on actual values ---
x = pd.to_numeric(df[args.x], errors="coerce")
y = pd.to_numeric(df[args.y], errors="coerce")

mask = x.notna() & y.notna()
x = x[mask]
y = y[mask]

if len(x) < 2:
    raise ValueError("Not enough data points after hour filter.")

corr = x.corr(y, method=args.method)

print(f"{args.label} Pearson Correlation (hour >= {args.hours}) = {corr:.5f}")
