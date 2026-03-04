import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# =======================
# USER SETTINGS (EDIT ME)
# =======================

CSV_PATH = "daily_EZIE_ranges_filtered.csv"
# X_COL = "ctemp_range"
X_COL = "ctemp_min"

Y_COLS = [
    # "actual_noise_Bh",
    "noise_Bh_range",
]

TITLE = "Bh_noise_range vs ctemp_min"
X_LABEL = "Minimum Temperature (°C)"
Y_LABEL = "Bh_noise_range (nT)"

ROTATE_X = 45
SAVE_PLOT = True
OUT_PATH = "Bh_noise_range_vs_ctemp_min.png"

# --- smoothing settings (same idea as createSurveyPlot.py) ---
# RESAMPLE_RULE = "1min"
RESAMPLE_RULE = None
REDUCER = None

# =======================
# PLOTTING LOGIC
# =======================

df = pd.read_csv(CSV_PATH)

# Parse timeString like survey script
if X_COL not in df.columns:
    raise ValueError(f"X column '{X_COL}' not found.")

ts = df[X_COL].astype(str).str.strip()
time = pd.to_datetime(ts, utc=True, errors="coerce", format="mixed")
if time.isna().any():
    mask = time.isna()
    time.loc[mask] = pd.to_datetime(ts[mask], utc=True, errors="coerce", format="ISO8601")

df["time"] = time
df = df.dropna(subset=["time"]).set_index("time").sort_index()

plt.figure(figsize=(12, 4))

for col in Y_COLS:
    if col not in df.columns:
        print(f"[WARN] Column '{col}' not found, skipping.")
        continue

    # numeric + drop NaNs
    y = pd.to_numeric(df[col], errors="coerce")

    sub = pd.DataFrame({col: y}).dropna()

    # --- SAME smoothing approach: 1-min bins + median ---
    if RESAMPLE_RULE:
        if REDUCER == "median":
            sub = sub.resample(RESAMPLE_RULE).median()
        else:
            sub = sub.resample(RESAMPLE_RULE).mean()

    # thin line so it doesn't look “thick”
    label = col
    if RESAMPLE_RULE:
        label = f"{col} ({RESAMPLE_RULE} {REDUCER})"

    plt.plot(sub.index, sub[col], label=label, linewidth=0.7)

ax = plt.gca()

ax.xaxis.set_major_locator(mdates.HourLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

ax.xaxis.set_minor_locator(mdates.MinuteLocator(interval=15))
ax.tick_params(axis="x", which="minor", length=3)

plt.xlabel(X_LABEL)
plt.ylabel(Y_LABEL)
plt.title(TITLE)
plt.legend()

if ROTATE_X:
    plt.xticks(rotation=ROTATE_X)

plt.tight_layout()

if SAVE_PLOT:
    plt.savefig(OUT_PATH, dpi=200)
    print(f"[OK] Saved plot to {OUT_PATH}")
    plt.show()
else:
    plt.show()