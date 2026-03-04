import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# USER SETTINGS (EDIT ONCE)
# =========================
CSV_PATH = "despiked_EZIE_data/EZIE_Oct_10_2024_despiked.csv"
HOURS_TO_PLOT = 24.0

# Resampling (robust to spikes)
RESAMPLE_RULE = "10S"   # examples: "1S", "5S", "10S", "30S", "1min"

# Column names
TIME_COL = "timeString"
BH_COL = "Bh"          # Bh_true
CTEMP_COL = "ctemp"

SAVE_PLOT = True
OUT_PATH = "bh_ctemp_overlay_median_10s_FULLDAY.png"
DPI = 300

START_TIME_UTC = "2024-10-10T00:00:00Z" # Morning
# START_TIME_UTC = "2024-10-10T11:00:00Z" # Midday
# START_TIME_UTC = "2024-10-10T18:00:00Z" # Dusk


# =========================
# LOAD + PREP
# =========================
df = pd.read_csv(CSV_PATH)

df[TIME_COL] = pd.to_datetime(df[TIME_COL], utc=True, errors="coerce")
df = df.dropna(subset=[TIME_COL, BH_COL, CTEMP_COL]).sort_values(TIME_COL)

# Select first N hours
t0 = pd.to_datetime(START_TIME_UTC, utc=True)
t1 = t0 + pd.Timedelta(hours=HOURS_TO_PLOT)
w = df[(df[TIME_COL] >= t0) & (df[TIME_COL] <= t1)].copy()

if w.empty:
    raise RuntimeError("Selected time window is empty.")

# =========================
# RESAMPLE (MEDIAN)
# =========================
w = (
    w.set_index(TIME_COL)[[BH_COL, CTEMP_COL]]
     .resample(RESAMPLE_RULE)
     .median()
     .dropna()
     .reset_index()
)

if w.empty:
    raise RuntimeError("Resampling produced an empty dataframe.")

# =========================
# SCALE CTEMP → BH RANGE
# =========================
bh = w[BH_COL].astype(float).to_numpy()
ct = w[CTEMP_COL].astype(float).to_numpy()

bh_min, bh_max = np.nanmin(bh), np.nanmax(bh)
ct_min, ct_max = np.nanmin(ct), np.nanmax(ct)

if np.isclose(ct_max, ct_min):
    ct_scaled = np.full_like(ct, (bh_min + bh_max) / 2)
    scale = 1.0
    offset = ct_min
else:
    scale = (bh_max - bh_min) / (ct_max - ct_min)
    offset = ct_min
    ct_scaled = (ct - offset) * scale + bh_min

def scaled_to_ctemp(y):
    return (y - bh_min) / scale + offset

def ctemp_to_scaled(y):
    return (y - offset) * scale + bh_min

# =========================
# PLOT
# =========================
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(w[TIME_COL], bh, linewidth=1.0, label="Bh_true")
ax.plot(w[TIME_COL], ct_scaled, linewidth=1.0,
        label=f"ctemp (scaled, median {RESAMPLE_RULE})")

ax.set_xlabel("Time (UTC)")
ax.set_ylabel("Bh_true")
ax.grid(True, alpha=0.3)

secax = ax.secondary_yaxis("right",
    functions=(scaled_to_ctemp, ctemp_to_scaled)
)
secax.set_ylabel("ctemp (°C)")

ax.set_title(
    f"Bh_true and ctemp Overlay "
    f"({HOURS_TO_PLOT} hr window, median-resampled)"
)

ax.legend(loc="upper right")
plt.tight_layout()

if SAVE_PLOT:
    plt.savefig(OUT_PATH, dpi=DPI, bbox_inches="tight")

plt.show()