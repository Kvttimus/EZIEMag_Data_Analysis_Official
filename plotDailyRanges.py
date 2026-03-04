import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =======================
# USER SETTINGS (EDIT ME)
# =======================

# CSV_PATH = "daily_EZIE_ranges.csv"   # input file
CSV_PATH = "daily_EZIE_ranges_filtered.csv"
X_COL = "ctemp_max" 

Y_COLS = [
    "noise_Bh_range",
]

TITLE = "Bh_noise_range vs ctemp_max"
X_LABEL = "ctemp_max(°C)"
Y_LABEL = "Bh_noise_range (nT)"

ROTATE_X = 45        # degrees (0, 45, 90)
SAVE_PLOT = True
OUT_PATH = "Bh_noise_range_vs_ctemp_range"

# =======================
# PLOTTING LOGIC
# =======================

df = pd.read_csv(CSV_PATH)

# Drop rows where ctemp_range > 20 (also handles non-numeric safely)
df["ctemp_range"] = pd.to_numeric(df["ctemp_range"], errors="coerce")
df = df[df["ctemp_range"].notna() & (df["ctemp_range"] <= 20)].copy()

# X axis
if X_COL is None:
    x = range(len(df))
    x_label = "index"
else:
    if X_COL not in df.columns:
        raise ValueError(f"X column '{X_COL}' not found.")
    x = df[X_COL]

    # Try to parse dates nicely
    if df[X_COL].dtype == object:
        parsed = pd.to_datetime(df[X_COL], errors="coerce")
        if parsed.notna().any():
            x = parsed
    x_label = X_LABEL if X_LABEL else X_COL

plt.figure()

for col in Y_COLS:
    if col not in df.columns:
        print(f"[WARN] Column '{col}' not found, skipping.")
        continue

    if col != "ctemp_range":
        y = pd.to_numeric(df[col], errors="coerce")
    else:
        y = pd.to_numeric(df[col], errors="coerce")
    plt.scatter(x, y, label=col, s=2)
    # plt.plot(x, y, label=[col])

# --- START OF LINE OF BEST FIT CODE --
# regress on first one
reg_y_col = Y_COLS[0]

# build numeric x,y for regression
x_num = pd.to_numeric(df[X_COL], errors="coerce")

if reg_y_col != "ctemp_range":
    y_num = pd.to_numeric(df[reg_y_col], errors="coerce")
else:
    y_num = pd.to_numeric(df[reg_y_col], errors="coerce")

# drop NaNs so polyfit doesn't break
mask = x_num.notna() & y_num.notna()
x_fit = x_num[mask].to_numpy()
y_fit = y_num[mask].to_numpy()

m, b = np.polyfit(x_fit, y_fit, 1)
correlation_matrix = np.corrcoef(x_fit, y_fit)
r_value = correlation_matrix[0, 1]

# draw regression line across the x range
x_line = np.linspace(x_fit.min(), x_fit.max(), 200)
plt.plot(x_line, m * x_line + b, color="red",
         label=f"{reg_y_col} fit: y={m:.2f}x+{b:.2f}")

print(f"\nThe Pearson correlation constant (r) between x and y is: {r_value:.3f}")

# --- END OF LINE OF BEST FIT CODE --

# plt.ylim(0, 50)
plt.xlabel(x_label)
plt.ylabel(Y_LABEL)
plt.title(TITLE)
plt.legend(
    loc="lower right",   # move legend to lower-right
    fontsize=8,          # smaller legend text
    framealpha=0.7       # (optional) slightly transparent legend box
)

if ROTATE_X:
    plt.xticks(rotation=ROTATE_X)

# --- START OF LINE OF BEST FIT CODE --
plt.text(
    0.02, 0.95,
    f"Pearson r = {r_value:.3f}",
    transform=plt.gca().transAxes,  # axes-relative coordinates
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
)
# --- END OF LINE OF BEST FIT CODE --

plt.tight_layout()

if SAVE_PLOT:
    plt.savefig(OUT_PATH, dpi=200)
    print(f"[OK] Saved plot to {OUT_PATH}") 
    plt.show()
else:
    plt.show()
