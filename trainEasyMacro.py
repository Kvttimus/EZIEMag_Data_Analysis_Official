import subprocess
import sys

PYTHON_EXEC = sys.executable
SCRIPT = "trainEasyRFModel.py"

# -------------------------
# Data
# -------------------------
DATA_DIR = "filtered_training_data"
GLOB = "*_noise.csv"

TIMESTRING_COL = "timeString"

TARGET_COL = "noise_Bh"
BH_COL = "Bh"
CTEMP_COL = "ctemp"

# -------------------------
# Experiment config
# -------------------------
# RESAMPLE = "5min"
# LAGS = 144

RESAMPLE = "1min"
LAGS = 720

OUT_DIR = "out_easy"

USE_INPUTS = "bh"

ADD_STATS = True
ADD_INTERACTIONS = True 

MODEL_STATE_NAME = "model_states_filtered/rf_bh_1min"
MODEL_DIR = "." 

# -------------------------
# Command
# -------------------------
cmd = [
    PYTHON_EXEC, SCRIPT,
    "--data_dir", DATA_DIR,
    "--glob", GLOB,
    "--timestring_col", TIMESTRING_COL,
    "--target_col", TARGET_COL,
    "--bh_col", BH_COL,
    "--ctemp_col", CTEMP_COL,
    "--resample", RESAMPLE,
    "--lags", str(LAGS),
    "--out_dir", OUT_DIR,
    "--use_inputs", USE_INPUTS,
    "--model_state_name", MODEL_STATE_NAME,
    "--model_dir", MODEL_DIR,
]

if ADD_STATS:
    cmd.append("--add_stats")

if ADD_INTERACTIONS:
    cmd.append("--add_interactions")

print("======================================")
print(f"Running EASY RF model (use_inputs={USE_INPUTS})")
print("Command:")
print(" ".join(cmd))
print("--------------------------------------")

result = subprocess.run(cmd)
print("Done" if result.returncode == 0 else "Failed")
