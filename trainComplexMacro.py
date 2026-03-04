import subprocess
import sys

PYTHON_EXEC = sys.executable
SCRIPT = "trainComplexXGBoostModel.py"

DATA_DIR = "filtered_training_data"
GLOB = "*_noise.csv"

TIME_COL = "time"
TIMESTRING_COL = "timeString"

TARGET_COL = "noise_Bh"
BH_COL = "Bh"
CTEMP_COL = "ctemp"

RESAMPLE = "5min"
LAGS = 10

ADD_STATS = True
ADD_INTERACTIONS = True
SCREENING = "none"

OUT_DIR = "out_complex"

USE_INPUTS = "both"

MODEL_STATE_NAME = "model_states_filtered/xgb_both_5min_lesslag"


cmd = [
    PYTHON_EXEC, SCRIPT,
    "--data_dir", DATA_DIR,
    "--glob", GLOB,
    "--time_col", TIME_COL,
    "--timestring_col", TIMESTRING_COL,
    "--target_col", TARGET_COL,
    "--bh_col", BH_COL,
    "--ctemp_col", CTEMP_COL,
    "--resample", RESAMPLE,
    "--lags", str(LAGS),
    "--screening", SCREENING,
    "--out_dir", OUT_DIR,
    "--use_inputs", USE_INPUTS,
    "--model_state_name", MODEL_STATE_NAME,
]

print("======================================")
print("Running COMPLEX XGBoost model")
print("Command:")
print(" ".join(cmd))
print("--------------------------------------")

result = subprocess.run(cmd)
print("Done" if result.returncode == 0 else "Failed")
