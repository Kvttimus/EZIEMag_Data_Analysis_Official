"""
Parameters
    ----------
    input_path : str or Path
        Path to a single CSV file OR a directory containing CSV files.
    output_path : str or Path
        REQUIRED. Directory where cleaned files are written.
    cols : list of str
        Column names to despike (e.g., ["Bx", "By", "Bz", "Ax", "Ay", "Az"]).
    time_col : str
        Name of the time column (default "tval").
    window : int
        Rolling window size for median/MAD (default 101).
    z_thresh : float
        Robust z-score threshold for spikes (default 8.0).
    suffix : str
        Suffix added before file extension for output (default "_despiked").
"""

from despikeEZIE import despikeEZIE

# # Despike a single file
# despikeEZIE(
#     input_path="EZIE_Oct_7_2024.csv",
#     output_path="despiked_EZIE_data/"
#     cols=["Bx", "By", "Bz", "Ax", "Ay", "Az", "Gx", "Gy", "Gz"],
# )

# Despike all CSVs in a folder
despikeEZIE(
    input_path="decompressed_EZIE_data/",
    output_path="despiked_EZIE_data/",
    # cols=["Bx", "By", "Bz", "Ax", "Ay", "Az", "Gx", "Gy", "Gz"],
    cols=["Gx"],
    # window=101,
    # z_thresh=8.0,
)
