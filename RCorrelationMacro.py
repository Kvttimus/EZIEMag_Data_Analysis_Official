"""
correlateMacro.py

Simple macro wrapper around compute_R from correlate_any.
"""

from RCorrelation import compute_R

year = 2024
month = 10

def correlateMacro(file1, col1, file2=None, col2=None, hours_min=None, hours_max=None, day=None):
    """
    Macro-style helper:
      - file1, col1: first signal
      - file2, col2: second signal (if file2 is None, uses file1)

    Prints and returns R.
    """
    R = compute_R(file1=file1, col1=col1, file2=file2, col2=col2, hours_min=hours_min, hours_max=hours_max)
    if file2 is None:
        file2 = file1
    # print(f"R({col1} in {file1}, {col2} in {file2}) = {R:.6f}")
    print(f"October {day}, {year} --- R({col1} vs {col2}) = {R:.6f}")
    return R

# CSV vs SEC

for day in range(7, 28):
    day_str = f"{day:02d}"            # 07, 08, … 27
    date_str = f"{year}-{month:02d}-{day_str}"  # 2024-10-07, etc.

    R_cross = correlateMacro(
        file1=f"despiked_EZIE_data/EZIE_Oct_{day}_{year}_despiked.csv",
        col1="Gz",
        # file2=f"despiked_FRD_data/frd{year}{month:02d}{day_str}psec_despiked.sec",
        file2=f"despiked_EZIE_data/EZIE_Oct_{day}_{year}_despiked.csv",
        col2="ctemp",
        # hours_min=0,
        # hours_max=12,
        day=day,
    )
