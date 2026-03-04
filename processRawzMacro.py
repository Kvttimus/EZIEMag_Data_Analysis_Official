import subprocess

# Constants
base_command = "python processRawzToCSV.py"
input_dir_root = "uncompressed_EZIE_data"
output_dir = "decompressed_EZIE_data"

MONTH_MAP = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

year = 2025
# month = 10


for month in range(1, 7):
    month_name = MONTH_MAP.get(month, f"{month:02d}")
    # Loop from Oct 7 to Oct 27 inclusive
    for day in range(1, 32):
        day_str = f"{day:02d}"  # 07, 08, ..., 27
        date_str = f"{year}{month:02d}{day_str}"

        input_dir = f"{input_dir_root}/{date_str}"
        output_file = f"EZIE_{month_name}_{day}_{year}.csv"

        # Full command as a list (safer for subprocess)
        cmd = [
            "python",
            "processRawzToCSV.py",
            input_dir,
            "--recursive",
            "--outdir", output_dir,
            "--outfile", output_file,
        ]

        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
