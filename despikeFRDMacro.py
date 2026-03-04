from despikeFRD import despikeFRD

# # Despike a single file
# despikeFRD(
#     input_path="frd20241019psec.sec",
#     output_dir="despiked_FRD_data/",
# )

# Despike all CSVs in a folder
despikeFRD(
    input_path="frd_data/",
    output_dir="despiked_FRD_data/",
)
