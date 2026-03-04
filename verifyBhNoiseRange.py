import numpy as np
import pandas as pd

df = pd.read_csv("noise_EZIE_data/EZIE_Oct_27_2024_noise.csv")

mask_valid = (
    np.isfinite(df["Bh"]) &
    np.isfinite(df["Bh_pred"]) &
    np.isfinite(df["noise_Bh"])
)
df_valid = df[mask_valid]

print("noise max:", df_valid["noise_Bh"].max())
print("noise min:", df_valid["noise_Bh"].min())
print("noise range:",
      df_valid["noise_Bh"].max() - df_valid["noise_Bh"].min())
