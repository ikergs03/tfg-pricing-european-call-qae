# prepare_data.py
# Generates features.npy using the EXACT same 6 features as the paper's
# compute_financial_features(returns, t, window=5):
#   f1 = r_t                       (current return)
#   f2 = mean(window returns)      (rolling mean)
#   f3 = std(window returns, ddof=1) (rolling std)
#   f4 = prod(1 + window) - 1      (cumulative return)
#   f5 = skew(window returns)       (skewness)
#   f6 = kurtosis(window returns)   (kurtosis)

import pandas as pd
import numpy as np
from scipy import stats

# ===============================
# 1. Load prices
# ===============================
prices = pd.read_parquet("../data/prices.parquet")
prices = prices.dropna(axis=0, how="any")

# ===============================
# 2. Returns
# ===============================
returns = prices.pct_change().dropna()
returns.to_csv("../data/returns.csv")

T, N = returns.shape
print("Returns shape:", returns.shape)

# ===============================
# 3. Features (P = 6) — paper's compute_financial_features
# ===============================
WINDOW = 5  # same default as the paper

features = np.zeros((N, T, 6))

for i in range(N):
    print(f"[{i+1:3d}/{N}] Computing features for {returns.columns[i]}")
    r = returns.iloc[:, i].values  # 1-D array of returns

    for t in range(T):
        start = max(0, t - WINDOW + 1)
        w = r[start:t+1]

        features[i, t, 0] = r[t]                              # f1: current return
        features[i, t, 1] = np.mean(w)                        # f2: rolling mean
        features[i, t, 2] = np.std(w, ddof=1) if len(w) > 1 else 0.0  # f3: rolling std
        features[i, t, 3] = np.prod(1 + w) - 1                # f4: cumulative return
        features[i, t, 4] = stats.skew(w) if len(w) > 2 else 0.0      # f5: skewness
        features[i, t, 5] = stats.kurtosis(w) if len(w) > 3 else 0.0  # f6: kurtosis

# ===============================
# 4. Save (NO pre-standardization; average_density_matrix
#    does its own min-max normalization internally)
# ===============================
np.save("../data/features.npy", features)
print("✓ features.npy:", features.shape)
print("✓ Dimensions: (N_assets={}, T={}, P=6)".format(N, T))
