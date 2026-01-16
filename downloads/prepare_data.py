# prepare_data.py (CORREGIDO)

import pandas as pd
import numpy as np

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
# 3. Features (P = 1, PER DAY)
# ===============================
# shape: (N assets, T observations, P=1)
features = np.zeros((N, T, 1))

for i in range(N):
    features[i, :, 0] = returns.iloc[:, i].values

# Normalization PER ASSET (paper)
features -= features.mean(axis=1, keepdims=True)
features /= features.std(axis=1, keepdims=True)

np.save("../data/features.npy", features)

print("✓ features.npy:", features.shape)
