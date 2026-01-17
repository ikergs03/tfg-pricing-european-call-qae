# prepare_data.py (CORREGIDO)
"""""
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
"""
# prepare_data.py (CORREGIDO - 6 FEATURES)

import pandas as pd
import numpy as np

# ===============================
# 1. Load prices and volumes
# ===============================

prices = pd.read_parquet("../data/prices.parquet")
volumes = pd.read_parquet("../data/volumes.parquet")

prices = prices.dropna(axis=0, how="any")
volumes = volumes.loc[prices.index]

# ===============================
# 2. Returns
# ===============================

returns = prices.pct_change().dropna()
returns.to_csv("../data/returns.csv")

T, N = returns.shape
print("Returns shape:", returns.shape)

# ===============================
# 3. Features (P = 6)
# ===============================

# shape: (N assets, T observations, P=6)
features = np.zeros((N, T, 6))

rolling_vol_window = 30
momentum_14_window = 14
momentum_60_window = 60
reversal_window = 7
volume_window = 30
volume_yearly_window = 252  # días trading en un año

for i in range(N):
    print(f"[{i+1:3d}/{N}] Computing features for {returns.columns[i]}")
    
    asset_returns = returns.iloc[:, i].values
    asset_volumes = volumes.iloc[:, i].values
    
    for t in range(T):
        # Feature 1: Daily Return (log)
        features[i, t, 0] = asset_returns[t]
        
        # Feature 2: 30-Day Rolling Volatility (annualized)
        start_vol = max(0, t - rolling_vol_window + 1)
        if t >= rolling_vol_window - 1:
            vol_window = asset_returns[start_vol:t+1]
            features[i, t, 1] = np.std(vol_window, ddof=1) * np.sqrt(252)  # Annualized
        else:
            features[i, t, 1] = 0
        
        # Feature 3: 14-Day Momentum
        start_mom_14 = max(0, t - momentum_14_window + 1)
        if t >= momentum_14_window - 1:
            mom_14_window = asset_returns[start_mom_14:t+1]
            features[i, t, 2] = np.sum(mom_14_window)  # Cumulative return
        else:
            features[i, t, 2] = 0
        
        # Feature 4: 60-Day Momentum
        start_mom_60 = max(0, t - momentum_60_window + 1)
        if t >= momentum_60_window - 1:
            mom_60_window = asset_returns[start_mom_60:t+1]
            features[i, t, 3] = np.sum(mom_60_window)  # Cumulative return
        else:
            features[i, t, 3] = 0
        
        # Feature 5: 7-Day Short-Term Reversal
        start_rev = max(0, t - reversal_window + 1)
        if t >= reversal_window - 1:
            rev_window = asset_returns[start_rev:t+1]
            features[i, t, 4] = -np.sum(rev_window)  # Negated for reversal
        else:
            features[i, t, 4] = 0
        
        # Feature 6: 30-Day Rolling Volume Average (normalized by yearly average)
        start_vol_avg = max(0, t - volume_window + 1)
        start_vol_yearly = max(0, t - volume_yearly_window + 1)
        
        if t >= volume_window - 1:
            vol_avg_window = asset_volumes[start_vol_avg:t+1]
            vol_yearly_window = asset_volumes[start_vol_yearly:t+1]
            
            avg_vol_30 = np.mean(vol_avg_window)
            avg_vol_yearly = np.mean(vol_yearly_window)
            
            if avg_vol_yearly > 0:
                features[i, t, 5] = avg_vol_30 / avg_vol_yearly
            else:
                features[i, t, 5] = 0
        else:
            features[i, t, 5] = 0

# ===============================
# 4. Standardization (PER ASSET)
# ===============================

# El paper normaliza por activo para evitar que un activo volatil domine
for i in range(N):
    for p in range(6):
        feature_vec = features[i, :, p]
        # Evita división por cero
        std_val = np.std(feature_vec, ddof=1)
        if std_val > 0:
            features[i, :, p] = (feature_vec - np.mean(feature_vec)) / std_val

# ===============================
# 5. Save
# ===============================

np.save("../data/features.npy", features)
print("✓ features.npy:", features.shape)
print("✓ Dimensiones: (N_assets={}, T={}, P=6)".format(N, T))
