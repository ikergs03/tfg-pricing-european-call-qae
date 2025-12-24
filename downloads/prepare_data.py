import numpy as np
import pandas as pd

# ===============================
# 1. Cargar precios
# ===============================
prices = pd.read_parquet("../data/prices.parquet")

# 🔴 LIMPIEZA CRÍTICA
if isinstance(prices.columns, pd.MultiIndex):
    prices.columns = prices.columns.get_level_values(-1)

prices.columns.name = None
prices.index.name = None

# ===============================
# 2. Retornos logarítmicos
# ===============================
returns = np.log(prices / prices.shift(1)).dropna()
returns.to_csv("../data/returns.csv")

print(f"returns.csv creado con shape {returns.shape}")

# ===============================
# 3. Construcción de features (P=6)
# ===============================
window_vol = 30
window_mom1 = 14
window_mom2 = 60

features = []

for ticker in returns.columns:
    r = returns[ticker]
    
    f1 = r.values
    f2 = r.rolling(window_vol).std().fillna(method='bfill').fillna(method='ffill').values
    f3 = r.rolling(window_mom1).mean().fillna(method='bfill').fillna(method='ffill').values
    f4 = r.rolling(window_mom2).mean().fillna(method='bfill').fillna(method='ffill').values
    f5 = -r.rolling(7).mean().fillna(method='bfill').fillna(method='ffill').values
    # Cambio: usa returns en lugar de prices para f6
    f6 = r.rolling(30).mean().fillna(method='bfill').fillna(method='ffill').values
    
    X = np.column_stack([f1, f2, f3, f4, f5, f6])
    
    # estandarización
    X = (X - np.nanmean(X, axis=0)) / np.nanstd(X, axis=0)
    
    features.append(X)

# Alinear longitud temporal
min_T = min(f.shape[0] for f in features)
features = np.array([f[-min_T:] for f in features])

np.save("../data/features.npy", features)

print(f"features.npy creado con shape {features.shape}")


