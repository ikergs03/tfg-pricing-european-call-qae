import numpy as np
import pandas as pd

# ===============================
# 1. Cargar precios
# ===============================
prices = pd.read_parquet("../data/prices.parquet")

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

    f1 = r
    f2 = r.rolling(window_vol).std()
    f3 = r.rolling(window_mom1).mean()
    f4 = r.rolling(window_mom2).mean()
    f5 = -r.rolling(7).mean()
    f6 = prices[ticker].pct_change().rolling(30).mean()

    X = pd.concat([f1, f2, f3, f4, f5, f6], axis=1).dropna()

    # estandarización
    X = (X - X.mean()) / X.std()

    features.append(X.values)

# Alinear longitud temporal
min_T = min(f.shape[0] for f in features)
features = np.array([f[-min_T:] for f in features])

np.save("../data/features.npy", features)

print(f"features.npy creado con shape {features.shape}")
