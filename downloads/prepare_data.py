# prepare_data.py
#
# Replicación exacta del pipeline fetch_stock_data del autor en
# quantum_hrp.py — sin filtrado ni features personalizadas.
#
# Pipeline (de quantum_hrp.py líneas 249-310):
#   1. Cargar precios (todos los tickers, incluyendo crypto/RIVN/LCID etc.)
#   2. dropna(axis=1, how='all')   — eliminar tickers sin datos
#   3. dropna(how='all')           — eliminar filas completamente NaN
#   4. pct_change().dropna()       — retornos, eliminar filas con CUALQUIER NaN
#   5. compute_financial_features(returns, t, window=5):
#        f1 = retorno actual
#        f2 = media móvil  (ventana=5)
#        f3 = desv. estándar móvil  (ventana=5, ddof=1)
#        f4 = retorno acumulado (ventana=5)
#        f5 = asimetría      (ventana=5, scipy.stats.skew)
#        f6 = curtosis      (ventana=5, scipy.stats.kurtosis, Fisher/exceso)

import pandas as pd
import numpy as np
from scipy import stats

# ===============================
# 1. Cargar precios brutos
# ===============================
prices_raw = pd.read_parquet("../data/prices.parquet")

# Aplanar columnas MultiIndex de yfinance (ej. ('SPY','SPY') → 'SPY')
if isinstance(prices_raw.columns, pd.MultiIndex):
    prices_raw.columns = prices_raw.columns.get_level_values(0)
elif isinstance(prices_raw.columns[0], tuple):
    prices_raw.columns = [c[0] if isinstance(c, tuple) else c for c in prices_raw.columns]

print(f"Precios brutos: {prices_raw.shape}  ({prices_raw.index.min().date()} → {prices_raw.index.max().date()})")
print(f"Tasa de NaN bruta: {prices_raw.isna().sum().sum() / prices_raw.size * 100:.1f}%")

# ===============================
# 2. Manejo exacto de NaN del autor (quantum_hrp.py líneas 269-280)
# ===============================
# Eliminar tickers (columnas) completamente NaN
prices = prices_raw.dropna(axis=1, how='all')
print(f"Tras dropna(axis=1, how='all'): {prices.shape}")

# Eliminar filas con todos NaN
prices = prices.dropna(how='all')
print(f"Tras dropna(how='all'): {prices.shape}")

# ===============================
# 3. Calcular retornos diarios — enfoque exacto del autor
# ===============================
# NOTA: pct_change() en un DataFrame con huecos NaN produce retornos NaN
# en las transiciones NaN→válido y válido→NaN. Luego dropna() (how='any')
# elimina TODAS las filas donde CUALQUIER ticker tenga un retorno NaN.
# Con RIVN (primer dato válido 2021-11-10), esto trunca a ~1147 días.
asset_returns = prices.pct_change().dropna()

# Aplanar nombres de columnas (asegurar strings simples)
asset_returns.columns = [c[0] if isinstance(c, tuple) else c for c in asset_returns.columns]

T, N = asset_returns.shape
print(f"\nRetornos: {asset_returns.shape}  (T={T}, N={N})")
print(f"Rango de fechas: {asset_returns.index.min().date()} → {asset_returns.index.max().date()}")

asset_returns.to_csv("../data/returns.csv")
print("✓ returns.csv guardado")

# ===============================
# 4. Características — compute_financial_features del autor (returns, t, window=5)
# ===============================
P = 6
WINDOW = 5
asset_returns_np = asset_returns.to_numpy()  # shape (T, N)

features = np.zeros((N, T, P))

for i in range(N):
    ticker = asset_returns.columns[i]
    asset_series = asset_returns_np[:, i]

    if (i + 1) % 20 == 0 or i == 0 or i == N - 1:
        print(f"[{i+1:3d}/{N}] Calculando características para {ticker}")

    for t in range(T):
        ret = asset_series[t]
        start = max(0, t - WINDOW + 1)
        window_returns = asset_series[start:t+1]

        f1 = ret                                        # retorno actual
        f2 = np.mean(window_returns)                    # media móvil
        f3 = np.std(window_returns, ddof=1)             # desv. estándar móvil
        f4 = np.prod(1 + window_returns) - 1            # retorno acumulado
        f5 = stats.skew(window_returns)                 # asimetría
        f6 = stats.kurtosis(window_returns)             # curtosis (exceso)

        features[i, t, :] = np.array([f1, f2, f3, f4, f5, f6])

# ===============================
# 5. Guardar
# ===============================
np.save("../data/features.npy", features)

# Guardar también la lista de tickers como referencia
pd.Series(asset_returns.columns.tolist()).to_csv("../data/tickers.csv", index=False, header=["ticker"])

print(f"\n✓ features.npy guardado — shape {features.shape}")
print(f"  Cantidad de NaN en features: {np.isnan(features).sum()}")
print(f"  Cantidad de Inf en features: {np.isinf(features).sum()}")
print(f"✓ tickers.csv guardado — {N} tickers")

print(f"\n✓ features.npy: {features.shape}")
print(f"✓ Dimensiones: (N_activos={N}, T={T}, P=6)")
print(f"✓ tickers.csv: {N} tickers guardados")
