# prepare_data.py
#
# EXACT replication of the author's fetch_stock_data pipeline from
# quantum_hrp.py — no filtering, no custom features.
#
# Pipeline (from quantum_hrp.py lines 249-310):
#   1. Load prices (all tickers, including crypto/RIVN/LCID etc.)
#   2. dropna(axis=1, how='all')   — drop tickers with zero data
#   3. dropna(how='all')           — drop fully-NaN rows
#   4. pct_change().dropna()       — returns, drop rows with ANY NaN
#   5. compute_financial_features(returns, t, window=5):
#        f1 = current return
#        f2 = rolling mean  (window=5)
#        f3 = rolling std   (window=5, ddof=1)
#        f4 = cumulative return (window=5)
#        f5 = skewness      (window=5, scipy.stats.skew)
#        f6 = kurtosis      (window=5, scipy.stats.kurtosis, Fisher/excess)

import pandas as pd
import numpy as np
from scipy import stats

# ===============================
# 1. Load raw prices
# ===============================
prices_raw = pd.read_parquet("../data/prices.parquet")

# Flatten MultiIndex columns from yfinance (e.g. ('SPY','SPY') → 'SPY')
if isinstance(prices_raw.columns, pd.MultiIndex):
    prices_raw.columns = prices_raw.columns.get_level_values(0)
elif isinstance(prices_raw.columns[0], tuple):
    prices_raw.columns = [c[0] if isinstance(c, tuple) else c for c in prices_raw.columns]

print(f"Raw prices: {prices_raw.shape}  ({prices_raw.index.min().date()} → {prices_raw.index.max().date()})")
print(f"Raw NaN rate: {prices_raw.isna().sum().sum() / prices_raw.size * 100:.1f}%")

# ===============================
# 2. Author's exact NaN handling (quantum_hrp.py lines 269-280)
# ===============================
# Drop tickers (columns) that are entirely NaN
prices = prices_raw.dropna(axis=1, how='all')
print(f"After dropna(axis=1, how='all'): {prices.shape}")

# Drop rows with all NaNs
prices = prices.dropna(how='all')
print(f"After dropna(how='all'): {prices.shape}")

# ===============================
# 3. Compute daily returns — author's exact approach
# ===============================
# NOTE: pct_change() on a DataFrame with NaN holes produces NaN returns
# at NaN→valid and valid→NaN transitions. Then dropna() (how='any')
# removes ALL rows where ANY ticker has a NaN return.
# With RIVN (first valid 2021-11-10), this truncates to ~1147 days.
asset_returns = prices.pct_change().dropna()

# Flatten column names (ensure simple strings)
asset_returns.columns = [c[0] if isinstance(c, tuple) else c for c in asset_returns.columns]

T, N = asset_returns.shape
print(f"\nReturns: {asset_returns.shape}  (T={T}, N={N})")
print(f"Date range: {asset_returns.index.min().date()} → {asset_returns.index.max().date()}")

asset_returns.to_csv("../data/returns.csv")
print("✓ returns.csv saved")

# ===============================
# 4. Features — author's compute_financial_features(returns, t, window=5)
# ===============================
P = 6
WINDOW = 5
asset_returns_np = asset_returns.to_numpy()  # shape (T, N)

features = np.zeros((N, T, P))

for i in range(N):
    ticker = asset_returns.columns[i]
    asset_series = asset_returns_np[:, i]

    if (i + 1) % 20 == 0 or i == 0 or i == N - 1:
        print(f"[{i+1:3d}/{N}] Computing features for {ticker}")

    for t in range(T):
        ret = asset_series[t]
        start = max(0, t - WINDOW + 1)
        window_returns = asset_series[start:t+1]

        f1 = ret                                        # current return
        f2 = np.mean(window_returns)                    # rolling mean
        f3 = np.std(window_returns, ddof=1)             # rolling std
        f4 = np.prod(1 + window_returns) - 1            # cumulative return
        f5 = stats.skew(window_returns)                 # skewness
        f6 = stats.kurtosis(window_returns)             # excess kurtosis

        features[i, t, :] = np.array([f1, f2, f3, f4, f5, f6])

# ===============================
# 5. Save
# ===============================
np.save("../data/features.npy", features)

# Also save the ticker list for reference
pd.Series(asset_returns.columns.tolist()).to_csv("../data/tickers.csv", index=False, header=["ticker"])

print(f"\n✓ features.npy saved — shape {features.shape}")
print(f"  NaN count in features: {np.isnan(features).sum()}")
print(f"  Inf count in features: {np.isinf(features).sum()}")
print(f"✓ tickers.csv saved — {N} tickers")

print(f"\n✓ features.npy: {features.shape}")
print(f"✓ Dimensions: (N_assets={N}, T={T}, P=6)")
print(f"✓ tickers.csv: {N} tickers saved")
