import yfinance as yf
import pandas as pd
import numpy as np
import time
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

os.makedirs('../data', exist_ok=True)

# ==========================================
# TICKERS - 170 activos (PAPER)
# ==========================================
tickers = [
    'SPY', 'QQQ', 'DIA', 'IWM', 'IVV', 'VTI', 'VOO', 'EFA', 'EEM', 'VEA',
    'VWO', 'ACWI', 'SPDW', 'SCHF', 'XLK', 'XLF', 'XLE', 'XLI', 'XLY', 'XLV',
    'XLC', 'XLP', 'XLRE', 'XLU', 'XLB', 'VNQ', 'IYR', 'RWX', 'SCHH', 'FREL',
    'RWO', 'BNDX', 'VWOB', 'IGOV', 'IEMG', 'EMB', 'GLD', 'SLV', 'PPLT', 'COPX',
    'DBB', 'DBC', 'PDBC', 'USO', 'UGA', 'CORN', 'WEAT', 'SOYB', 'TLT', 'IEF',
    'LQD', 'HYG', 'SHY', 'TIP', 'AGG', 'BND', 'MBB', 'SCHZ', 'VCSH', 'VCIT',
    'BTC-USD', 'ETH-USD', 'SOL-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD',
    'DOT-USD', 'AVAX-USD', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'BRK-B',
    'META', 'NFLX', 'DIS', 'JPM', 'GS', 'BAC', 'C', 'WFC', 'MS', 'AXP', 'BLK',
    'SCHW', 'TFC', 'BA', 'CAT', 'HON', 'LMT', 'UNP', 'CSX', 'NSC', 'UPS',
    'FDX', 'RTX', 'CVX', 'XOM', 'OXY', 'MPC', 'PSX', 'VLO', 'SLB', 'HAL',
    'BKR', 'PG', 'KO', 'PEP', 'MO', 'PM', 'WMT', 'TGT', 'HD', 'LOW',
    'MCD', 'SBUX', 'NKE', 'TJX', 'JNJ', 'PFE', 'UNH', 'CVS', 'ABBV', 'LLY',
    'MRNA', 'REGN', 'BMY', 'GILD', 'AMD', 'INTC', 'TSM', 'ORCL', 'IBM', 'ADBE',
    'CRM', 'SNOW', 'PANW', 'AVGO', 'VZ', 'T', 'CMCSA', 'CHTR', 'TMUS', 'WBD',
    'BABA', 'TCEHY', 'NIO', 'INFY', 'SHOP', 'SE', 'MELI', 'BYDDF', 'GM', 'F',
    'RIVN', 'LCID', 'NKLA', 'XPEV', 'LI', 'ICLN', 'TAN', 'PBW', 'QCLN', 'LIT',
    'FAN', 'PBD', 'KRBN'
]

# ==========================================
# PARÁMETROS CRÍTICOS (PAPER)
# ==========================================
start_date = '2005-01-01'  # Desde 2005
end_date = '2025-01-01'     # Hasta 2025
# VENTANA WALK-FORWARD:
# - Entrenamiento: 756 días (aprox. 3 años)
# - Test: 21 días (aprox. 1 mes)

print(f"Descargando datos de {len(tickers)} tickers...")
print(f"Período: {start_date} a {end_date}")

all_data = {}
all_volume = {}
failed_tickers = []

for idx, ticker in enumerate(tickers, 1):
    max_retries = 3
    retry_delay = 3
    
    for attempt in range(max_retries):
        try:
            print(f"[{idx:3d}/{len(tickers)}] Descargando {ticker:12s} (intento {attempt+1}/{max_retries})", end='', flush=True)
            
            # Descargar con límite de tiempo
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                timeout=30
            )
            
            if df is not None and not df.empty and len(df) > 1:
                if isinstance(df, pd.DataFrame):
                    if 'Close' in df.columns:
                        price_series = df['Close']
                    elif 'Adj Close' in df.columns:
                        price_series = df['Adj Close']
                    else:
                        price_series = df.iloc[:, 0]

                    volume_series = df['Volume'] if 'Volume' in df.columns else pd.Series(index=price_series.index, dtype=float)

                    all_data[ticker] = price_series
                    all_volume[ticker] = volume_series
                else:
                    all_data[ticker] = df
                    all_volume[ticker] = pd.Series(index=df.index, dtype=float)
                
                print(" ✓")
                break
            else:
                raise ValueError("DataFrame vacío o insuficiente")
                
        except Exception as e:
            error_msg = str(e)[:60]
            if attempt < max_retries - 1:
                print(f" ✗ Reintentando... ({error_msg})")
                time.sleep(retry_delay)
            else:
                print(f" ✗ FALLO")
                failed_tickers.append((ticker, error_msg))
    
    time.sleep(1)

print(f"\n✓ Descarga completada: {len(all_data)}/{len(tickers)} tickers exitosos")

if failed_tickers:
    print(f"\n⚠ Fallos ({len(failed_tickers)}):")
    for ticker, error in failed_tickers[:10]:
        print(f"  - {ticker}: {error}")
    if len(failed_tickers) > 10:
        print(f"  ... y {len(failed_tickers) - 10} más")

# ==========================================
# GUARDAR DATOS
# ==========================================
if all_data:
    try:
        price_df = pd.concat(all_data, axis=1)
        volume_df = pd.concat(all_volume, axis=1)
        
        print(f"\n📊 Datos consolidados: {price_df.shape}")
        print(f"Período: {price_df.index.min()} a {price_df.index.max()}")
        print(f"Completitud: {(price_df.notna().sum().sum() / (price_df.shape[0] * price_df.shape[1]) * 100):.1f}%")
        
        # Guardar como Parquet
        price_df.to_parquet('../data/prices.parquet')
        print("✓ Archivo guardado: ../data/prices.parquet")

        volume_df.to_parquet('../data/volumes.parquet')
        print("✓ Archivo guardado: ../data/volumes.parquet")
        
        # Guardar como CSV
        price_df.to_csv('../data/prices.csv')
        print("✓ Archivo guardado: ../data/prices.csv")

        volume_df.to_csv('../data/volumes.csv')
        print("✓ Archivo guardado: ../data/volumes.csv")
        
    except Exception as e:
        print(f"✗ Error al guardar: {e}")

if failed_tickers:
    with open('../data/failed_tickers.txt', 'w') as f:
        for ticker, error in failed_tickers:
            f.write(f"{ticker}: {error}\n")
    print("✓ Archivo guardado: ../data/failed_tickers.txt")