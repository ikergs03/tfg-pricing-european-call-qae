import yfinance as yf
import pandas as pd
import time
import warnings
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# Crear carpeta 'data' si no existe
os.makedirs('data', exist_ok=True)

# --- Lista completa de tickers ---
tickers = [
    "SPY", "QQQ", "DIA", "IWM", "IVV", "VTI", "VOO",
    "EFA", "EEM", "VEA", "VWO", "ACWI", "SPDW", "SCHF",
    "XLK", "XLF", "XLE", "XLI", "XLY", "XLV", "XLC", "XLP", "XLRE", "XLU", "XLB",
    "VNQ", "IYR", "RWX", "SCHH", "FREL", "RWO",
    "BNDX", "VWOB", "IGOV", "IEMG", "EMB",
    "GLD", "SLV", "PPLT", "COPX", "DBB", "DBC", "PDBC", "USO", "UGA", "CORN", "WEAT", "SOYB",
    "TLT", "IEF", "LQD", "HYG", "SHY", "TIP", "AGG", "BND", "MBB", "SCHZ", "VCSH", "VCIT",
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD", "DOT-USD", "AVAX-USD",
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "BRK-B", "META", "NFLX", "DIS",
    "JPM", "GS", "BAC", "C", "WFC", "MS", "AXP", "BLK", "SCHW", "TFC",
    "BA", "CAT", "HON", "LMT", "UNP", "CSX", "NSC", "UPS", "FDX", "RTX",
    "CVX", "XOM", "OXY", "MPC", "PSX", "VLO", "SLB", "HAL", "BKR",
    "PG", "KO", "PEP", "MO", "PM", "WMT", "TGT", "HD", "LOW", "MCD", "SBUX", "NKE", "TJX",
    "JNJ", "PFE", "UNH", "CVS", "ABBV", "LLY", "MRNA", "REGN", "BMY", "GILD",
    "AMD", "INTC", "TSM", "ORCL", "IBM", "ADBE", "CRM", "SNOW", "PANW", "AVGO",
    "VZ", "T", "CMCSA", "CHTR", "TMUS", "WBD",
    "BABA", "TCEHY", "NIO", "INFY", "SHOP", "SE", "MELI", "BYDDF",
    "GM", "F", "RIVN", "LCID", "NKLA", "XPEV", "LI",
    "ICLN", "TAN", "PBW", "QCLN", "LIT", "FAN", "PBD", "KRBN"
]

start = "2005-01-01"
end = "2025-01-01"
all_data = {}
failed_tickers = []

print(f"Descargando {len(tickers)} tickers...")
print(f"Período: {start} a {end}\n")

for idx, t in enumerate(tickers, 1):
    max_retries = 3
    retry_delay = 3  # segundos entre intentos
    
    for attempt in range(max_retries):
        try:
            print(f"[{idx:3d}/{len(tickers)}] → {t:15s} (intento {attempt+1}/{max_retries})", end=" ... ")
            
            # Descargar datos CON parámetros mejorados
            df = yf.download(
                t, 
                start=start, 
                end=end, 
                progress=False,
                timeout=30  # Timeout explícito
            )
            
            if df is not None and not df.empty and len(df) > 1:
                # Asegurar que tenemos la columna correcta
                if isinstance(df, pd.DataFrame):
                    if "Adj Close" in df.columns:
                        all_data[t] = df["Adj Close"]
                    elif "Close" in df.columns:
                        all_data[t] = df["Close"]
                    else:
                        all_data[t] = df.iloc[:, 0]  # Primera columna como fallback
                else:
                    # Si es Series directamente
                    all_data[t] = df
                
                print(f"✔ ({len(df)} filas)")
                break
            else:
                raise ValueError("DataFrame vacío o con pocos datos")
                
        except Exception as e:
            error_msg = str(e)[:60]  # Primeros 60 caracteres del error
            
            if attempt < max_retries - 1:
                print(f"✗ Reintentando ({error_msg})")
                time.sleep(retry_delay)
            else:
                print(f"✗ FALLÓ")
                failed_tickers.append((t, error_msg))
    
    # Delay entre requests para evitar rate limiting
    time.sleep(1)

# Crear DataFrame unificado
print(f"\n{'='*60}")
print(f"Descarga completada: {len(all_data)}/{len(tickers)} tickers exitosos")
print(f"Fallos: {len(failed_tickers)}")

if failed_tickers:
    print(f"\n⚠ Tickers que fallaron:")
    for ticker, error in failed_tickers[:10]:  # Mostrar máximo 10
        print(f"  - {ticker}: {error}")
    if len(failed_tickers) > 10:
        print(f"  ... y {len(failed_tickers) - 10} más")

# Guardar solo los datos válidos
if all_data:
    try:
        # Concatenar Series con indices alineados
        price_df = pd.concat(all_data, axis=1)
        print(f"\nShape final: {price_df.shape}")
        print(f"Período de datos: {price_df.index.min()} a {price_df.index.max()}")
        print(f"Completitud: {(price_df.notna().sum().sum() / (price_df.shape[0] * price_df.shape[1]) * 100):.1f}%")
        
        # Guardar parquet en carpeta data/
        price_df.to_parquet("data/prices.parquet")
        print(f"\n✔ Archivo creado: data/prices.parquet")
        
        # Guardar también CSV para inspeccionar
        price_df.to_csv("data/prices.csv")
        print(f"✔ Archivo creado: data/prices.csv")
        
    except Exception as e:
        print(f"✗ Error al crear DataFrame: {e}")
        # Fallback: guardar como dict
        pd.Series(all_data).to_csv("data/prices_series.csv")
        print(f"✔ Archivo alternativo creado: data/prices_series.csv")
    
    # Guardar lista de tickers que fallaron
    if failed_tickers:
        with open("data/failed_tickers.txt", "w") as f:
            for ticker, error in failed_tickers:
                f.write(f"{ticker}: {error}\n")
        print(f"✔ Archivo creado: data/failed_tickers.txt")
else:
    print("\n✗ ERROR: No se descargó ningún dato válido")