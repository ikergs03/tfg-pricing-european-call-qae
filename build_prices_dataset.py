import pandas as pd
import pandas_datareader.data as web
import datetime as dt

# Lista de activos (acciones + ETFs, sin crypto, versión reducida)
TICKERS = [
    # Índices y grandes ETFs
    "SPY", "QQQ", "DIA", "IWM", "VOO", "IVV", "VTI",

    # Global / emergentes
    "EFA", "EEM", "VEA", "VWO", "ACWI",

    # Sectores (SPDRs)
    "XLK", "XLF", "XLE", "XLI", "XLY", "XLV", "XLC", "XLP", "XLRE", "XLU", "XLB",

    # Bonos
    "TLT", "IEF", "LQD", "HYG", "SHY", "AGG",

    # Materias primas
    "GLD", "SLV", "DBC", "USO",

    # Acciones grandes (FAANG + blue chips)
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "BAC", "DIS"
]

START = dt.datetime(2010, 1, 1)
END = dt.datetime(2024, 12, 31)


def main():
    all_prices = pd.DataFrame()

    for ticker in TICKERS:
        print(f"Descargando {ticker} desde Stooq...")
        try:
            df = web.DataReader(ticker, "stooq", START, END)
            # Stooq devuelve las fechas en orden descendente → las ponemos ascendentes
            df = df.sort_index()
            all_prices[ticker] = df["Close"]
        except Exception as e:
            print(f"Error con {ticker}: {e}")

    # Quitamos columnas completamente vacías
    all_prices = all_prices.dropna(axis=1, how="all")

    # Rellenamos huecos internos con forward-fill
    all_prices = all_prices.ffill().bfill()

    print("Shape final de la matriz de precios:", all_prices.shape)
    print("Tickers usados:", list(all_prices.columns))

    # Guardar en parquet dentro de la carpeta data/
    all_prices.to_parquet("data/prices.parquet")
    print("Guardado en data/prices.parquet")


if __name__ == "__main__":
    main()
