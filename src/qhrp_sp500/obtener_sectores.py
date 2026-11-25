#!/usr/bin/env python3
"""Sector de cada una de las 357 acciones (equivalente a data/asset_metadata.csv).

- Acciones que siguen cotizando: sector e industria de Yahoo Finance (misma
  clasificacion que asset_metadata.csv), buscando por el ticker del campo `ric`.
- Acciones deslistadas (`ric` con '^', nombre "DEAD - DELIST"): Yahoo no las
  tiene o el ticker lo usa ya otra empresa, asi que el sector se asigna a mano
  con la tabla SECTORES_MANUALES (columna fuente = "manual").

Salida: src/qhrp_sp500/datos/asset_metadata_357.csv (ya incluido en el repositorio; solo hace
falta volver a ejecutarlo para descargar de nuevo los sectores, y necesita yfinance e internet)
Uso: python obtener_sectores.py
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STOCKS_DIR = PROJECT_ROOT / "data" / "datos_raffinot" / "stocks_357_candidate" / "v1"
OUT_PATH = Path(__file__).resolve().parent / "datos" / "asset_metadata_357.csv"

# Sectores con los nombres de Yahoo para las acciones deslistadas
SECTORES_MANUALES = {
    "DD": "Basic Materials", "HES": "Energy", "APC": "Energy", "BCR": "Healthcare",
    "BBBYQ": "Consumer Cyclical", "AVB": "Real Estate", "CTRA": "Energy", "CELG": "Healthcare",
    "CTXS": "Technology", "CHKAQ": "Energy", "CERN": "Healthcare", "FTRCQ": "Communication Services",
    "CMA": "Financial Services", "CA": "Technology", "SEE": "Consumer Cyclical", "DOFSQ": "Energy",
    "DOW": "Basic Materials", "EMC": "Technology", "EA": "Communication Services", "FLIR": "Technology",
    "HAR": "Consumer Cyclical", "HOLX": "Healthcare", "IPG": "Communication Services", "JCI": "Industrials",
    "K": "Consumer Defensive", "LM": "Financial Services", "LEG": "Consumer Cyclical", "LLTC": "Technology",
    "SCG": "Utilities", "ATVI": "Communication Services", "KSU": "Industrials", "NFX": "Energy",
    "NBL": "Energy", "JWN": "Consumer Cyclical", "PDCO": "Healthcare", "AET": "Healthcare",
    "PX": "Basic Materials", "RTN": "Industrials", "STJ": "Healthcare", "SWN": "Energy",
    "SPLS": "Consumer Cyclical", "STI": "Financial Services", "ANDV": "Energy", "TIF": "Consumer Cyclical",
    "TSS": "Technology", "MRO": "Energy", "VAR": "Healthcare", "WFM": "Consumer Defensive",
    "FL": "Consumer Cyclical", "XLNX": "Technology", "VIAB": "Communication Services",
    "HOT": "Consumer Cyclical", "PBCT": "Financial Services", "GGP": "Real Estate",
    "TWX": "Communication Services", "ESRX": "Healthcare", "WBA": "Healthcare", "AGN": "Healthcare",
    "TGNA": "Communication Services",
    # Cotizan, pero Yahoo no devuelve sector con el ticker de `ric`
    "BFb": "Consumer Defensive", "FISV": "Technology",
}


def yahoo_ticker(ric: str) -> str:
    base = ric.split("^")[0]
    base = re.sub(r"\.(O|K|N|PK)$", "", base)
    return base.replace(".", "-")


def main() -> None:
    codes = pd.read_csv(STOCKS_DIR / "returns.csv", nrows=0).columns[2:]
    roster = pd.read_csv(STOCKS_DIR / "roster_audit.csv", dtype=str)
    roster = roster[roster["datastream_code"].isin(codes)][["datastream_code", "ric", "name"]]

    rows = []
    for code, ric, name in roster.itertuples(index=False):
        ticker = yahoo_ticker(ric)
        row = {"ticker": code, "ric": ric, "name": name, "yahoo_ticker": ticker}
        if "^" in ric:
            row.update(sector=SECTORES_MANUALES.get(ticker, "unknown"), industry="unknown", fuente="manual")
        else:
            try:
                info = yf.Ticker(ticker).info
                row.update(
                    sector=info.get("sector") or "unknown",
                    industry=info.get("industry") or "unknown",
                    yahoo_name=info.get("longName") or "",
                    fuente="yahoo",
                )
            except Exception as exc:  # ticker no encontrado, red, etc.
                row.update(sector="unknown", industry="unknown", fuente=f"error: {type(exc).__name__}")
            time.sleep(0.2)
            if row["sector"] == "unknown" and ticker in SECTORES_MANUALES:
                row.update(sector=SECTORES_MANUALES[ticker], fuente="manual")
        rows.append(row)

    meta = pd.DataFrame(rows)
    meta.to_csv(OUT_PATH, index=False)
    print(meta["fuente"].value_counts().to_string())
    print(meta["sector"].value_counts().to_string())
    print(f"\nGuardado en: {OUT_PATH}")


if __name__ == "__main__":
    main()
