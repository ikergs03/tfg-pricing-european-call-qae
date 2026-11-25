# QHRP con los datos S&P 500 de Raffinot

Réplica de las tablas 2 y 4 de Raffinot (2018), *Hierarchical Clustering-Based Asset Allocation*, añadiendo el QHRP del TFG (original y una versión mejorada) como forma de agrupar los activos. También incluye el análisis de grupos por sector y un backtest HRP como el del TFG.

El resumen de resultados está en `resultados_raffinot/conclusiones.md`, y una explicación de cada número de las tablas en `resultados_raffinot/explicacion_tablas.pdf`.

## 1. Preparar el entorno

Con Python 3.11:

```
python3.11 -m venv raffinot-env
raffinot-env/bin/pip install -r src/qhrp_sp500/requirements.txt
```

Todos los scripts se ejecutan desde `src/qhrp_sp500/` con ese entorno.

## 2. Poner los datos

Los datos de LSEG **no están en el repositorio** (licencia). Hay que copiarlos en la raíz del repositorio con esta estructura:

```
data/datos_raffinot/
├── sectors_10/v1/returns.csv
└── stocks_357_candidate/v1/
    ├── returns.csv
    └── roster_audit.csv
```

El sector de cada una de las 357 acciones sí está incluido: `datos/asset_metadata_357.csv`. Se sacó de Yahoo Finance; las 61 empresas que ya no cotizan llevan el sector puesto a mano (columna `fuente = manual`).

## 3. Qué ejecutar

Los tiempos son aproximados, con 7 procesos en un portátil de 16 núcleos.

| Para obtener... | Script | Tiempo | Resultados en |
|---|---|---|---|
| **Tablas 2 y 4 de Raffinot con los QHRP** | `raffinot_tablas.py --panel sectores` | ~25 min | `resultados_raffinot/sectores_paso1/` |
| | `raffinot_tablas.py --panel acciones --paso 21`, y después `raffinot_tablas.py --panel acciones --paso 21 --k-fijo "WM,QHRP original WM:2"` | ~90 min + 10 min | `resultados_raffinot/acciones_paso21/` |
| Probar todos los números de grupos fijos frente al artículo | `probar_k_raffinot.py --panel sectores` / `--panel acciones --paso 21 --sin-dbht` | ~15 min / ~5 min | `resultados_raffinot/probar_k/` |
| Leyenda y PDF explicativo de las tablas | `leyenda_raffinot.py`, `explicacion_pdf.py` | segundos | `resultados_raffinot/` |
| Figuras 3 y 4 del TFG con estos datos, grupos y pureza por sector | `analisis_sp500.py --panel sectores` / `--panel acciones` | ~1 min / ~5 min | `resultados/` |
| QHRP nuevo: grupos por sector | `qhrp_dia_a_dia.py --panel acciones --escala percentil` | ~5 min | `resultados_dia_a_dia/` |
| Backtest HRP como en el TFG (756/21 días) | `backtest_sp500.py --panel sectores` / `--panel acciones --workers 7` | ~3 min / ~50 min | `resultados_backtest/` |
| Reparto de Raffinot con número de grupos fijo y prueba MCS | `hcaa_sp500.py --panel sectores --paso 1 --k-fijos 2,3,4,5`, después `mcs_hcaa.py <carpeta>` | ~20 min | `resultados_hcaa/` |
| Imágenes de la presentación anterior | `crear_presentacion.py` (necesita los resultados de los demás) | segundos | `presentacion/` |

Las tablas en formato del artículo son `tabla_articulo.png` y `tabla_articulo.csv`. `metricas.csv` tiene además el Sharpe, los p-valores de la prueba MCS y el número medio de grupos.

## 4. Qué hace cada script

| Script | Qué hace |
|---|---|
| `qhrp_core.py` | El QHRP del TFG (6 features con ventana de 5 días, circuito de 6 qubits, matrices de densidad, Frobenius, Ward), simulado con numpy. `verificar_contra_qiskit()` comprueba que coincide con Qiskit. |
| `qhrp_dia_a_dia.py` | QHRP nuevo: compara los activos el mismo día y escala por percentiles. |
| `raffinot_tablas.py` | Las tablas del artículo: EW, MV, MDP, ERC, IVRB, SL, CL, AL, WM, DBHT y los tres QHRP, con ASR, CEQ, MDD, TO, SSPW y la prueba MCS. |
| `hcaa_sp500.py`, `mcs_hcaa.py` | Reparto de Raffinot (HCAA), índice Gap, métricas y prueba MCS. Los usa `raffinot_tablas.py`. |
| `probar_k_raffinot.py` | Prueba números de grupos fijos para SL, CL, AL, WM y DBHT frente al artículo. |
| `analisis_sp500.py` | Figuras 3 y 4 del TFG, grupos, pureza y rachas de sector. |
| `backtest_sp500.py` | Backtest HRP walk-forward como el del TFG. |
| `obtener_sectores.py` | Vuelve a descargar los sectores de las acciones (necesita `yfinance` e internet). |
| `leyenda_raffinot.py`, `explicacion_pdf.py`, `crear_presentacion.py` | Imágenes y PDF explicativos. |

## 5. Diferencias con el artículo

- **Número de grupos:** índice Gap con un mínimo de 2 grupos. En acciones, los métodos con Ward usan 2 grupos fijos, que es lo que más se acerca al artículo. SL y AL no se reproducen con ningún número fijo (`probar_k/`).
- **Acciones:** recalculadas cada 21 días en vez de cada día, porque el DBHT tarda ~2 min por fecha con 357 acciones.
- **DBHT:** implementación de Riskfolio-Lib.
- **IVRB:** presupuestos proporcionales a 1/volatilidad, que es lo que reproduce los números del artículo; su fórmula impresa usa la varianza.
