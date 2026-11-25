#!/usr/bin/env python3
"""Imagen-leyenda de las tablas de raffinot_tablas.py: qué mide cada columna y qué es cada fila.

Salida: resultados_raffinot/leyenda_tablas.png
Uso: python leyenda_raffinot.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "resultados_raffinot" / "leyenda_tablas.png"

COLUMNS = [
    ("ASR", "Sharpe ajustado", "↑ mejor",
     "Rentabilidad por unidad de riesgo (rentabilidad media anual / volatilidad anual), "
     "penalizada si hay más caídas fuertes que subidas fuertes o muchos días extremos.",
     "0,42 = gana 0,42 % al año por cada 1 % de volatilidad."),
    ("CEQ", "Rentabilidad equivalente cierta (%)", "↑ mejor",
     "Rentabilidad anual que un inversor aceptaría a cambio de no correr riesgo: "
     "rentabilidad media anual menos la mitad de la varianza.",
     "6,80 = equivale a ganar un 6,8 % al año sin riesgo."),
    ("MDD", "Caída máxima (%)", "↓ mejor",
     "Mayor pérdida desde un máximo hasta el mínimo siguiente: el peor momento de la cartera.",
     "54,2 = en el peor momento (2008-09) llegó a perder el 54,2 %."),
    ("TO", "Rotación (%)", "↓ mejor",
     "Cuánto cambia la cartera en cada rebalanceo (suma de los cambios de pesos). "
     "Más rotación = más compras y ventas = más costes.",
     "1,8 = en cada rebalanceo se compra y vende el 1,8 % de la cartera."),
    ("SSPW", "Concentración", "↓ más diversificada",
     "Suma de los pesos al cuadrado. Va de 1/N (todo repartido por igual) a 1 (todo en un solo activo).",
     "0,100 con 10 sectores = repartido por igual; 0,5 = casi todo en 2 activos."),
]

MARKS = [
    ("0.422ᵇ", "Entre los mejores con mucha seguridad: la prueba MCS no lo distingue del mejor método (p ≥ 0,70)."),
    ("0.422ᵃ", "Entre los mejores, con menos seguridad (0,20 ≤ p < 0,70)."),
    ("0.422", "Peor que el mejor, y no por casualidad (p < 0,20)."),
]

ROWS = [
    ("Referencia", [("EW", "Pesos iguales: el mismo dinero en cada activo.")]),
    ("Presupuesto de riesgo\n(usan la covarianza)", [
        ("MV", "Mínima varianza: la cartera con menos volatilidad."),
        ("MDP", "Cartera más diversificada (Choueifaty): máxima diversificación por correlaciones."),
        ("ERC", "Igual contribución al riesgo: cada activo aporta el mismo riesgo."),
        ("IVRB", "Presupuesto de riesgo proporcional a 1 / volatilidad de cada activo."),
    ]),
    ("Agrupación jerárquica\n(reparto de Raffinot, HCAA)", [
        ("SL", "Árbol por correlación, enlace simple: une grupos por sus dos activos más cercanos."),
        ("CL", "Enlace completo: une grupos por sus dos activos más lejanos."),
        ("AL", "Enlace medio: une grupos por la distancia media entre sus activos."),
        ("WM", "Ward: une los grupos que menos aumentan la dispersión interna."),
        ("DBHT", "Árbol a partir de una red de correlaciones (grafo planar); elige él solo los grupos."),
    ]),
    ("Cuánticos (añadidos,\nen gris en la tabla)", [
        ("QHRP original WM", "Distancia cuántica del TFG (matrices de densidad promedio) + Ward."),
        ("QHRP nuevo WM", "Distancia cuántica día a día con escala por percentiles + Ward."),
        ("QHRP nuevo AL", "La misma distancia cuántica nueva + enlace medio."),
    ]),
]

HOW = ("Cómo se lee: cada fila es una forma de repartir el dinero. Con los 252 días anteriores se calcula la "
       "cartera y se mide lo que gana después. En los métodos de agrupación el dinero se reparte 50 % a cada "
       "rama del árbol hasta llegar a los grupos, y a partes iguales dentro de cada grupo. Para valorar el "
       "QHRP, compara cada QHRP con el clásico del mismo enlace: QHRP original WM y QHRP nuevo WM frente a WM; "
       "QHRP nuevo AL frente a AL.")


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width))


def main() -> None:
    fig = plt.figure(figsize=(14, 15))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ink, muted, accent = "#1a1a1a", "#555555", "#2a6fb0"

    y = 0.975
    ax.text(0.04, y, "Leyenda de las tablas: qué significa cada número", fontsize=19, weight="bold", color=ink, va="top")
    y -= 0.045

    # Columnas
    ax.text(0.04, y, "COLUMNAS", fontsize=12, weight="bold", color=accent, va="top")
    y -= 0.028
    for short, name, better, what, example in COLUMNS:
        ax.text(0.04, y, short, fontsize=15, weight="bold", style="italic", color=ink, va="top")
        ax.text(0.13, y, name, fontsize=12.5, weight="bold", color=ink, va="top")
        ax.text(0.96, y, better, fontsize=12, weight="bold", color=accent, va="top", ha="right")
        body = wrap(what, 95)
        ax.text(0.13, y - 0.022, body, fontsize=11, color=ink, va="top", linespacing=1.35)
        lines = body.count("\n") + 1
        ax.text(0.13, y - 0.022 - 0.0165 * lines, "Ejemplo: " + example, fontsize=10.5, color=muted,
                va="top", style="italic")
        y -= 0.022 + 0.0165 * lines + 0.033
    y -= 0.005

    # Marcas
    ax.text(0.04, y, "MARCAS a Y b JUNTO A ASR Y CEQ (prueba estadística MCS del artículo)", fontsize=12,
            weight="bold", color=accent, va="top")
    y -= 0.028
    for mark, text in MARKS:
        ax.text(0.06, y, mark, fontsize=13, color=ink, va="top")
        ax.text(0.17, y, text, fontsize=11, color=ink, va="top")
        y -= 0.024
    y -= 0.018

    # Filas
    ax.text(0.04, y, "FILAS (métodos)", fontsize=12, weight="bold", color=accent, va="top")
    y -= 0.028
    step = 0.021
    for group, items in ROWS:
        top = y
        height = max(step * len(items), 0.04)
        if group.startswith("Cuánticos"):
            ax.add_patch(plt.Rectangle((0.03, top - height - 0.014), 0.94, height + 0.02,
                                       facecolor="#e4e2dc", edgecolor="none", zorder=0))
        ax.text(0.04, y, group, fontsize=11, weight="bold", color=muted, va="top", linespacing=1.3)
        for short, text in items:
            ax.text(0.27, y, short, fontsize=11, weight="bold", color=ink, va="top")
            ax.text(0.45, y, text, fontsize=10.5, color=ink, va="top")
            y -= step
        y = top - height - 0.016
    y -= 0.008

    ax.plot([0.04, 0.96], [y + 0.01, y + 0.01], color="#cccccc", lw=1)
    ax.text(0.04, y - 0.005, wrap(HOW, 118), fontsize=10.5, color=muted, va="top", linespacing=1.4)

    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUT)


if __name__ == "__main__":
    main()
