#!/usr/bin/env python3
"""PDF corto que explica, sin dar nada por sabido, las tablas de raffinot_tablas.py.

Salida: resultados_raffinot/explicacion_tablas.pdf
Lee los numeros de resultados_raffinot/{sectores_paso1,acciones_paso21}/metricas.csv.

Uso (desde src/qhrp_sp500): ../../raffinot-env/bin/python explicacion_pdf.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

HERE = Path(__file__).resolve().parent
RES = HERE / "resultados_raffinot"
OUT = RES / "explicacion_tablas.pdf"
SECT = pd.read_csv(RES / "sectores_paso1" / "metricas.csv", index_col=0)
ACC = pd.read_csv(RES / "acciones_paso21" / "metricas.csv", index_col=0)

BLUE = colors.HexColor("#2a6fb0")
GREY = colors.HexColor("#ecebe7")
BOX = colors.HexColor("#eef4fb")

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=18, textColor=BLUE, spaceAfter=10, spaceBefore=4)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=13.5, textColor=colors.black, spaceBefore=12, spaceAfter=5)
P = ParagraphStyle("P", parent=ss["BodyText"], fontSize=10.5, leading=15, alignment=TA_LEFT, spaceAfter=6)
SMALL = ParagraphStyle("S", parent=P, fontSize=9, leading=12, textColor=colors.HexColor("#555555"))
CELL = ParagraphStyle("C", parent=P, fontSize=9.5, leading=12.5, spaceAfter=0)
CELLB = ParagraphStyle("CB", parent=CELL, fontName="Helvetica-Bold")


def n(x: float, d: int = 3) -> str:
    """Numero con coma decimal."""
    return f"{x + 1e-9:.{d}f}".replace(".", ",")  # redondeo hacia arriba en el ,5, como en la tabla


def box(text: str) -> Table:
    t = Table([[Paragraph(text, P)]], colWidths=[16.5 * cm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BOX), ("BOX", (0, 0), (-1, -1), 0.5, BLUE),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    return t


def table(rows: list, widths: list, header: bool = True, shade_from: int | None = None,
          shade_rows: list | None = None) -> Table:
    data = [[Paragraph(str(c), CELLB if (header and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if header else 0)
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.black),
             ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#cccccc")),
             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if shade_from is not None:
        style.append(("BACKGROUND", (0, shade_from), (-1, -1), GREY))
    for r in shade_rows or []:
        style.append(("BACKGROUND", (0, r), (-1, r), GREY))
    t.setStyle(TableStyle(style))
    return t


def tree_figure(path: Path) -> None:
    """Dibujo del reparto 50/50 de Raffinot con 5 activos y 3 grupos (su Exhibit 1)."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.axis("off")
    seg = lambda x0, y0, x1, y1: ax.plot([x0, x1], [y0, y1], color="#333333", lw=1.6)
    # raiz
    seg(1.5, 3, 4.75, 3); seg(3.1, 3, 3.1, 3.4)
    ax.text(3.1, 3.5, "100 %", ha="center", fontsize=11, weight="bold")
    seg(1.5, 3, 1.5, 2); seg(4.75, 3, 4.75, 2.2)
    ax.text(1.5, 2.55, "50 %", ha="center", fontsize=10.5, color="#2a6fb0", weight="bold",
            bbox=dict(fc="white", ec="none"))
    ax.text(4.75, 2.65, "50 %", ha="center", fontsize=10.5, color="#2a6fb0", weight="bold",
            bbox=dict(fc="white", ec="none"))
    # grupo 1: activos 1 y 2
    seg(1, 2, 2, 2); seg(1, 2, 1, 0.6); seg(2, 2, 2, 0.6)
    # rama derecha: activo 5 | (3,4)
    seg(4, 2.2, 5.5, 2.2); seg(4, 2.2, 4, 0.6); seg(5.5, 2.2, 5.5, 1.6)
    ax.text(4, 1.45, "25 %", ha="center", fontsize=10, color="#2a6fb0", weight="bold", bbox=dict(fc="white", ec="none"))
    ax.text(5.5, 1.85, "25 %", ha="center", fontsize=10, color="#2a6fb0", weight="bold", bbox=dict(fc="white", ec="none"))
    seg(5, 1.6, 6, 1.6); seg(5, 1.6, 5, 0.6); seg(6, 1.6, 6, 0.6)
    for x, name, w in ((1, "Activo 1", "25 %"), (2, "Activo 2", "25 %"), (4, "Activo 5", "25 %"),
                       (5, "Activo 3", "12,5 %"), (6, "Activo 4", "12,5 %")):
        ax.text(x, 0.35, name, ha="center", fontsize=9.5)
        ax.text(x, 0.0, w, ha="center", fontsize=10, weight="bold")
    for (x0, x1, c, lab) in ((0.6, 2.4, "#2a6fb0", "Grupo A"), (3.6, 4.4, "#d9822b", "Grupo B"),
                             (4.6, 6.4, "#2e9e6b", "Grupo C")):
        ax.add_patch(plt.Rectangle((x0, -0.25), x1 - x0, 1.15, fill=False, ec=c, lw=1.6, ls="--"))
        ax.text((x0 + x1) / 2, -0.55, lab, ha="center", fontsize=9.5, color=c, weight="bold")
    ax.set_xlim(0.3, 6.8)
    ax.set_ylim(-0.75, 3.8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def ranking() -> pd.DataFrame:
    """Puesto por panel = media de los puestos en ASR, CEQ y MDD; puesto final = media de los dos paneles."""
    pos = {}
    for name, m in (("sectores", SECT), ("acciones", ACC)):
        score = pd.concat([m.ASR.rank(ascending=False), m.CEQ.rank(ascending=False), m.MDD.rank()], axis=1).mean(axis=1)
        pos[name] = score.rank(method="min")
    df = pd.DataFrame(pos)
    df["final"] = df.mean(axis=1).rank(method="min")
    return df.sort_values(["final", "sectores"])


def scaled_image(path: Path, width_cm: float) -> Image:
    img = Image(str(path))
    ratio = img.imageHeight / img.imageWidth
    img.drawWidth = width_cm * cm
    img.drawHeight = width_cm * cm * ratio
    return img


def build() -> None:
    tree_png = RES / "_arbol_ejemplo.png"
    tree_figure(tree_png)
    ew, mv = SECT.loc["EW"], SECT.loc["MV"]
    s = []

    # Portada e idea general
    s += [Paragraph("Las tablas de Raffinot y el QHRP, explicadas", H1),
          box("<b>En una frase:</b> comparamos muchas formas de repartir dinero entre los 10 sectores y entre 357 "
              "acciones del S&amp;P 500, para ver cuál habría ido mejor entre 1996 y 2016. Añadimos tres formas "
              "que usan el QHRP del TFG para ver si están a la altura."),
          Spacer(1, 6),
          Paragraph("<b>Cómo se calcula:</b> nos ponemos en un día del pasado, miramos solo el último año, cada "
                    "método decide cuánto dinero pone en cada activo y vemos cuánto gana después. Luego avanzamos "
                    "un día (sectores) o un mes (acciones) y repetimos hasta 2016.", P),
          Paragraph("<b>Cómo se lee la tabla:</b> como un boletín de notas. Cada <b>fila</b> es un método (una forma "
                    "de repartir) y cada <b>columna</b> una nota. Las filas en gris son las del QHRP, añadidas por "
                    "nosotros.", P)]

    # Columnas
    s += [Paragraph("1. Las columnas", H2),
          table([["Columna", "Qué es", "Mejor si...", "Ejemplo (sectores)"],
                 ["<b>ASR</b>", "Cuánto gana, castigando los sustos. Es la nota principal. Un solo día raro "
                                "(+25 % o -15 %) puede bajarla mucho.", "alto",
                  f"Repartir a partes iguales: {n(ew.ASR)}"],
                 ["<b>CEQ</b>", "Lo que de verdad ganas al año, quitando el susto. Un 7 es como ganar un 7 % al "
                                "año sin riesgo.", "alto", f"Partes iguales: {n(ew.CEQ, 2)} %"],
                 ["<b>MDD</b>", "Lo máximo que llegaste a perder (casi siempre en la crisis de 2008).", "bajo",
                  f"Partes iguales perdió el {n(ew.MDD, 1)} %; MV solo el {n(mv.MDD, 1)} %"],
                 ["<b>TO</b>", "Cuánto dinero se mueve en cada ajuste de la cartera. Más movimiento = más "
                               "comisiones.", "bajo", f"MV mueve el {n(mv.TO, 1)} % cada día"],
                 ["<b>SSPW</b>", "Si el dinero está muy junto o muy repartido (huevos en pocas o muchas cestas). "
                                 "Lo más repartido posible: 0,100 con 10 sectores.", "bajo",
                  f"Partes iguales: {n(ew.SSPW)}; MV: {n(mv.SSPW)}"]],
                [2.3, 7.7, 1.9, 4.6]),
          Spacer(1, 6),
          Paragraph("<b>Las letras pequeñas</b> junto a ASR y CEQ vienen de una prueba estadística que dice si "
                    "las diferencias son reales o pueden ser suerte: <super>b</super> = entre los mejores con mucha "
                    "seguridad; <super>a</super> = entre los mejores con menos seguridad; sin letra = peor que el "
                    "mejor.", P)]

    # Metodos
    s += [Paragraph("2. Las filas (los métodos)", H2),
          table([["Método", "Qué hace"],
                 ["EW", "Lo mismo en todos. Es la referencia."],
                 ["MV", "La cartera que menos sube y baja. Cae poco, pero mete casi todo en pocos activos."],
                 ["MDP", "La cartera más diversificada: busca activos que no se mueven a la vez."],
                 ["ERC", "Cada activo aporta el mismo riesgo: menos dinero a los activos movidos."],
                 ["IVRB", "Como ERC, pero dando aún más peso a los activos tranquilos."]],
                [2.2, 14.3]),
          Spacer(1, 8),
          Paragraph("<b>SL, CL, AL, WM y DBHT</b> son el método de Raffinot. Primero juntan en un árbol los "
                    "activos que se parecen. Después reparten el dinero por el árbol: mitad para cada rama hasta "
                    "llegar a los grupos, y a partes iguales dentro de cada grupo. Así 40 bancos parecidos no se "
                    "llevan 40 veces más dinero que una eléctrica solitaria.", P),
          scaled_image(tree_png, 10.5),
          Paragraph("Todos reparten el dinero igual. Lo único que cambia es <b>cómo construyen el árbol</b>, es decir, "
                    "cómo deciden si dos grupos se parecen lo bastante para juntarlos:", P),
          table([["Método", "Cómo decide si junta dos grupos", "Ejemplo: ¿junta {banco A, banco B} con "
                                                                "{banco C, petrolera}?", "Qué grupos salen"],
                 ["SL<br/>(simple)", "Mira los dos activos <b>más parecidos</b> entre los dos grupos.",
                  "Sí: banco A y banco C se parecen mucho, y con eso basta.",
                  "Cadenas largas y algún grupo enorme. Se deja llevar por un solo parecido."],
                 ["CL<br/>(completo)", "Mira los dos activos <b>más distintos</b> entre los dos grupos.",
                  "No: banco A y la petrolera son muy distintos.",
                  "Grupos pequeños y muy parecidos por dentro."],
                 ["AL<br/>(medio)", "Hace la <b>media</b> de lo que se parecen todos con todos.",
                  "Depende: hay dos parejas banco-banco (se parecen) y dos banco-petrolera (no se parecen); "
                  "decide la media.",
                  "Término medio. Según el artículo, el más fiable."],
                 ["WM<br/>(Ward)", "Junta los dos grupos que, al unirse, quedan <b>menos mezclados</b>.",
                  "Probablemente no: la petrolera mezclaría el grupo de bancos.",
                  "Grupos compactos y de tamaño parecido. Es el que se usaba en el TFG."],
                 ["DBHT", "Dibuja una <b>red</b> uniendo los activos más relacionados y busca zonas muy "
                          "conectadas.", "Lo decide la red: la petrolera estaría en la zona de las energéticas.",
                  "Decide él solo cuántos grupos hay."]],
                [2.1, 4.6, 5.2, 4.6])]

    # QHRP
    s += [Paragraph("3. Las filas en gris (el QHRP)", H2),
          Paragraph("Usan el mismo reparto de Raffinot, pero el árbol se construye con el QHRP en vez de con la "
                    "correlación.", P),
          table([["Parte del nombre", "Qué significa"],
                 ["QHRP", "El método cuántico del TFG: los datos de cada activo pasan por un circuito de 6 qubits "
                          "y con eso se mide cuánto se parecen dos activos."],
                 ["original", "Tal cual estaba en el TFG."],
                 ["nuevo", "Con dos cambios que hicimos para arreglar un fallo (abajo)."],
                 ["WM / AL", "Tipo de árbol: Ward (el del TFG) o enlace medio (el más fiable según el artículo)."]],
                [3.2, 13.3]),
          Spacer(1, 8),
          Paragraph("<b>Por qué un QHRP nuevo:</b> el original hace la media de todos los días de cada activo antes "
                    "de compararlos, así que no ve si dos activos suben y bajan a la vez. Por eso no encontraba "
                    "los sectores: solo el 17 % de las acciones quedaba junto a otra de su sector (el método "
                    "clásico, el 61 %). El nuevo compara los activos <b>el mismo día</b> y escala los datos con un "
                    "ranking para que los días rarísimos no lo deformen. Llega al <b>79 %</b>.", P),
          table([["", "TFG", "Esta prueba"],
                 ["Datos", "169 activos variados (acciones, bonos, cripto...), 2021-2024",
                  "10 sectores y 357 acciones del S&amp;P 500, 1996-2016"],
                 ["Reparto del dinero", "HRP: cortar la lista por la mitad", "Raffinot: mitad por cada rama del árbol"],
                 ["QHRP", "Solo el original", "Original y nuevo"]],
                [3.4, 6.5, 6.6])]

    # Resultados y conclusiones
    w, qo, qn, qa, al = (ACC.loc[m] for m in ("WM", "QHRP original WM", "QHRP nuevo WM", "QHRP nuevo AL", "AL"))
    ws, qos, qns, qas, als = (SECT.loc[m] for m in ("WM", "QHRP original WM", "QHRP nuevo WM", "QHRP nuevo AL", "AL"))
    s += [KeepTogether([Paragraph("4. Qué dicen las tablas", H2),
          Paragraph("Cada QHRP se compara con el clásico del mismo árbol: los WM con <b>WM</b> y el AL con "
                    "<b>AL</b>.", P),
          table([["", "Sectores: ASR", "Acciones: ASR", "Acciones: CEQ", "Acciones: caída máx."],
                 ["WM clásico", n(ws.ASR), n(w.ASR), n(w.CEQ, 2), f"{n(w.MDD, 1)} %"],
                 ["QHRP original WM", n(qos.ASR), n(qo.ASR), n(qo.CEQ, 2), f"{n(qo.MDD, 1)} %"],
                 ["QHRP nuevo WM", n(qns.ASR), n(qn.ASR), n(qn.CEQ, 2), f"{n(qn.MDD, 1)} %"],
                 ["AL clásico", n(als.ASR), n(al.ASR), n(al.CEQ, 2), f"{n(al.MDD, 1)} %"],
                 ["QHRP nuevo AL", n(qas.ASR), n(qa.ASR), n(qa.CEQ, 2), f"{n(qa.MDD, 1)} %"]],
                [3.6, 2.9, 2.9, 2.9, 4.2], shade_from=None),
          Spacer(1, 4),
          Paragraph("El QHRP nuevo AL tiene un ASR muy bajo en acciones porque algunos meses mete mucho dinero en "
                    "una sola acción, y un día de +27 % le hunde la nota.", SMALL)]),
          Paragraph("<b>Conclusiones</b>", P),
          table([["1", "La réplica está bien hecha: EW, MDP, ERC e IVRB salen prácticamente iguales que en el "
                       "artículo."],
                 ["2", "En sectores, el QHRP nuevo rinde igual que su versión clásica."],
                 ["3", "En acciones, con Ward, los dos QHRP mejoran al Ward clásico, pero los mejores siguen "
                       "siendo AL y SL clásicos."],
                 ["4", "Con 6 qubits no hay ventaja cuántica, y el QHRP es más lento que los métodos clásicos."]],
                [1, 15.5], header=False)]

    # Ranking
    rank = ranking()
    rows = [["Puesto", "Método", "Puesto en sectores", "Puesto en acciones"]]
    rows += [[str(int(r.final)), f"<b>{m}</b>" if m.startswith("QHRP") else m, str(int(r.sectores)),
              str(int(r.acciones))] for m, r in rank.iterrows()]
    s += [KeepTogether([
        Paragraph("5. Ranking: ¿cuál es mejor?", H2),
        Paragraph("Para cada panel ordenamos los 13 métodos por <b>ASR</b> (cuánto gana con los sustos), <b>CEQ</b> "
                  "(cuánto gana de verdad) y <b>caída máxima</b> (cuánto puede perder), y hacemos la media de los "
                  "tres puestos. El puesto final es la media de los dos paneles. TO y SSPW no cuentan: dicen cómo es "
                  "la cartera, no si es mejor.", P),
        table(rows, [2, 6, 4.2, 4.3], shade_rows=[i + 1 for i, m in enumerate(rank.index) if m.startswith("QHRP")]),
        Spacer(1, 6),
        box("<b>Cómo leerlo:</b> el <b>AL clásico</b> es el más regular: bueno en los dos paneles. El <b>QHRP nuevo "
            "AL</b> queda segundo sobre todo porque es de los que menos pierden en las crisis, aunque en acciones "
            "tiene el peor ASR. <b>Cuidado:</b> la prueba estadística dice que muchas de estas diferencias podrían "
            "ser suerte, sobre todo en sectores, así que un puesto arriba o abajo no significa mucho. Lo que sí es "
            "claro es quién está arriba (AL, QHRP nuevo AL, SL) y quién abajo (WM, QHRP nuevo WM, CL).")])]

    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="Las tablas de Raffinot y el QHRP, explicadas", author="Iker González Sánchez")
    doc.build(s)
    tree_png.unlink()
    print(OUT)


if __name__ == "__main__":
    build()
