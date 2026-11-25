# Réplica de Raffinot (2018) con QHRP: conclusiones

**En resumen:** la réplica de las tablas 2 y 4 del artículo es fiable, porque los métodos que no dependen del número de grupos salen prácticamente iguales que en el artículo. Añadiendo el QHRP: en sectores rinde igual que su versión clásica, y en acciones el QHRP con Ward mejora al Ward clásico, aunque los mejores siguen siendo los métodos clásicos SL y AL.

Tablas: `sectores_paso1/tabla_articulo.png` y `acciones_paso21/tabla_articulo.png`. Qué significa cada número: `leyenda_tablas.png`.

## 1. Qué se ha hecho

- Mismos datos que el artículo: 10 sectores del S&P 500 (1995–2016) y 357 acciones del S&P 500 (1996–2016).
- Mismo método: con los 252 días anteriores se calcula la cartera y se mide lo que gana después.
- Mismas filas: EW, MV, MDP, ERC, IVRB, SL, CL, AL, WM y DBHT.
- Mismas columnas: ASR, CEQ, MDD, TO y SSPW, con la prueba estadística MCS.
- Añadidas al final tres filas cuánticas:
  - **QHRP original WM:** la distancia cuántica del TFG con Ward.
  - **QHRP nuevo WM y QHRP nuevo AL:** la versión mejorada del QHRP, que compara los activos día a día y escala por percentiles, con Ward y con enlace medio.

## 2. La réplica es fiable

Los métodos que no dependen de cuántos grupos se hagan salen igual que en el artículo:

| Panel | Método | ASR (nuestro / artículo) | Caída máx. % (nuestro / artículo) | SSPW (nuestro / artículo) |
|---|---|---|---|---|
| Sectores | EW | 0,422 / 0,422 | 54,2 / 54,2 | 0,100 / 0,100 |
| Sectores | MDP | 0,344 / 0,344 | 57,6 / 57,6 | 0,218 / 0,217 |
| Sectores | IVRB | 0,428 / 0,428 | 49,0 / 49,0 | 0,121 / 0,121 |
| Sectores | MV | 0,445 / 0,448 | 38,1 / 37,8 | 0,520 / 0,480 |
| Acciones | EW | 0,593 / 0,595 | 52,1 / 52,2 | 0,0028 / 0,0028 |
| Acciones | ERC | 0,568 / 0,570 | 49,3 / 49,4 | 0,0036 / 0,0036 |
| Acciones | IVRB | 0,559 / 0,560 | 46,9 / 47,1 | 0,0045 / 0,0045 |

Los métodos de agrupación (SL, CL, AL, WM y DBHT) salen parecidos pero no iguales, porque el artículo no explica con detalle cómo elige el número de grupos (ver el apartado 5).

## 3. Resultados del QHRP

Lo justo es comparar cada QHRP con el método clásico que usa el mismo enlace.

**10 sectores (recálculo diario)**

| | ASR | CEQ % | Caída máx. % |
|---|---|---|---|
| WM clásico | 0,418 | 7,07 | 53,6 |
| QHRP nuevo WM | 0,419 | 7,04 | 54,8 |
| QHRP original WM | 0,388 | 6,31 | 54,0 |
| AL clásico | 0,437 | 8,40 | 57,6 |
| QHRP nuevo AL | 0,437 | 7,31 | 52,3 |

- El QHRP nuevo rinde **igual** que su versión clásica: mismo ASR con Ward y con enlace medio.
- El QHRP nuevo AL **cae menos** en la peor crisis: 52,3 % frente a 57,6 %.
- El QHRP original queda algo por debajo.
- La prueba MCS **no distingue ningún método del mejor**: con 10 sectores todas las carteras se parecen mucho.

**357 acciones (recálculo mensual)**

| | ASR | CEQ % | Caída máx. % |
|---|---|---|---|
| WM clásico | 0,498 | 12,46 | 56,0 |
| QHRP original WM | 0,585 | 13,49 | 50,6 |
| QHRP nuevo WM | 0,549 | 12,43 | 56,0 |
| AL clásico | 0,744 | 23,15 | 52,1 |
| QHRP nuevo AL | 0,087 | 17,51 | 49,5 |

- **Con Ward, los dos QHRP mejoran el ASR del Ward clásico** (0,585 y 0,549 frente a 0,498). El QHRP original además gana en CEQ y cae menos; el QHRP nuevo queda igual en esas dos. Es el resultado más favorable al QHRP.
- **El QHRP nuevo AL** tiene un ASR muy bajo. Algunos meses mete mucho dinero en una sola acción, y un día extremo (+27 % en mayo de 2012) dispara la penalización del ASR.
- **En CEQ**, la prueba MCS deja como mejores a los clásicos **AL y SL**. Los QHRP quedan en el grupo de "no se distinguen del mejor, con poca seguridad" (marca ᵃ).

## 4. Conclusión

- El QHRP **no supera a los mejores métodos clásicos** del artículo en ninguno de los dos paneles.
- **Con Ward**, la distancia cuántica funciona mejor que la de correlación en acciones, pero no lo suficiente para superar a AL y SL clásicos.
- El QHRP nuevo con enlace medio **cae menos en las crisis** en los dos paneles. Merece la pena estudiarlo más.
- Con 6 qubits **no hay ventaja cuántica**: el circuito se simula sin problema en un ordenador normal, y el QHRP es bastante más lento que los métodos clásicos.

## 5. Diferencias con el artículo

- **Número de grupos:** se usa el índice Gap, con un mínimo de 2 grupos como dice el texto del artículo ("we consider at least two clusters"). En acciones, los métodos con Ward (WM y QHRP original WM) usan 2 grupos fijos, que es lo que más se acerca al artículo; con el Gap salían unos 9.
- **Se han probado todos los números de grupos fijos** (`probar_k/`). WM, CL y DBHT se acercan al artículo con 2 grupos. **SL y AL no se reproducen con ningún número fijo:** su concentración en el artículo queda entre la de 1 grupo y la de 2, lo que indica que el artículo usa 1 grupo casi todos los días y 2 o más en algunos (en torno al 10–20 %). Sin su código no se puede saber qué días.
- **Acciones con recálculo mensual**, no diario: el DBHT tarda unos 2 minutos por fecha con 357 acciones. La columna TO de acciones no se puede comparar con la del artículo.
- **DBHT:** se usa la implementación de la librería Riskfolio-Lib.
- **IVRB:** se usa 1/volatilidad. La fórmula impresa en el artículo usa la varianza, pero solo la volatilidad reproduce sus números.
- **MV en acciones** no coincide con el artículo (SSPW 0,078 frente a 0,0048). Una mínima varianza de verdad se concentra en pocas acciones; el valor del artículo parece indicar que su algoritmo no llegó a converger.
- **Prueba MCS:** la nuestra distingue menos entre métodos que la del artículo, así que salen más marcas ᵃ/ᵇ.

## 6. Cómo reproducirlo

Desde `src/qhrp_sp500/`:

```
../../raffinot-env/bin/python raffinot_tablas.py --panel sectores
../../raffinot-env/bin/python raffinot_tablas.py --panel acciones --paso 21
../../raffinot-env/bin/python leyenda_raffinot.py
```
