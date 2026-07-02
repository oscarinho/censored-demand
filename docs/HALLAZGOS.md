# Hallazgos del profiling — FreshRetailNet-50K

> Análisis sobre muestra de 20,000 días (tienda × producto × día). Dataset total: 4.5M filas.

## Veredicto: este dataset SÍ vale. Tiene censura real y señal real.

### 1. La censura está etiquetada y es enorme
- `hours_stock_status == 1` = **STOCKOUT** (percha vacía): venta media **0.004/h** vs **0.054/h** cuando hay stock → **13× menos**. Confirmado.
- **25.1%** de todas las horas-tienda están en stockout.
- **44.6%** de los días (tienda × producto) tienen al menos una hora de quiebre.
- En un día con quiebre, en promedio **7.2 de las ~17 horas** de venta (6:00–22:00) están sin stock → ~43% del día comercial perdido.

### 2. El impacto de la venta perdida (lo que el sistema no ve)
- En las horas de stockout la venta observada es ~0 (censurada).
- Si esas horas vendieran al ritmo normal, serían **~6,500 unidades** solo en la muestra — demanda real que **desaparece de los registros**.
- Quien pronostica `sale_amount` directo entrena el modelo para **subestimar la demanda justo donde más importa** (el quiebre ocurre porque la demanda superó al stock). Esa es la trampa.

### 3. Señal real (lo que el dataset sintético NO tenía)
| Driver | Antes (sintético) | Ahora (FreshRetailNet) |
|---|---|---|
| Descuento | corr 0.003 (muerto) | **corr −0.338** (elasticidad real) |
| Feriado | corr 0.000 | **+24% de venta** (0.92 → 1.13) |
| Temperatura | — | corr +0.061 |
| Lluvia | — | corr +0.031 |

Hay drivers de verdad. No es ruido + oráculo.

## Esquema (19 columnas)
IDs (city/store/category/product) · `dt` · `sale_amount` (venta diaria) · `hours_sale` (24h) · `hours_stock_status` (24h, 1=stockout) · `stock_hour6_22_cnt` · `discount` · `holiday_flag` · `activity_flag` · clima (`precpt`, `avg_temperature`, `avg_humidity`, `avg_wind_level`).

## Plan CRISP-ML (acotado, ahora con datos vistos)

1. ✅ **Profiling** — hecho. Censura real (44.6% días) + señal real.
2. **La trampa (baseline)** — pronosticar `sale_amount` directo y mostrar que subestima sistemáticamente en días de quiebre.
3. **El aporte del lab — recuperar la demanda latente:** estimar la demanda real en las horas censuradas (imputar desde las horas con stock del mismo día/producto), reconstruir la demanda "sin restricción" y re-pronosticar sobre ella.
4. **El payoff de negocio** — cuantificar venta perdida (€/unidades) y mostrar que planear sobre demanda real reduce quiebres.
5. **Deploy + narrativa** — solo si 2–4 dan algo sólido. Ángulo: "la venta perdida que tu ERP no ve" + conexión a detección de percha vacía (visión computacional).

No avanzamos de paso sin que el anterior valga.
