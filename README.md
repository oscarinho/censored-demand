# Censored Demand — recuperar la venta que el sistema no ve

**En retail, `Units Sold` no es demanda: es lo que alcanzó a venderse antes de que la percha quedara vacía.** Este lab recupera la demanda latente que el quiebre de stock esconde y mide cuánto sesga eso al pronóstico — con datos reales de quiebre etiquetado.

🔗 **Caso completo:** [oscarponce.com/labs/censored-demand](https://oscarponce.com/labs/censored-demand)
📓 **Notebooks (gemelos bilingües):** [`Censored_Demand_CRISPML.ipynb`](notebooks/Censored_Demand_CRISPML.ipynb) (EN) · [`Censored_Demand_CRISPML_ES.ipynb`](notebooks/Censored_Demand_CRISPML_ES.ipynb) (ES)

## Hallazgo principal (benchmark completo: 50.000 series · 4.5M filas)

Pronosticar la **venta observada** subestima la demanda de forma sistemática (bias **−0.132**) porque el **44.3%** de los días tienen al menos una hora de quiebre. Entrenar el **mismo** modelo sobre la **demanda recuperada** casi cierra ese sesgo (**−0.014**) y baja el MAE **0.402 → 0.378** (**−5.9%**) sobre un holdout limpio (206k filas, solo días con stock).

La mejora se concentra **donde vive la censura**: en series propensas a quiebre el bias pasa de **−0.154 → −0.022**; las estables ya estaban bien. La venta perdida oculta ≈ **19%** del volumen observado — **866.515 unidades** que el registro nunca vio.

**No es un free lunch.** Sobre **11 familias de modelos** (Naive · Seasonal Naive · MovingAvg · ETS/Holt-Winters · Theta · SARIMA · Prophet · LightGBM · XGBoost · CatBoost · LSTM): recuperar el target sube el MAE de casi todas (la señal recuperada es más ruidosa), pero **solo los modelos ricos en features mantienen el bias cerca de cero** (gradient boosting + Prophet). Los univariados puros bajan MAE pero **voltean el bias a positivo** — suben el nivel sin poder localizar la corrección. Recuperar el target es necesario, pero también hace falta un modelo capaz de usarlo.

> ⚠️ La demanda real es inobservable: la recuperada es una **estimación**, nunca se evalúa contra ella. Cifras de la corrida completa (`N_SERIES = None`); dependen del supuesto de imputación.

## Insights

- **La trampa, cuantificada.** En el holdout limpio (donde no falta nada que perder) el modelo entrenado sobre venta censurada igual subestima: el sesgo está en el *target de entrenamiento*, no en el algoritmo.
- **La mejora es focalizada.** Concentrada en series propensas a quiebre; las estables casi no se mueven. Una mejora uniforme sería sospechosa.
- **Qué palancas explican la demanda oculta.** Al de-censurar, la importancia del **descuento** cae (≈46% → 5% de ganancia) y suben los **lags/rolling** de la propia demanda. *Ojo:* parte de eso es mecánico — la demanda recuperada se construye desde el nivel con-stock de la serie, así que queda más autorregresiva por diseño. No es un veredicto causal de que las promociones dejaron de importar.
- **Ángulo de negocio (OSA ~84%).** Leído como brecha bruta de servicio (demanda suprimida ÷ observada), ese ≈19% invisible equivale a operar cerca del **~84% de disponibilidad real** aunque el ERP reporte 100%.
- **Efecto látigo.** La censura no solo baja el pronóstico: deprime el cálculo del stock de seguridad en el ERP, dejando el sistema crónicamente vulnerable a nuevos quiebres — un ciclo que se auto-alimenta.

## La pregunta

En retail, `Units Sold` **no es demanda** — es venta **censurada** por el stock: cuando la percha se vacía, la demanda real existió pero quedó invisible. Eso es **venta perdida / demanda no satisfecha**. La pregunta del lab:

**¿Puedo estimar la demanda real (no censurada) a partir de la venta observada + las marcas de quiebre de stock?**

Esto se llama **censored demand estimation / latent demand recovery / demand unconstraining**. Es un problema real y poco trillado, y conecta con operaciones: la única forma de "ver" la venta perdida es detectar la percha vacía (On-Shelf Availability) — hoy con **visión computacional / IoT**. Operaciones + CV + forecasting.

## El dataset

**FreshRetailNet-50K** (Dingdong-Inc, HuggingFace) — el primer benchmark grande de demanda **censurada con anotación de stockouts**:

- 50.000 series tienda × producto, 90 días, **resolución horaria**.
- 898 tiendas, 18 ciudades, **863** SKUs perecederos.
- ≈**25%** de las horas-tienda en quiebre, **etiquetado** (la mayoría de datasets lo esconde).
- Incluye descuento, clima (lluvia, temperatura, humedad) y calendario.

## Método (CRISP-ML(Q))

1. **Profiling** — confirmar la censura: ≈25% de horas-tienda en quiebre, venta 13× menor en horas OOS, señal real en precio/clima/feriado.
2. **Recuperación** — imputar **solo** las horas censuradas (forma horaria por categoría × nivel con-stock de la serie); la demanda recuperada nunca baja de la observada. El estado de stock se usa para *ubicar* la censura, nunca como predictor de la venta.
3. **Evaluación honesta** — solo sobre días con cero horas de quiebre (holdout limpio), donde venta observada ≈ demanda real. Si el modelo censurado subestima incluso ahí, la trampa es real.
4. **Zoo de modelos** — 11 familias sobre ambos targets, para ver a cuáles ayuda de verdad la recuperación.
5. **Cuantificación** — venta perdida oculta y su traducción a disponibilidad (OSA).

## Cómo reproducirlo

```bash
# Dependencias (núcleo del estudio)
pip install datasets pandas numpy pyarrow matplotlib scikit-learn \
            lightgbm xgboost catboost prophet statsforecast torch pytorch-lightning
# En Apple Silicon, LightGBM/XGBoost necesitan OpenMP:
brew install libomp

# Datos: baja FreshRetailNet-50K y deja una muestra en data/
python scripts/get_and_profile.py
```

Luego abrir el notebook (`notebooks/Censored_Demand_CRISPML.ipynb` o el gemelo `_ES`):

- `N_SERIES = 4000` → itera rápido con la muestra local.
- `N_SERIES = None` → benchmark completo (50.000 series).

> El dataset y el PDF del paper no se versionan (ver `.gitignore`); se descargan / se citan.

## Relación con el paper

Este lab usa modelos accesibles y reproducibles (estadísticos clásicos + gradient boosting + un LSTM simple) e imputación heurística. El paper de FreshRetailNet-50K compara modelos *deep* SOTA (TimesNet, TFT, DLinear, iTransformer) e imputadores de atención (SAITS, ImputeFormer). Compartimos el **marco de dos etapas** (recuperar demanda latente → pronosticar), no las arquitecturas; por eso las cifras no son directamente comparables.

## Fuentes

- **Dataset / paper:** Wang Y. et al. (2026). *FreshRetailNet-50K: A Stockout-Annotated Censored Demand Dataset for Latent Demand Recovery and Forecasting in Fresh Retail.* [arXiv:2505.16319](https://arxiv.org/abs/2505.16319) · [HuggingFace](https://huggingface.co/datasets/Dingdong-Inc/FreshRetailNet-50K)
- **Metodología:** Studer S. et al. (2020). *Towards CRISP-ML(Q).* [arXiv:2003.05155](https://arxiv.org/abs/2003.05155)

---

**Autor:** [Oscar Ponce](https://oscarponce.com) — Operaciones + Datos + Visión Computacional
