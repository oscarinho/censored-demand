# Censored Demand — recuperar la venta que el sistema no ve

> Lab nuevo, tesis clara desde el día 1 (lección aprendida del lab anterior: no empezar sin una pregunta real).

## Headline finding (benchmark completo: 50.000 series, 4.5M filas)

Pronosticar la **venta observada** subestima la demanda de forma sistemática (bias **−0.132**)
porque ~44% de los días están censurados por quiebres. Entrenar el **mismo** modelo sobre la
**demanda recuperada** casi cierra ese sesgo (bias **−0.014**) y baja el MAE **0.402 → 0.378**
(−5.9%) sobre un holdout *limpio* (206k filas, solo días con stock). La mejora se concentra
**donde vive la censura**: en series propensas a quiebre el bias pasa de **−0.154 → −0.022**;
en series estables ya estaban bien. La venta perdida oculta ≈ **19%** del volumen observado
(**866.515 unidades** que el registro nunca vio).

**La recuperación no es un free lunch.** Probado sobre **10 familias de modelos** (Naive ·
MovingAvg · ETS/Holt-Winters · Theta · SARIMA · Prophet · LightGBM · XGBoost · CatBoost ·
LSTM): recuperar el target sube el MAE de casi todas, pero **solo los modelos ricos en
features mantienen el bias cerca de cero** (gradient boosting + Prophet: **+7% de MAE** y bias
≈0). Los extrapoladores univariados puros mejoran el MAE pero **se pasan de largo** (flip a
bias positivo): suben el nivel sin poder localizar la corrección. Recuperar el target es
necesario, pero también hace falta un modelo lo bastante rico para usarlo.

> ⚠️ La demanda real es inobservable: la demanda recuperada es una *estimación*, nunca se
> evalúa contra ella. Números de esta corrida; correr con `N_SERIES = None` para el benchmark completo.

**Notebooks:** [`notebooks/Censored_Demand_CRISPML.ipynb`](notebooks/Censored_Demand_CRISPML.ipynb) (EN) ·
[`notebooks/Censored_Demand_CRISPML_ES.ipynb`](notebooks/Censored_Demand_CRISPML_ES.ipynb) (ES, gemelo)

## La pregunta

En retail, `Units Sold` **no es demanda** — es venta **censurada** por el stock: cuando la percha se vacía, la demanda real existió pero quedó invisible. Eso es **venta perdida / demanda no satisfecha**. La pregunta del lab:

**¿Puedo estimar la demanda real (no censurada) a partir de la venta observada + las marcas de quiebre de stock?**

Esto se llama **censored demand estimation / latent demand recovery / demand unconstraining**. Es un problema real, poco trillado, y conecta con tu mundo: la única forma operativa de "ver" la venta perdida es detectar la percha vacía (On-Shelf Availability) — hoy con **visión computacional / IoT**. Operaciones + CV + forecasting: tu carril.

## El dataset

**FreshRetailNet-50K** (Dingdong-Inc, HuggingFace) — el primer benchmark grande de demanda **censurada con anotación de stockouts**:
- 50,000 series tienda × producto, 90 días, **resolución horaria**.
- 898 tiendas, 18 ciudades, 865 SKUs perecederos.
- ~20% de datos con quiebre de stock **etiquetado** (la mayoría de datasets esconden esto).
- Incluye descuentos, clima (lluvia, temperatura, humedad), y covariables.

A diferencia del lab anterior (Kaggle sintético, oráculo + ruido), aquí hay **señal real y la censura está etiquetada** — que es justo lo que querías.

## Cómo traer los datos (corre donde haya internet: tu Mac / Claude Code)

Desde esta carpeta (`labs/censored-demand/`), con tu env `ml-exp`:

```bash
conda activate ml-exp
pip install datasets        # único faltante (ya tienes pandas, pyarrow, numpy, kaggle, kagglehub)
python scripts/get_and_profile.py
```

El script baja el dataset, guarda una muestra en `data/` y imprime un perfil inicial. Cuando el parquet quede en `data/` (sincronizado), yo hago el análisis de censura y te propongo el plan CRISP-ML.

**Alternativa LATAM (opcional):** Store Sales – Favorita (Kaggle, retailer ecuatoriano). Ya tienes `kaggle` en `ml-exp`; solo falta el token (`~/.kaggle/kaggle.json`) y aceptar las reglas en la web: `kaggle competitions download -c store-sales-time-series-forecasting`.

## Plan (acotado — definimos al ver los datos)

1. **Profiling** — confirmar tasa de stockout, evidencia de censura (venta topada por stock), señal real (precio/promo/clima ≠ 0).
2. **Baseline** — pronóstico ingenuo sobre venta observada (lo que hace todo el mundo, censurado).
3. **El punto del lab** — modelar la censura: estimar demanda latente en las horas con quiebre y comparar contra el baseline.
4. **Deploy + narrativa** — solo si los pasos 1-3 dan algo real.

No avanzamos al paso siguiente sin que el anterior valga.
