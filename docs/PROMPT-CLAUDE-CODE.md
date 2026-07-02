# Prompt para Claude Code — Notebook CRISP-ML(Q): Censored Demand

Estás construyendo el notebook de un lab nuevo. Iguala el estilo y el rigor de los labs existentes del repo, pero aprende de un error clave del lab anterior.

## Contexto
- Carpeta: `labs/censored-demand/`. Autor: Oscar Ponce (IA aplicada a operaciones). Audiencia: técnica + líderes de operaciones, bilingüe.
- Env: conda `ml-exp` (Python 3.11). Disponible: pandas, numpy, pyarrow, scikit-learn, lightgbm, xgboost, catboost, statsforecast, neuralforecast, pmdarima, prophet, statsmodels, shap, matplotlib, seaborn. Usa lo que encaje; prefiere lightgbm + statsforecast por velocidad.
- Plantilla de ESTILO (formato, NO conclusiones): `labs/forecasting-inventory/notebooks/Inventory_Forecasting_CRISPML.ipynb` y su README. Replica su estructura de 6 fases CRISP-ML(Q), markdown bilingüe EN/ES, tono de "caveat honesto", y el "headline finding" arriba. ⚠️ Ese lab fue retirado por problemas de datos — copia su forma, no sus números.

## La tesis (lo que este lab prueba)
En retail, la venta observada ≠ demanda. Cuando la percha se vacía (stockout), la demanda real queda censurada por la derecha e invisible — "venta perdida". Pronosticar la venta observada entrena al modelo para subestimar la demanda justo cuando más importa. Este lab: (1) muestra la trampa, (2) recupera la demanda latente (no censurada), (3) pronostica sobre la demanda recuperada, (4) cuantifica la venta perdida oculta.

## Datos (ya descargados)
- `data/freshretailnet_train_sample.parquet` (muestra de 20k filas para iterar rápido) y el dataset completo `Dingdong-Inc/FreshRetailNet-50K` (train 4.5M, eval 350k) vía `from datasets import load_dataset`.
- Esquema (19 cols): IDs (city_id, store_id, management_group_id, first/second/third_category_id, product_id), `dt` (fecha), `sale_amount` (venta diaria), `hours_sale` (array[24] venta horaria), `hours_stock_status` (array[24], **1=STOCKOUT/percha vacía, 0=con stock**), `stock_hour6_22_cnt` (horas OOS 6–22), `discount`, `holiday_flag`, `activity_flag`, `precpt`, `avg_temperature`, `avg_humidity`, `avg_wind_level`.

## Lo que ya encontramos (VERIFICALO contra los datos, no lo asumas)
- 1 = stockout: venta horaria media 0.004 (OOS) vs 0.054 (con stock), gap ~13×.
- 25% de las horas-tienda son stockout; 44.6% de los días (tienda×producto) tienen ≥1 hora OOS; ~7 de 17 horas comerciales perdidas en días con quiebre.
- Señal real (a diferencia de un set sintético): corr(venta, discount) = −0.34, lift de feriado +24%, efecto de clima leve.

## Metodología (concreta, por fase)
1. **Business Understanding** — el problema de demanda censurada / venta perdida, On-Shelf Availability, por qué importa operativamente. Bilingüe.
2. **Data Understanding** — verificá los hallazgos de arriba; EDA de prevalencia de stockout, perfil horario, correlaciones de señal. Quality gates (assert: arrays de largo 24, status ∈ {0,1}, fechas contiguas por serie).
3. **Data Preparation** — features (calendario, lags/rolling de demanda, discount, holiday, clima, categoría/tienda). Split **temporal** (train termina estrictamente antes del eval; sin leakage). Usa el split train/eval del dataset.
4. **Modeling**
   - a. **BASELINE / "la trampa"**: pronosticar `sale_amount` (censurado) directo con gradient boosting (+ un baseline de statsforecast).
   - b. **LATENT DEMAND RECOVERY**: estimar la demanda real en las horas OOS imputando desde las horas con stock (ej.: por tienda×producto×día, escalar por la fracción de horas con stock, o modelar el perfil de demanda hora-del-día desde las horas con stock y predecir las censuradas). Reconstruir `demand_unconstrained`.
   - c. **FORECAST SOBRE DEMANDA RECUPERADA**: mismo modelo entrenado sobre `demand_unconstrained`.
5. **Evaluation** — PUNTO CRÍTICO DE RIGOR: no puedes evaluar contra la demanda real (no es observable). Evalúa **solo sobre observaciones NO censuradas** (días/horas con stock), o reserva series totalmente in-stock. Muestra que el modelo entrenado sobre venta censurada **subestima** vs el de demanda recuperada en ese holdout limpio. Reporta MAE/RMSE/bias por segmento (series con tendencia a quiebre vs no). Usa el split eval.
6. **Deployment** — cuantificá la venta perdida (unidades/€), impacto de negocio, y el hook operativo: detectar perchas vacías (OSA, visión computacional) para marcar la censura en tiempo real. Limitaciones honestas (supuestos de imputación, qué generaliza).

## Guardrails (lecciones del lab anterior — NO repetir)
- No uses ninguna feature que filtre el target (el lab previo filtró vía Inventory Level, un proxy de censura). Aquí el stock/status se usa para **modelar la censura, nunca como predictor ingenuo de la venta**.
- No presentes números de un solo split como universales — backtest o usa el split eval.
- Declará supuestos y límites explícitamente (la "Q" de CRISP-ML). La demanda imputada es una estimación, no verdad de campo — decilo.
- Cero over-claiming. Si la recuperación solo ayuda en series con quiebre, decí exactamente eso.

## Entregables
- `notebooks/Censored_Demand_CRISPML.ipynb` (EN) y un gemelo en español `notebooks/Censored_Demand_CRISPML_ES.ipynb`.
- Un headline finding en `README.md` (1–2 líneas) cuando estén los resultados.
- Desarrolla con la muestra de 20k; corre los números finales sobre el split completo/eval.

## Definition of Done
- El notebook corre de principio a fin en `ml-exp` sin errores.
- Toda afirmación cuantitativa se calcula en el notebook (nada hardcodeado).
- La trampa (subestimación por censura) y la mejora por recuperación se muestran con números sobre un holdout limpio (no censurado), con caveats honestos.

Empezá cargando la muestra, verificá los hallazgos, y construí fase por fase. **Pregunta antes de cualquier corrida larga sobre el dataset completo.**
