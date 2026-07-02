"""Validación end-to-end de la metodología antes de armar el notebook.
Corre sobre data/dev_*.parquet (series completas). Imprime los números clave.
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

COMM = slice(6, 22)  # horas comerciales 6:00-22:00 (16h)

def load():
    tr = pd.read_parquet("data/dev_train.parquet")
    ev = pd.read_parquet("data/dev_eval.parquet")
    for d in (tr, ev):
        d["dt"] = pd.to_datetime(d["dt"])
    return tr.sort_values(["store_id","product_id","dt"]), ev.sort_values(["store_id","product_id","dt"])

def hourly_matrices(df):
    hs = np.vstack(df["hours_sale"].values).astype(float)        # n x 24
    st = np.vstack(df["hours_stock_status"].values).astype(int)  # n x 24, 1=OOS
    return hs, st

def recover_demand(df):
    """Reconstruye demanda no censurada imputando horas OOS desde un perfil
    hora-del-dia (a nivel categoria) escalado por el nivel diario in-stock de la serie."""
    hs, st = hourly_matrices(df)
    instock = (st == 0)
    # perfil hora-del-dia por segunda categoria, SOLO con horas in-stock (no censuradas)
    cat = df["second_category_id"].values
    prof = {}
    for c in np.unique(cat):
        m = cat == c
        num = np.where(instock[m], hs[m], np.nan)
        p = np.nanmean(num, axis=0)            # demanda media por hora (in-stock)
        p = np.nan_to_num(p, nan=0.0)
        prof[c] = p
    profile = np.vstack([prof[c] for c in cat])  # n x 24, forma de demanda esperada
    # normalizar el perfil a "peso" por hora (suma 1 sobre horas comerciales)
    w = profile.copy(); w[:, :6] = 0; w[:, 22:] = 0
    wsum = w.sum(axis=1, keepdims=True); wsum[wsum == 0] = 1
    w = w / wsum
    # nivel diario in-stock de la serie: media de sale_amount en dias 100% in-stock
    df = df.copy()
    full_instock_day = instock[:, 6:22].all(axis=1)
    df["_lvl"] = df["sale_amount"]
    lvl = (df[full_instock_day].groupby(["store_id","product_id"])["_lvl"].mean())
    glob = df.loc[full_instock_day, "_lvl"].mean()
    key = list(zip(df.store_id, df.product_id))
    series_lvl = np.array([lvl.get(k, np.nan) for k in key])
    series_lvl = np.where(np.isnan(series_lvl), df["sale_amount"].values, series_lvl)
    # demanda esperada por hora = nivel diario * peso de la hora
    expected_hourly = series_lvl[:, None] * w
    # demanda no censurada = observado donde in-stock, imputado donde OOS
    recovered = np.where(instock, hs, expected_hourly)
    demand = recovered[:, 6:22].sum(axis=1) + hs[:, :6].sum(axis=1) + hs[:, 22:].sum(axis=1)
    # nunca menor que lo observado
    demand = np.maximum(demand, df["sale_amount"].values)
    n_oos_comm = (st[:, 6:22] == 1).sum(axis=1)
    return demand, n_oos_comm

def add_features(df, target_col):
    df = df.sort_values(["store_id","product_id","dt"]).copy()
    g = df.groupby(["store_id","product_id"])[target_col]
    for L in (7, 14, 28):
        df[f"lag_{L}"] = g.shift(L)
    df["roll7"]  = g.transform(lambda s: s.shift(7).rolling(7).mean())
    df["roll28"] = g.transform(lambda s: s.shift(7).rolling(28).mean())
    df["dow"] = df["dt"].dt.dayofweek
    df["dom"] = df["dt"].dt.day
    return df

FEATS = ["lag_7","lag_14","lag_28","roll7","roll28","dow","dom",
         "discount","holiday_flag","activity_flag",
         "precpt","avg_temperature","avg_humidity","avg_wind_level",
         "second_category_id","third_category_id","store_id"]

def fit_predict(train, evalf, target):
    tr = add_features(train, target).dropna(subset=["lag_28"])
    m = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                          subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=-1)
    m.fit(tr[FEATS], tr[target])
    # features para eval: concat para que los lags miren al train
    full = pd.concat([train.assign(_ev=0), evalf.assign(_ev=1)], ignore_index=True)
    full = add_features(full, target)
    ef = full[full["_ev"] == 1].copy()
    ef["pred"] = m.predict(ef[FEATS])
    return ef[["store_id","product_id","dt","pred"]]

def main():
    tr, ev = load()
    print("train", tr.shape, "eval", ev.shape)
    # recuperar demanda en train y eval
    tr["demand"], tr["n_oos"] = recover_demand(tr)
    ev["demand"], ev["n_oos"] = recover_demand(ev)
    upl = (tr["demand"] - tr["sale_amount"])
    print("\n== censura / lost sales (train dev) ==")
    print("dias con OOS comercial:", (tr.n_oos>0).mean().round(3))
    print("uplift medio demanda vs venta:", upl.mean().round(4),
          " total unidades latentes:", upl.sum().round(0))
    print("uplift pct sobre venta observada: %.1f%%" % (100*upl.sum()/tr.sale_amount.sum()))

    # Modelo A (la trampa): entrena sobre venta censurada
    predA = fit_predict(tr, ev, "sale_amount").rename(columns={"pred":"predA"})
    # Modelo B (recuperado): entrena sobre demanda recuperada
    predB = fit_predict(tr, ev, "demand").rename(columns={"pred":"predB"})

    e = ev.merge(predA, on=["store_id","product_id","dt"]).merge(predB, on=["store_id","product_id","dt"])
    # HOLDOUT LIMPIO: solo dias eval 100% in-stock en horas comerciales -> venta ~= demanda real
    clean = e[e["n_oos"] == 0].copy()
    print("\n== evaluacion en holdout LIMPIO (eval in-stock) ==")
    print("n filas eval:", len(e), " limpias:", len(clean),
          "(%.1f%%)" % (100*len(clean)/len(e)))
    y = clean["sale_amount"].values
    for name, p in [("A_censurado", clean["predA"]), ("B_recuperado", clean["predB"])]:
        p = p.values
        print(f"  {name:14s} MAE={mean_absolute_error(y,p):.4f} "
              f"RMSE={mean_squared_error(y,p)**.5:.4f} bias={ (p-y).mean():+.4f}")

    # bias por segmento: series propensas a quiebre vs no
    sprone = tr.groupby(["store_id","product_id"])["n_oos"].apply(lambda s:(s>0).mean())
    clean = clean.merge(sprone.rename("oos_rate"), on=["store_id","product_id"])
    clean["seg"] = np.where(clean["oos_rate"]>=0.3, "propensa_quiebre", "estable")
    print("\n== bias por segmento (holdout limpio) ==")
    for seg, gdf in clean.groupby("seg"):
        y=gdf.sale_amount.values
        print(f"  {seg:18s} n={len(gdf):5d}  biasA={ (gdf.predA-y).mean():+.4f}  "
              f"biasB={ (gdf.predB-y).mean():+.4f}  "
              f"MAE_A={mean_absolute_error(y,gdf.predA):.3f} MAE_B={mean_absolute_error(y,gdf.predB):.3f}")

if __name__ == "__main__":
    main()
