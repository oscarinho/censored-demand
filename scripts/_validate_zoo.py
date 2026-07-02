"""Valida el 'model zoo' ampliado: cada familia de forecasting entrenada sobre
venta censurada vs demanda recuperada, evaluada en holdout limpio.
Corre sobre data/dev_*.parquet. Imprime el leaderboard.
"""
import os, time, logging, warnings
os.environ["NIXTLA_ID_AS_COL"] = "1"
warnings.filterwarnings("ignore")
for n in ["prophet","cmdstanpy","pytorch_lightning","lightning.pytorch"]:
    logging.getLogger(n).setLevel(logging.ERROR)
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

COMM = slice(6, 22); RNG = 42
MODEL_SAMPLE = 600

def hourly(df):
    return (np.vstack(df["hours_sale"].values).astype(float),
            np.vstack(df["hours_stock_status"].values).astype(int))

def recover_demand(df):
    df = df.reset_index(drop=True); hs, st = hourly(df); instock = st == 0
    cat = df["second_category_id"].values; shape = np.zeros((len(df), 24))
    for c in np.unique(cat):
        m = cat == c
        shape[m] = np.nan_to_num(np.nanmean(np.where(instock[m], hs[m], np.nan), 0))
    w = shape.copy(); w[:, :6] = 0; w[:, 22:] = 0
    s = w.sum(1, keepdims=True); s[s == 0] = 1; w /= s
    full = instock[:, COMM].all(1)
    lvl = df.assign(_f=full, _s=df.sale_amount)[full].groupby(
        ["store_id","product_id"])["_s"].mean() if full.any() else pd.Series(dtype=float)
    key = list(zip(df.store_id, df.product_id))
    sl = np.array([lvl.get(k, np.nan) for k in key])
    sl = np.where(np.isnan(sl), df.sale_amount.values, sl)
    rec = np.where(instock, hs, sl[:, None] * w)
    dem = rec[:, COMM].sum(1) + hs[:, :6].sum(1) + hs[:, 22:].sum(1)
    return np.maximum(dem, df.sale_amount.values), (st[:, COMM] == 1).sum(1)

# ---------------- load ----------------
tr = pd.read_parquet("data/dev_train.parquet"); ev = pd.read_parquet("data/dev_eval.parquet")
for d in (tr, ev): d["dt"] = pd.to_datetime(d["dt"])
tr = tr.sort_values(["store_id","product_id","dt"]).reset_index(drop=True)
ev = ev.sort_values(["store_id","product_id","dt"]).reset_index(drop=True)
tr["demand"], tr["n_oos"] = recover_demand(tr)
ev["demand"], ev["n_oos"] = recover_demand(ev)
for d in (tr, ev): d["uid"] = d.store_id.astype(str) + "_" + d.product_id.astype(str)

uids = pd.Series(tr.uid.unique()).sample(MODEL_SAMPLE, random_state=RNG).values
trz = tr[tr.uid.isin(uids)].copy(); evz = ev[ev.uid.isin(uids)].copy()
truth = evz[["uid","dt","sale_amount","n_oos"]].rename(columns={"dt":"ds"})
print(f"model-zoo: {len(uids)} series, {trz.shape[0]} train rows, {evz.shape[0]} eval rows")

# ---------------- statsforecast ----------------
def run_stats(target):
    from statsforecast import StatsForecast
    from statsforecast.models import (Naive, SeasonalNaive, WindowAverage,
                                       AutoETS, AutoTheta, AutoARIMA)
    df = trz[["uid","dt",target]].rename(columns={"uid":"unique_id","dt":"ds",target:"y"})
    models = [Naive(), SeasonalNaive(season_length=7), WindowAverage(window_size=7),
              AutoETS(season_length=7), AutoTheta(season_length=7), AutoARIMA(season_length=7)]
    sf = StatsForecast(models=models, freq="D", n_jobs=1)
    fc = sf.forecast(df=df, h=7).rename(columns={"unique_id":"uid","SeasonalNaive":"SeasonalNaive(7)",
        "WindowAverage":"MovingAvg(7)","AutoETS":"ETS/Holt-Winters","AutoTheta":"Theta","AutoARIMA":"SARIMA"})
    return fc

# ---------------- prophet ----------------
def run_prophet(target):
    from prophet import Prophet
    out = []
    for u, g in trz.groupby("uid"):
        m = Prophet(weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=False)
        m.add_regressor("discount"); m.add_regressor("holiday_flag")
        m.fit(g[["dt",target,"discount","holiday_flag"]].rename(columns={"dt":"ds",target:"y"}))
        fut = evz[evz.uid == u][["dt","discount","holiday_flag"]].rename(columns={"dt":"ds"})
        p = m.predict(fut)[["ds","yhat"]]; p["uid"] = u; out.append(p)
    return pd.concat(out).rename(columns={"yhat":"Prophet"})

# ---------------- global ML ----------------
def add_features(df, target):
    df = df.sort_values(["store_id","product_id","dt"]).copy()
    g = df.groupby(["store_id","product_id"])[target]
    for L in (7,14,28): df[f"lag_{L}"] = g.shift(L)
    df["roll7"] = g.transform(lambda s: s.shift(7).rolling(7).mean())
    df["roll28"] = g.transform(lambda s: s.shift(7).rolling(28).mean())
    df["dow"], df["dom"] = df.dt.dt.dayofweek, df.dt.dt.day
    return df
FEATS = ["lag_7","lag_14","lag_28","roll7","roll28","dow","dom","discount","holiday_flag",
         "activity_flag","precpt","avg_temperature","avg_humidity","avg_wind_level",
         "second_category_id","third_category_id","store_id"]
def run_ml(target, frame_tr, frame_ev):
    trf = add_features(frame_tr, target).dropna(subset=["lag_28"])
    full = pd.concat([frame_tr.assign(_e=0), frame_ev.assign(_e=1)], ignore_index=True)
    ef = add_features(full, target); ef = ef[ef._e == 1].copy()
    res = ef[["uid","dt"]].rename(columns={"dt":"ds"}).copy()
    for name, M in [("LightGBM", lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                        num_leaves=63, subsample=.8, colsample_bytree=.8, random_state=0, n_jobs=-1, verbose=-1)),
                    ("XGBoost", xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=7,
                        subsample=.8, colsample_bytree=.8, random_state=0, n_jobs=-1, verbosity=0)),
                    ("CatBoost", CatBoostRegressor(iterations=400, learning_rate=0.05, depth=7,
                        random_state=0, verbose=0))]:
        M.fit(trf[FEATS], trf[target]); res[name] = np.maximum(M.predict(ef[FEATS]), 0)
    return res

# ---------------- LSTM ----------------
def run_lstm(target):
    from neuralforecast import NeuralForecast
    from neuralforecast.models import LSTM
    df = trz[["uid","dt",target]].rename(columns={"uid":"unique_id","dt":"ds",target:"y"})
    nf = NeuralForecast(models=[LSTM(h=7, input_size=28, max_steps=300, scaler_type="robust",
                                     enable_progress_bar=False, logger=False)], freq="D")
    nf.fit(df); fc = nf.predict().rename(columns={"unique_id":"uid"})
    fc["LSTM"] = np.maximum(fc["LSTM"], 0); return fc[["uid","ds","LSTM"]]

# ---------------- run both targets ----------------
def all_models(target):
    t0 = time.time()
    s = run_stats(target)
    p = run_prophet(target)
    m = run_ml(target, trz, evz)
    l = run_lstm(target)
    out = (s.merge(p, on=["uid","ds"]).merge(m, on=["uid","ds"]).merge(l, on=["uid","ds"]))
    print(f"  [{target}] models in {time.time()-t0:.0f}s")
    return out

MODELS = ["Naive","SeasonalNaive(7)","MovingAvg(7)","ETS/Holt-Winters","Theta","SARIMA",
          "Prophet","LightGBM","XGBoost","CatBoost","LSTM"]
print("running censored target..."); cen = all_models("sale_amount")
print("running recovered target..."); rec = all_models("demand")

# ---------------- evaluate on clean holdout ----------------
def score(pred):
    d = pred.merge(truth, on=["uid","ds"]); d = d[d.n_oos == 0]
    y = d["sale_amount"].values; rows = {}
    for mdl in MODELS:
        p = d[mdl].values
        rows[mdl] = (mean_absolute_error(y, p), mean_squared_error(y, p)**.5, (p - y).mean())
    return d, rows

dclean, sc = score(cen); _, sr = score(rec)
print(f"\nclean holdout rows: {len(dclean)} of {len(truth)}")
lb = pd.DataFrame({
    "MAE_cens":[sc[m][0] for m in MODELS], "MAE_rec":[sr[m][0] for m in MODELS],
    "bias_cens":[sc[m][2] for m in MODELS], "bias_rec":[sr[m][2] for m in MODELS],
}, index=MODELS)
lb["MAE_gain%"] = (1 - lb.MAE_rec / lb.MAE_cens) * 100
lb["|bias|_drop"] = lb.bias_cens.abs() - lb.bias_rec.abs()
pd.set_option("display.width", 200, "display.float_format", lambda v: f"{v:,.4f}")
print("\n================ LEADERBOARD (clean holdout) ================")
print(lb.sort_values("MAE_rec"))
print("\nmean bias censored:", lb.bias_cens.mean().round(4),
      " recovered:", lb.bias_rec.mean().round(4))
print("models where recovery cut |bias|:", int((lb["|bias|_drop"] > 0).sum()), "of", len(MODELS))
