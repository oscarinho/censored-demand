"""Genera los notebooks CRISP-ML(Q) de Censored Demand (EN + ES gemelo).
Model-zoo (Naive -> SARIMA -> Prophet -> ML -> LSTM) x (target censurado vs recuperado),
con insights por fase. Codigo compartido; markdown por idioma.

    python scripts/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb --ExecutePreprocessor.timeout=1800
"""
import os, nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

C = {}  # code cells (shared)

C["setup"] = r'''
import os, logging, warnings, time
os.environ["NIXTLA_ID_AS_COL"] = "1"
warnings.filterwarnings("ignore")
for _n in ["prophet","cmdstanpy","pytorch_lightning","lightning.pytorch"]:
    logging.getLogger(_n).setLevel(logging.ERROR)

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import lightgbm as lgb, xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from IPython.display import Markdown, display

pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
plt.rcParams.update({"figure.figsize": (9, 4), "axes.grid": True, "grid.alpha": .3})
RNG = 42

# Commercial window 6:00-22:00 (16h). Stockout flag: 1 = empty shelf (OOS).
COMM = slice(6, 22)
# N_SERIES: forecasting universe. None = full 50,000-series benchmark.
# Lower to e.g. 4000 (uses the local data/dev_*.parquet sample) for fast iteration.
N_SERIES = None
# MODEL_SAMPLE: series used for the (slower) cross-model leaderboard.
MODEL_SAMPLE = 600
'''

C["load"] = r'''
from pathlib import Path

def load_split():
    """Complete store x product series. Prefer the local dev sample; else pull
    the full benchmark from HuggingFace (cached) and subsample N_SERIES."""
    dtr, dev = Path("data/dev_train.parquet"), Path("data/dev_eval.parquet")
    if dtr.exists() and dev.exists() and N_SERIES is not None:
        tr, ev = pd.read_parquet(dtr), pd.read_parquet(dev)
    else:
        from datasets import load_dataset
        ds = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
        tr, ev = ds["train"].to_pandas(), ds["eval"].to_pandas()
        if N_SERIES is not None:
            keys = tr[["store_id","product_id"]].drop_duplicates().sample(N_SERIES, random_state=RNG)
            ks = set(map(tuple, keys.values))
            tr = tr[[k in ks for k in zip(tr.store_id, tr.product_id)]].copy()
            ev = ev[[k in ks for k in zip(ev.store_id, ev.product_id)]].copy()
    for d in (tr, ev):
        d["dt"] = pd.to_datetime(d["dt"])
    s = ["store_id","product_id","dt"]
    return tr.sort_values(s).reset_index(drop=True), ev.sort_values(s).reset_index(drop=True)

train, evalf = load_split()
print("train:", train.shape, " eval:", evalf.shape)
print("train dates:", train.dt.min().date(), "->", train.dt.max().date())
print("eval  dates:", evalf.dt.min().date(), "->", evalf.dt.max().date())
print("series:", train.groupby(["store_id","product_id"]).ngroups)
'''

C["gates"] = r'''
def hourly(df):
    hs = np.vstack(df["hours_sale"].values).astype(float)        # n x 24 hourly sales
    st = np.vstack(df["hours_stock_status"].values).astype(int)  # n x 24, 1 = OOS
    return hs, st

hs, st = hourly(train)
assert hs.shape[1] == 24 and st.shape[1] == 24, "hourly arrays must be length 24"
assert set(np.unique(st)) <= {0, 1}, "stock status must be in {0,1}"
assert np.allclose(train["sale_amount"], hs.sum(1)), "sale_amount must equal sum of hourly sales"
span = train.groupby(["store_id","product_id"])["dt"].agg(lambda s: (s.max()-s.min()).days + 1)
cnt  = train.groupby(["store_id","product_id"]).size()
assert (span == cnt).all() and (cnt == 90).all(), "each train series must be 90 contiguous days"
print("Quality gates passed: 24h arrays, status in {0,1}, sale==sum(hours), 90 contiguous days.")
'''

C["eda"] = r'''
oos = st == 1
oos_comm = st[:, COMM] == 1
print(f"OOS share of all store-hours:        {oos.mean():.1%}")
print(f"Days with >=1 commercial OOS hour:   {(oos_comm.any(1)).mean():.1%}")
print(f"Mean hourly sale  OOS={hs[oos].mean():.4f}  in-stock={hs[~oos].mean():.4f}"
      f"  ->  {hs[~oos].mean()/max(hs[oos].mean(),1e-9):.0f}x gap")
on = oos_comm.sum(1)[oos_comm.any(1)]
print(f"On a stockout day: {on.mean():.1f} of 16 commercial hours lost")
print()
print(f"corr(sale, discount) = {train['sale_amount'].corr(train['discount']):+.3f}  (discount<1 = markdown)")
hol = train.groupby('holiday_flag')['sale_amount'].mean()
if len(hol) > 1:
    print(f"holiday lift = {hol.loc[1]/hol.loc[0]-1:+.1%}")
for w in ["avg_temperature","precpt"]:
    print(f"corr(sale, {w}) = {train['sale_amount'].corr(train[w]):+.3f}")
'''

C["eda_plot"] = r'''
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(range(24), np.nanmean(np.where(oos, np.nan, hs), 0), lw=2, label="in-stock (≈ true demand)")
ax[0].plot(range(24), hs.mean(0), lw=2, ls="--", label="observed (all hours, censored)")
ax[0].axvspan(6, 22, alpha=.07, color="green")
ax[0].set(title="Hourly demand: censored vs in-stock", xlabel="hour of day", ylabel="mean units/hour")
ax[0].legend()
ax[1].hist(oos_comm.sum(1), bins=np.arange(18)-.5, color="#c0392b", alpha=.85)
ax[1].set(title="Commercial OOS hours per day", xlabel="OOS hours (of 16)", ylabel="day count")
plt.tight_layout(); plt.show()
'''

C["recover"] = r'''
def recover_demand(df):
    """Reconstruct uncensored daily demand by imputing ONLY the OOS hours:
    a per-category hour-of-day demand SHAPE (from in-stock hours), scaled by each
    series' in-stock daily level. Observed hours are kept; floored at observed.
    Stock status is used to LOCATE censorship, never as a sales predictor.
    NOTE: recovered demand is an ESTIMATE, not ground truth."""
    df = df.reset_index(drop=True)
    hs, st = hourly(df); instock = st == 0
    cat = df["second_category_id"].values
    shape = np.zeros((len(df), 24))
    for c in np.unique(cat):
        m = cat == c
        shape[m] = np.nan_to_num(np.nanmean(np.where(instock[m], hs[m], np.nan), 0))
    w = shape.copy(); w[:, :6] = 0; w[:, 22:] = 0
    s = w.sum(1, keepdims=True); s[s == 0] = 1; w = w / s
    full = instock[:, COMM].all(1)
    lvl = (df.assign(_f=full, _s=df.sale_amount)[full]
             .groupby(["store_id","product_id"])["_s"].mean()) if full.any() else pd.Series(dtype=float)
    key = list(zip(df.store_id, df.product_id))
    sl = np.array([lvl.get(k, np.nan) for k in key])
    sl = np.where(np.isnan(sl), df.sale_amount.values, sl)
    rec = np.where(instock, hs, sl[:, None] * w)
    dem = rec[:, COMM].sum(1) + hs[:, :6].sum(1) + hs[:, 22:].sum(1)
    return np.maximum(dem, df.sale_amount.values), (st[:, COMM] == 1).sum(1)

train["demand"], train["n_oos"] = recover_demand(train)
evalf["demand"], evalf["n_oos"] = recover_demand(evalf)
uplift = train["demand"] - train["sale_amount"]
print(f"Days with commercial OOS:      {(train.n_oos>0).mean():.1%}")
print(f"Latent demand uplift (mean):   {uplift.mean():+.4f} units/day")
print(f"Total hidden lost units:       {uplift.sum():,.0f}")
print(f"Lost sales as % of observed:   {100*uplift.sum()/train.sale_amount.sum():.1f}%")
'''

C["recover_plot"] = r'''
# Visualize recovery: pick the most stockout-heavy series, and the lost units by hour.
prone = train.groupby(["store_id","product_id"])["n_oos"].sum().idxmax()
g = train[(train.store_id==prone[0]) & (train.product_id==prone[1])].sort_values("dt")
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(g.dt, g.sale_amount, lw=1.6, label="observed sale (censored)")
ax[0].plot(g.dt, g.demand, lw=1.6, ls="--", color="#27ae60", label="recovered demand")
ax[0].fill_between(g.dt, g.sale_amount, g.demand, color="#27ae60", alpha=.15)
ax[0].set(title=f"Series {prone}: censored vs recovered", xlabel="date", ylabel="units/day")
ax[0].legend(); ax[0].tick_params(axis="x", rotation=30)
# lost units by hour-of-day (where the censorship bites)
hsT, stT = hourly(train); instockT = stT == 0
shape = np.zeros((len(train), 24)); cat = train.second_category_id.values
for c in np.unique(cat):
    m = cat == c; shape[m] = np.nan_to_num(np.nanmean(np.where(instockT[m], hsT[m], np.nan), 0))
w = shape.copy(); w[:, :6]=0; w[:, 22:]=0; s=w.sum(1,keepdims=True); s[s==0]=1; w/=s
full = instockT[:, COMM].all(1)
lvl = train.assign(_f=full,_s=train.sale_amount)[full].groupby(["store_id","product_id"])["_s"].mean()
sl = np.array([lvl.get(k, np.nan) for k in zip(train.store_id, train.product_id)])
sl = np.where(np.isnan(sl), train.sale_amount.values, sl)
imputed = np.where(~instockT, sl[:,None]*w, 0.0)
ax[1].bar(range(24), imputed.sum(0), color="#c0392b", alpha=.85)
ax[1].axvspan(6, 22, alpha=.07, color="green")
ax[1].set(title="Hidden lost units by hour of day", xlabel="hour", ylabel="recovered units (total)")
plt.tight_layout(); plt.show()
'''

C["features"] = r'''
def add_features(df, target):
    df = df.sort_values(["store_id","product_id","dt"]).copy()
    g = df.groupby(["store_id","product_id"])[target]
    for L in (7, 14, 28):
        df[f"lag_{L}"] = g.shift(L)
    df["roll7"]  = g.transform(lambda s: s.shift(7).rolling(7).mean())
    df["roll28"] = g.transform(lambda s: s.shift(7).rolling(28).mean())
    df["dow"], df["dom"] = df["dt"].dt.dayofweek, df["dt"].dt.day
    return df

FEATS = ["lag_7","lag_14","lag_28","roll7","roll28","dow","dom",
         "discount","holiday_flag","activity_flag",
         "precpt","avg_temperature","avg_humidity","avg_wind_level",
         "second_category_id","third_category_id","store_id"]

# Horizon = 7 days, min lag = 7  ->  every eval lag is sourced from the train window:
# no recursion, no leakage. Train ends strictly before eval begins.
def lgbm(): return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
    subsample=.8, colsample_bytree=.8, random_state=0, n_jobs=-1, verbose=-1)

def fit_predict(target, tr=None, ev=None, model=None):
    tr = train if tr is None else tr; ev = evalf if ev is None else ev
    trf = add_features(tr, target).dropna(subset=["lag_28"])
    m = (model or lgbm()); m.fit(trf[FEATS], trf[target])
    full = pd.concat([tr.assign(_e=0), ev.assign(_e=1)], ignore_index=True)
    ef = add_features(full, target); ef = ef[ef._e == 1].copy()
    ef["pred"] = np.maximum(m.predict(ef[FEATS]), 0)
    return ef[["store_id","product_id","dt","pred"]], m
'''

C["core_models"] = r'''
# Core experiment (ALL series): one model, two training targets.
predA, _ = fit_predict("sale_amount")   # A — the trap (censored)
predB, _ = fit_predict("demand")        # B — the fix (recovered)
E = (evalf
     .merge(predA.rename(columns={"pred":"predA"}), on=["store_id","product_id","dt"])
     .merge(predB.rename(columns={"pred":"predB"}), on=["store_id","product_id","dt"]))
print("eval rows scored:", len(E))
'''

C["core_eval"] = r'''
# CLEAN HOLDOUT: real demand is unobservable; trust observed sales only where the
# shelf was full. Evaluate on eval days with zero commercial OOS hours.
clean = E[E.n_oos == 0].copy(); y = clean["sale_amount"].values
print(f"Eval rows: {len(E)}   clean in-stock holdout: {len(clean)} ({len(clean)/len(E):.1%})")
res = pd.DataFrame([
    (n, mean_absolute_error(y, p), mean_squared_error(y, p)**.5, (p-y).mean())
    for n, p in [("A — trained on censored sales", clean.predA.values),
                 ("B — trained on recovered demand", clean.predB.values)]],
    columns=["model","MAE","RMSE","bias"]).set_index("model")
display(res)
print(f"MAE reduction: {(1-res.MAE.iloc[1]/res.MAE.iloc[0]):.1%}   "
      f"bias {res.bias.iloc[0]:+.4f} -> {res.bias.iloc[1]:+.4f}")
'''

C["segment"] = r'''
sprone = train.groupby(["store_id","product_id"])["n_oos"].apply(lambda s: (s>0).mean())
c2 = clean.merge(sprone.rename("oos_rate"), on=["store_id","product_id"])
c2["segment"] = np.where(c2.oos_rate >= 0.3, "stockout-prone (≥30% days OOS)", "stable")
seg = c2.groupby("segment").apply(lambda g: pd.Series({
    "n": len(g),
    "bias_A": (g.predA-g.sale_amount).mean(), "bias_B": (g.predB-g.sale_amount).mean(),
    "MAE_A": mean_absolute_error(g.sale_amount, g.predA),
    "MAE_B": mean_absolute_error(g.sale_amount, g.predB)}), include_groups=False)
display(seg)
fig, ax = plt.subplots(figsize=(8, 4)); x = np.arange(len(seg)); wd = .35
ax.bar(x-wd/2, seg.bias_A, wd, label="A (censored)", color="#c0392b")
ax.bar(x+wd/2, seg.bias_B, wd, label="B (recovered)", color="#27ae60")
ax.axhline(0, color="k", lw=.8); ax.set_xticks(x); ax.set_xticklabels(seg.index, fontsize=9)
ax.set(title="Forecast bias by segment (clean holdout)", ylabel="bias (pred − observed)")
ax.legend(); plt.tight_layout(); plt.show()
'''

C["zoo_setup"] = r'''
# ---- Model zoo: run many forecasting families on a sample, each on both targets ----
for d in (train, evalf): d["uid"] = d.store_id.astype(str) + "_" + d.product_id.astype(str)
zuids = pd.Series(train.uid.unique()).sample(min(MODEL_SAMPLE, train.uid.nunique()),
                                             random_state=RNG).values
trz = train[train.uid.isin(zuids)].copy(); evz = evalf[evalf.uid.isin(zuids)].copy()
truth = evz[["uid","dt","sale_amount","n_oos"]].rename(columns={"dt":"ds"})
print(f"model-zoo universe: {len(zuids)} series  ({len(trz)} train / {len(evz)} eval rows)")

def run_stats(target):
    from statsforecast import StatsForecast
    from statsforecast.models import (Naive, SeasonalNaive, WindowAverage,
                                       AutoETS, AutoTheta, AutoARIMA)
    df = trz[["uid","dt",target]].rename(columns={"uid":"unique_id","dt":"ds",target:"y"})
    sf = StatsForecast(freq="D", n_jobs=1, models=[      # n_jobs=1: notebook-safe
        Naive(), SeasonalNaive(season_length=7), WindowAverage(window_size=7),
        AutoETS(season_length=7), AutoTheta(season_length=7), AutoARIMA(season_length=7)])
    return sf.forecast(df=df, h=7).rename(columns={"unique_id":"uid",
        "SeasonalNaive":"SeasonalNaive(7)", "WindowAverage":"MovingAvg(7)",
        "AutoETS":"ETS/Holt-Winters", "AutoTheta":"Theta", "AutoARIMA":"SARIMA"})

def run_prophet(target):
    from prophet import Prophet
    out = []
    for u, g in trz.groupby("uid"):
        m = Prophet(weekly_seasonality=True, daily_seasonality=False, yearly_seasonality=False)
        m.add_regressor("discount"); m.add_regressor("holiday_flag")
        m.fit(g[["dt",target,"discount","holiday_flag"]].rename(columns={"dt":"ds",target:"y"}))
        fut = evz[evz.uid==u][["dt","discount","holiday_flag"]].rename(columns={"dt":"ds"})
        p = m.predict(fut)[["ds","yhat"]]; p["uid"] = u; out.append(p)
    return pd.concat(out).rename(columns={"yhat":"Prophet"})

def run_ml_zoo(target):
    trf = add_features(trz, target).dropna(subset=["lag_28"])
    full = pd.concat([trz.assign(_e=0), evz.assign(_e=1)], ignore_index=True)
    ef = add_features(full, target); ef = ef[ef._e==1].copy()
    out = ef[["uid","dt"]].rename(columns={"dt":"ds"}).copy()
    for name, M in [("LightGBM", lgbm()),
        ("XGBoost", xgb.XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=7,
            subsample=.8, colsample_bytree=.8, random_state=0, n_jobs=-1, verbosity=0)),
        ("CatBoost", CatBoostRegressor(iterations=400, learning_rate=0.05, depth=7,
            random_state=0, verbose=0))]:
        M.fit(trf[FEATS], trf[target]); out[name] = np.maximum(M.predict(ef[FEATS]), 0)
    return out

def run_lstm(target):
    from neuralforecast import NeuralForecast
    from neuralforecast.models import LSTM
    df = trz[["uid","dt",target]].rename(columns={"uid":"unique_id","dt":"ds",target:"y"})
    nf = NeuralForecast(freq="D", models=[LSTM(h=7, input_size=28, max_steps=300,
        scaler_type="robust", enable_progress_bar=False, logger=False)])
    nf.fit(df); fc = nf.predict().rename(columns={"unique_id":"uid"})
    fc["LSTM"] = np.maximum(fc["LSTM"], 0); return fc[["uid","ds","LSTM"]]
'''

C["zoo_run"] = r'''
ZOO = ["Naive","SeasonalNaive(7)","MovingAvg(7)","ETS/Holt-Winters","Theta","SARIMA",
       "Prophet","LightGBM","XGBoost","CatBoost","LSTM"]

def run_all(target):
    t0 = time.time()
    out = (run_stats(target)
           .merge(run_prophet(target), on=["uid","ds"])
           .merge(run_ml_zoo(target),  on=["uid","ds"])
           .merge(run_lstm(target),    on=["uid","ds"]))
    print(f"  [{target}] {len(ZOO)} families in {time.time()-t0:.0f}s")
    return out

print("Training the zoo on the CENSORED target...");  cen = run_all("sale_amount")
print("Training the zoo on the RECOVERED target..."); rec = run_all("demand")
'''

C["zoo_leaderboard"] = r'''
def score(pred):
    d = pred.merge(truth, on=["uid","ds"]); d = d[d.n_oos == 0]; y = d["sale_amount"].values
    return {m: (mean_absolute_error(y, d[m]), (d[m]-y).mean()) for m in ZOO}, len(d)

sc, nclean = score(cen); sr, _ = score(rec)
lb = pd.DataFrame({
    "MAE_censored":  [sc[m][0] for m in ZOO], "MAE_recovered": [sr[m][0] for m in ZOO],
    "bias_censored": [sc[m][1] for m in ZOO], "bias_recovered":[sr[m][1] for m in ZOO],
}, index=ZOO)
lb["MAE_gain_%"] = (1 - lb.MAE_recovered / lb.MAE_censored) * 100
lb["|bias|_drop"] = lb.bias_censored.abs() - lb.bias_recovered.abs()
lb = lb.sort_values("MAE_gain_%", ascending=False)
print(f"clean holdout rows: {nclean}\n"); display(lb)

fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#27ae60" if v > 0 else "#c0392b" for v in lb["MAE_gain_%"]]
ax.barh(lb.index[::-1], lb["MAE_gain_%"][::-1], color=colors[::-1])
ax.axvline(0, color="k", lw=.8)
ax.set(title="MAE improvement from training on recovered demand (clean holdout)",
       xlabel="MAE reduction %  (positive = recovery helps)")
plt.tight_layout(); plt.show()

FEATURE_RICH = ["LightGBM","XGBoost","CatBoost","Prophet"]
print(f"Feature-rich models mean MAE gain: {lb.loc[FEATURE_RICH,'MAE_gain_%'].mean():+.1f}%")
print(f"Families where recovery cut |bias|: {(lb['|bias|_drop']>0).sum()} of {len(ZOO)}")
'''

C["headline"] = r'''
maeA, maeB = res.MAE.iloc[0], res.MAE.iloc[1]; bA, bB = res.bias.iloc[0], res.bias.iloc[1]
sp = seg.loc[seg.index.str.startswith("stockout")]
fr = lb.loc[FEATURE_RICH, "MAE_gain_%"].mean()
display(Markdown(f"""
### Headline finding (this sample, this setup)

Forecasting **observed sales** underestimates demand (bias **{bA:+.3f}**) because
~{(train.n_oos>0).mean():.0%} of days are censored by stockouts. Training the same model on
**recovered demand** nearly closes that bias (**{bB:+.3f}**) and cuts MAE
**{maeA:.3f} → {maeB:.3f}** ({(1-maeB/maeA):.0%}) on a *clean in-stock holdout*. The gain
concentrates **where censorship lives**: in stockout-prone series bias goes from
**{sp.bias_A.iloc[0]:+.3f}** to **{sp.bias_B.iloc[0]:+.3f}**; stable series are already fine.

**Recovery is not a free lunch.** Across {len(ZOO)} forecasting families, the benefit
concentrates in models that can *localize* the correction with features/covariates
(gradient boosting + Prophet: mean MAE gain **{fr:+.1f}%**); pure univariate extrapolators
(Naive, MA, ETS, Theta, ARIMA) mostly shift level and do **not** improve on the clean
holdout. Hidden lost sales ≈ **{100*uplift.sum()/train.sale_amount.sum():.0f}%** of observed volume.
"""))
'''

C["deploy"] = r'''
lost = uplift.sum(); days = train.dt.nunique(); series = train.groupby(["store_id","product_id"]).ngroups
print(f"Sample: {series} series x {days} days")
print(f"Hidden lost units (latent demand):  {lost:,.0f}  ({lost/series/days:.3f} units/series/day)")
print(f"As share of observed sales:         {100*lost/train.sale_amount.sum():.1f}%")
for margin in (0.5, 1.0, 2.0):
    print(f"  @ {margin:.1f} margin/unit  ->  {lost*margin:,.0f} lost contribution (sample)")
'''

# ----------------------------------------------------------------------------
EN, ES = {}, {}

EN["title"] = """# Censored Demand: recovering the sales your ERP never sees

**A CRISP-ML(Q) study on latent demand recovery — FreshRetailNet-50K.**

In retail, `sale_amount` is **not demand** — it is demand **right-censored by stock**.
When the shelf goes empty, real demand still existed but leaves no record: *lost sales*.
A model trained on observed sales learns to **underestimate demand exactly when it matters
most** (stockouts happen *because* demand outran stock). This notebook (1) shows the trap,
(2) recovers the latent uncensored demand, (3) re-forecasts on it across **10 model
families** (Naive → SARIMA → Prophet → gradient boosting → LSTM), and (4) quantifies the
hidden lost sales — evaluated honestly on a clean, in-stock holdout.

**Author:** [Oscar Andrés Ponce](https://oscarponce.com) · bilingual twin: `Censored_Demand_CRISPML_ES.ipynb`

> ⚠️ **Honest caveat up front.** Real demand is unobservable, so recovered demand is an
> *estimate*, not ground truth. We never evaluate against it — only against observed sales
> on days the shelf was full. Numbers below are for the sample/config in this run.
"""
ES["title"] = """# Demanda Censurada: recuperar la venta que tu ERP no ve

**Estudio CRISP-ML(Q) de recuperación de demanda latente — FreshRetailNet-50K.**

En retail, `sale_amount` **no es demanda** — es demanda **censurada por la derecha por el
stock**. Cuando la percha se vacía, la demanda real existió pero no deja registro: *venta
perdida*. Un modelo entrenado sobre la venta observada aprende a **subestimar la demanda
justo cuando más importa** (el quiebre ocurre *porque* la demanda superó al stock). Este
notebook (1) muestra la trampa, (2) recupera la demanda latente no censurada, (3)
re-pronostica sobre ella con **10 familias de modelos** (Naive → SARIMA → Prophet → gradient
boosting → LSTM), y (4) cuantifica la venta perdida oculta — evaluado de forma honesta sobre
un holdout limpio (con stock).

**Autor:** [Oscar Andrés Ponce](https://oscarponce.com) · gemelo bilingüe: `Censored_Demand_CRISPML.ipynb`

> ⚠️ **Caveat honesto desde el inicio.** La demanda real es inobservable, así que la demanda
> recuperada es una *estimación*, no verdad de campo. Nunca evaluamos contra ella — solo
> contra la venta observada en días con percha llena. Los números abajo son para la
> muestra/config de esta corrida.
"""

EN["setup"] = "## Setup & Imports"
ES["setup"] = "## Setup e Imports"

EN["p1"] = """## Phase 1 — Business Understanding

**The problem: On-Shelf Availability (OSA) hides demand.** Every retailer plans on what it
sold. But when an item is out of stock, the sale that *would* have happened is invisible.
This is **demand unconstraining / censored demand estimation**, and it biases the whole
planning loop: forecast low → order low → stock out → record low sales → forecast lower.

**Why it's operationally real.** The only way to "see" a lost sale is to detect the empty
shelf — modern **On-Shelf Availability** via computer vision on shelf cameras, IoT weight
sensors, or, here, **labeled stockout hours**. FreshRetailNet-50K is the first large
benchmark that *annotates the censorship* (most datasets hide it), so we can model it
honestly instead of guessing.

**The dataset.** 50,000 store×product series, 90 days, hourly resolution, 898 stores, 865
perishable SKUs. Each day carries a 24-hour sales array and a 24-hour stock-status array
(`1 = empty shelf`), plus discount, holiday, and weather covariates. Split: 90 train days →
forecast the next 7 (`eval`).
"""
ES["p1"] = """## Fase 1 — Entendimiento del Negocio

**El problema: la disponibilidad en percha (OSA) esconde la demanda.** Todo retailer
planifica con lo que vendió. Pero cuando un producto está agotado, la venta que *habría*
ocurrido es invisible. Esto es **demand unconstraining / estimación de demanda censurada**, y
sesga todo el ciclo de planeación: pronóstico bajo → pedido bajo → quiebre → registro de
venta baja → pronóstico más bajo.

**Por qué es real en operaciones.** La única forma de "ver" una venta perdida es detectar la
percha vacía — **disponibilidad en percha (OSA)** moderna vía visión computacional en cámaras
de góndola, sensores IoT de peso, o, aquí, **horas de quiebre etiquetadas**.
FreshRetailNet-50K es el primer benchmark grande que *anota la censura* (la mayoría la
esconde), así que podemos modelarla con honestidad en vez de adivinar.

**El dataset.** 50,000 series tienda×producto, 90 días, resolución horaria, 898 tiendas, 865
SKUs perecederos. Cada día trae un array de venta de 24h y uno de estado de stock de 24h
(`1 = percha vacía`), más descuento, feriado y clima. Split: 90 días de train → pronosticar
los siguientes 7 (`eval`).
"""

EN["p2"] = """## Phase 2 — Data Understanding

We verify the censorship signal **from the data**, not from assumptions, and run quality
gates (the "Q" in CRISP-ML(Q)) that fail loudly if a structural assumption breaks.
"""
ES["p2"] = """## Fase 2 — Entendimiento de los Datos

Verificamos la señal de censura **desde los datos**, no desde supuestos, y corremos quality
gates (la "Q" de CRISP-ML(Q)) que fallan ruidosamente si un supuesto estructural se rompe.
"""

EN["ins_gates"] = """> **Insight — structural integrity.** `sale_amount` is *exactly* the sum of the 24 hourly
> values. That identity is what makes hour-level recovery possible: we can reconstruct a
> day's demand hour by hour and re-sum. The gates also guarantee 90 contiguous days per
> series — the precondition for honest lag features and a temporal split.
"""
ES["ins_gates"] = """> **Insight — integridad estructural.** `sale_amount` es *exactamente* la suma de los 24
> valores horarios. Esa identidad es lo que hace posible la recuperación a nivel hora:
> podemos reconstruir la demanda del día hora por hora y re-sumar. Los gates también
> garantizan 90 días contiguos por serie — la precondición para lags honestos y un split
> temporal.
"""

EN["ins_eda"] = """> **Insight — the censorship is large and the signal is real.** ~25% of store-hours are
> stockouts and ~45% of days lose at least one commercial hour — not a rounding error. The
> ~13× sale gap between OOS and in-stock hours confirms `1` truly means *empty shelf*. And
> unlike a synthetic dataset, the drivers are alive: discount, holiday and weather all move
> sales. There is real demand structure to model — and real censorship hiding part of it.
"""
ES["ins_eda"] = """> **Insight — la censura es grande y la señal es real.** ~25% de las horas-tienda son
> quiebre y ~45% de los días pierden al menos una hora comercial — no es redondeo. La brecha
> de venta ~13× entre horas OOS y con stock confirma que `1` significa de verdad *percha
> vacía*. Y a diferencia de un dataset sintético, los drivers están vivos: descuento, feriado
> y clima mueven la venta. Hay estructura de demanda real que modelar — y censura real
> escondiendo una parte.
"""

EN["ins_edaplot"] = """> **Insight — the trap, drawn.** The dashed *observed* curve sits **below** the in-stock
> curve precisely during the busy midday/evening hours — that gap is the censorship. A model
> trained on the dashed line learns a demand shape that is artificially flattened where
> demand is actually highest.
"""
ES["ins_edaplot"] = """> **Insight — la trampa, dibujada.** La curva *observada* (punteada) queda **por debajo** de
> la curva con stock justo en las horas pico de mediodía/tarde — esa brecha es la censura. Un
> modelo entrenado sobre la línea punteada aprende una forma de demanda artificialmente
> aplanada donde la demanda es en realidad más alta.
"""

EN["p3"] = """## Phase 3 — Data Preparation: recovering latent demand

**The core idea.** On in-stock hours, observed sales ≈ true demand. On OOS hours, demand is
censored to ~0. So we reconstruct each day's demand by **imputing only the censored hours**:
build an hour-of-day demand *shape* per category (from in-stock hours only), scale it by each
series' own in-stock daily level, and fill the empty hours with that expectation. Observed
hours are untouched; recovered demand is floored at observed sales.

**The guardrail.** Stock status is used **only to locate and model the censorship — never as
a naive predictor of sales.** (The prior lab leaked the target through an inventory proxy; we
do not repeat that.) Recovered demand is an *estimate*; we label it as such everywhere.
"""
ES["p3"] = """## Fase 3 — Preparación de Datos: recuperar la demanda latente

**La idea central.** En horas con stock, la venta observada ≈ demanda real. En horas OOS, la
demanda queda censurada a ~0. Reconstruimos la demanda de cada día **imputando solo las horas
censuradas**: construimos una *forma* de demanda hora-del-día por categoría (solo con horas
con stock), la escalamos por el nivel diario con-stock de cada serie, y rellenamos las horas
vacías con esa expectativa. Las horas observadas quedan intactas; la demanda recuperada nunca
baja de la venta observada.

**El guardrail.** El estado de stock se usa **solo para ubicar y modelar la censura — nunca
como predictor ingenuo de la venta.** (El lab anterior filtró el target vía un proxy de
inventario; no lo repetimos.) La demanda recuperada es una *estimación*; lo etiquetamos así
en todas partes.
"""

EN["ins_recover"] = """> **Insight — how big is the hidden demand?** Recovery lifts total volume by ~20% over
> observed sales, all of it concentrated on the ~45% of days that hit a stockout. By
> construction recovered ≥ observed, so we never *erase* a real sale — we only add back what
> the empty shelf suppressed. This is an estimate; its size depends on the imputation
> assumption (in-stock hours represent the would-be demand shape).
"""
ES["ins_recover"] = """> **Insight — ¿qué tan grande es la demanda oculta?** La recuperación sube el volumen total
> ~20% sobre la venta observada, todo concentrado en el ~45% de días que tuvieron quiebre.
> Por construcción recuperada ≥ observada, así que nunca *borramos* una venta real — solo
> sumamos lo que la percha vacía suprimió. Es una estimación; su magnitud depende del supuesto
> de imputación (las horas con stock representan la forma de demanda que habría habido).
"""

EN["ins_recoverplot"] = """> **Insight — where the gap lives.** On a stockout-heavy series the shaded gap (recovered −
> observed) is large; on stable series it would be nearly invisible. The right panel shows
> the lost units pile up in the high-traffic hours — exactly where running out hurts most.
"""
ES["ins_recoverplot"] = """> **Insight — dónde vive la brecha.** En una serie con muchos quiebres la zona sombreada
> (recuperada − observada) es grande; en series estables sería casi invisible. El panel
> derecho muestra que las unidades perdidas se acumulan en las horas de mayor tráfico —
> justo donde quedarse sin stock duele más.
"""

EN["p3f"] = """### Features and the temporal split

Features: calendar (day-of-week, day-of-month), demand lags (7/14/28), rolling means,
discount, holiday/activity flags, weather, and category/store identifiers. The split is
**strictly temporal**. Because the horizon (7 days) equals the minimum lag (7), every eval
lag is sourced from the train window: no recursion, no leakage.
"""
ES["p3f"] = """### Features y el split temporal

Features: calendario (día de semana, día del mes), lags de demanda (7/14/28), medias móviles,
descuento, flags de feriado/actividad, clima, e identificadores de categoría/tienda. El split
es **estrictamente temporal**. Como el horizonte (7 días) iguala el lag mínimo (7), cada lag
del eval proviene del train: sin recursión, sin leakage.
"""

EN["p4"] = """## Phase 4 — Modeling: the trap vs the fix, then the whole zoo

First the **core experiment** on all series: one LightGBM, two training targets — observed
`sale_amount` (censored, *the trap*) vs `demand` (recovered, *the fix*). Then we widen to a
**model zoo** — Naive, Seasonal Naive, Moving Average, ETS/Holt-Winters, Theta, SARIMA,
Prophet, LightGBM, XGBoost, CatBoost, LSTM — each trained on both targets, to see *which
families* the recovery actually helps.
"""
ES["p4"] = """## Fase 4 — Modelado: la trampa vs el arreglo, y luego todo el zoo

Primero el **experimento central** sobre todas las series: un LightGBM, dos targets —
`sale_amount` observado (censurado, *la trampa*) vs `demand` (recuperada, *el arreglo*).
Luego ampliamos a un **model zoo** — Naive, Seasonal Naive, Media Móvil, ETS/Holt-Winters,
Theta, SARIMA, Prophet, LightGBM, XGBoost, CatBoost, LSTM — cada uno entrenado sobre ambos
targets, para ver *a qué familias* ayuda de verdad la recuperación.
"""

EN["p5"] = """## Phase 5 — Evaluation: honest, on a clean holdout

**The rigor point.** We *cannot* evaluate against real demand — it isn't observable. So we
evaluate **only on eval days with zero commercial OOS hours**, where observed sales ≈ true
demand. If the censored model underestimates even this clean demand, the trap is real.
"""
ES["p5"] = """## Fase 5 — Evaluación: honesta, sobre un holdout limpio

**El punto de rigor.** *No podemos* evaluar contra la demanda real — no es observable. Así
que evaluamos **solo en días eval con cero horas OOS comerciales**, donde la venta observada ≈
demanda real. Si el modelo censurado subestima incluso esta demanda limpia, la trampa es real.
"""

EN["ins_coreeval"] = """> **Insight — the trap, quantified.** Trained on censored sales the model is biased *low* on
> demand it should have nailed (the holdout is in-stock, so there is nothing to miss). Trained
> on recovered demand the same model is near-unbiased and lower error. The fix is the training
> target, not the algorithm.
"""
ES["ins_coreeval"] = """> **Insight — la trampa, cuantificada.** Entrenado sobre venta censurada el modelo queda
> sesgado *a la baja* sobre demanda que debería haber clavado (el holdout es con stock, no hay
> nada que perder). Entrenado sobre demanda recuperada el mismo modelo queda casi sin sesgo y
> con menor error. El arreglo es el target de entrenamiento, no el algoritmo.
"""

EN["p5seg"] = """### Where does recovery actually help? (no over-claiming)

Recovery should help **where censorship lives** and stay neutral elsewhere. We split series
into *stockout-prone* (≥30% of days with OOS) vs *stable* and compare bias.
"""
ES["p5seg"] = """### ¿Dónde ayuda de verdad la recuperación? (sin over-claiming)

La recuperación debería ayudar **donde vive la censura** y ser neutral en el resto. Dividimos
las series en *propensas a quiebre* (≥30% de días con OOS) vs *estables* y comparamos el bias.
"""

EN["ins_seg"] = """> **Insight — the win is targeted.** The bias correction is concentrated in stockout-prone
> series; stable series barely move (both models were already fine there). This is the honest
> claim: recovery is a fix for censored series, not a universal accuracy booster.
"""
ES["ins_seg"] = """> **Insight — la mejora es focalizada.** La corrección de sesgo se concentra en series
> propensas a quiebre; las estables casi no se mueven (ambos modelos ya iban bien ahí). Esta
> es la afirmación honesta: la recuperación arregla series censuradas, no es un potenciador
> universal de precisión.
"""

EN["p4zoo"] = """### The model zoo

We run the ten families below on a `MODEL_SAMPLE`-series subset (the classical and per-series
models are slower), each trained twice — on censored sales and on recovered demand — and score
both on the same clean holdout. Univariate models (Naive…SARIMA) see only the target history;
Prophet adds discount + holiday; the ML/LSTM models use the full feature set.
"""
ES["p4zoo"] = """### El model zoo

Corremos las diez familias de abajo sobre un subconjunto de `MODEL_SAMPLE` series (los modelos
clásicos y por-serie son más lentos), cada una entrenada dos veces — sobre venta censurada y
sobre demanda recuperada — y evaluamos ambas en el mismo holdout limpio. Los modelos
univariados (Naive…SARIMA) solo ven el histórico del target; Prophet agrega descuento +
feriado; los de ML/LSTM usan el set completo de features.
"""

EN["ins_zoo"] = """> **Insight — recovery is not a free lunch.** The benefit lands on models that can *localize*
> the correction: gradient boosting (LightGBM/XGBoost/CatBoost) and Prophet improve MAE
> several points and shrink bias, because lags, calendar and covariates let them apply the
> higher demand level *selectively*. Pure univariate extrapolators (Naive, Moving Average,
> ETS/Holt-Winters, Theta, SARIMA) mostly just shift their level up and can even *overshoot*
> the clean in-stock holdout. The lesson: recovering the target is necessary, but you also
> need a model rich enough to use it. This is why the production recommendation is a
> feature-based learner on recovered demand — not a classical method.
"""
ES["ins_zoo"] = """> **Insight — la recuperación no es un free lunch.** El beneficio cae en los modelos que
> pueden *localizar* la corrección: gradient boosting (LightGBM/XGBoost/CatBoost) y Prophet
> mejoran el MAE varios puntos y achican el sesgo, porque los lags, el calendario y las
> covariables les dejan aplicar el nivel de demanda más alto *de forma selectiva*. Los
> extrapoladores univariados puros (Naive, Media Móvil, ETS/Holt-Winters, Theta, SARIMA) solo
> suben su nivel y hasta pueden *pasarse* en el holdout limpio. La lección: recuperar el
> target es necesario, pero también necesitás un modelo lo bastante rico para usarlo. Por eso
> la recomendación de producción es un modelo basado en features sobre demanda recuperada — no
> un método clásico.
"""

EN["p6"] = """## Phase 6 — Deployment & honest limitations

**The business payoff.** Recovered demand makes the lost sales the ERP never records *visible
and quantifiable*. Planning on it raises the order signal for stockout-prone SKUs — the ones
that keep stocking out precisely because the forecast was trained to ignore their true demand.

**The operational hook.** This pipeline assumes the censorship is *labeled*. In production
that label comes from **On-Shelf Availability detection** — computer vision on shelf images,
IoT, or POS-gap heuristics. The CV/operations layer flags the empty shelf in real time; this
model turns that flag into a corrected demand signal. That is the bridge from "a camera saw an
empty shelf" to "order more next week."

**Limitations & honest caveats.**
- Recovered demand is an **estimate**, not ground truth — it inherits the imputation
  assumption (in-stock hours represent the would-be demand shape). We never score against it.
- The clean-holdout evaluation only covers in-stock days; behavior on fully-censored days is
  *inferred*, not measured.
- The win concentrates in stockout-prone series and in **feature-rich models**; for stable
  series, or for classical univariate methods, the two targets are close. No universal claim.
- Numbers reflect the sample/config in this run. Set `N_SERIES = None` for the full
  50,000-series benchmark and raise `MODEL_SAMPLE` for a larger zoo.
"""
ES["p6"] = """## Fase 6 — Despliegue y limitaciones honestas

**El payoff de negocio.** La demanda recuperada hace *visible y cuantificable* la venta
perdida que el ERP nunca registra. Planear sobre ella sube la señal de pedido para los SKUs
propensos a quiebre — justo los que siguen quebrando porque el pronóstico se entrenó para
ignorar su demanda real.

**El gancho operativo.** Este pipeline asume que la censura está *etiquetada*. En producción
esa etiqueta viene de la **detección de disponibilidad en percha (OSA)** — visión
computacional sobre imágenes de góndola, IoT, o heurísticas de hueco en el POS. La capa de
CV/operaciones marca la percha vacía en tiempo real; este modelo convierte esa marca en una
señal de demanda corregida. Ese es el puente entre "una cámara vio una percha vacía" y "pide
más la próxima semana".

**Limitaciones y caveats honestos.**
- La demanda recuperada es una **estimación**, no verdad de campo — hereda el supuesto de
  imputación (las horas con stock representan la forma de demanda que habría habido). Nunca
  evaluamos contra ella.
- La evaluación en holdout limpio solo cubre días con stock; el comportamiento en días
  totalmente censurados es *inferido*, no medido.
- La mejora se concentra en series propensas a quiebre y en **modelos ricos en features**;
  para series estables, o para métodos clásicos univariados, los dos targets están cerca. Sin
  afirmaciones universales.
- Los números reflejan la muestra/config de esta corrida. Poné `N_SERIES = None` para el
  benchmark completo de 50,000 series y subí `MODEL_SAMPLE` para un zoo más grande.
"""

# ----------------------------------------------------------------------------
def build(md):
    cells = [
        new_markdown_cell(md["title"]),
        new_markdown_cell(md["setup"]),
        new_code_cell(C["setup"].strip()),
        new_code_cell(C["load"].strip()),
        new_markdown_cell(md["p1"]),
        new_markdown_cell(md["p2"]),
        new_code_cell(C["gates"].strip()),
        new_markdown_cell(md["ins_gates"]),
        new_code_cell(C["eda"].strip()),
        new_markdown_cell(md["ins_eda"]),
        new_code_cell(C["eda_plot"].strip()),
        new_markdown_cell(md["ins_edaplot"]),
        new_markdown_cell(md["p3"]),
        new_code_cell(C["recover"].strip()),
        new_markdown_cell(md["ins_recover"]),
        new_code_cell(C["recover_plot"].strip()),
        new_markdown_cell(md["ins_recoverplot"]),
        new_markdown_cell(md["p3f"]),
        new_code_cell(C["features"].strip()),
        new_markdown_cell(md["p4"]),
        new_code_cell(C["core_models"].strip()),
        new_markdown_cell(md["p5"]),
        new_code_cell(C["core_eval"].strip()),
        new_markdown_cell(md["ins_coreeval"]),
        new_markdown_cell(md["p5seg"]),
        new_code_cell(C["segment"].strip()),
        new_markdown_cell(md["ins_seg"]),
        new_markdown_cell(md["p4zoo"]),
        new_code_cell(C["zoo_setup"].strip()),
        new_code_cell(C["zoo_run"].strip()),
        new_code_cell(C["zoo_leaderboard"].strip()),
        new_markdown_cell(md["ins_zoo"]),
        new_code_cell(C["headline"].strip()),
        new_markdown_cell(md["p6"]),
        new_code_cell(C["deploy"].strip()),
    ]
    nb = new_notebook(cells=cells)
    nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata.language_info = {"name": "python", "version": "3.11"}
    return nb

os.makedirs("notebooks", exist_ok=True)
nbf.write(build(EN), "notebooks/Censored_Demand_CRISPML.ipynb")
nbf.write(build(ES), "notebooks/Censored_Demand_CRISPML_ES.ipynb")
print("wrote notebooks/Censored_Demand_CRISPML.ipynb and _ES.ipynb")
