"""Genera el hero chart para LinkedIn (1200x627). Cuenta la tesis: demanda real
(horas con stock) vs venta observada (censurada); el area roja = demanda invisible.
"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

COMM = slice(6, 22)
# Dataset completo (cache local) para que la cifra del callout sea la headline real.
BASE = ("/Users/oscarponce/.cache/huggingface/hub/datasets--Dingdong-Inc--"
        "FreshRetailNet-50K/snapshots/08c1fab7f9257bc73679d415d65d644165d351d4/data/train.parquet")
try:
    df = pd.read_parquet(BASE)
except FileNotFoundError:
    df = pd.read_parquet("data/dev_train.parquet")
hs = np.vstack(df["hours_sale"].values).astype(float)
st = np.vstack(df["hours_stock_status"].values).astype(int)
oos = st == 1

instock_profile = np.nanmean(np.where(oos, np.nan, hs), 0)   # demanda real (horas con stock)
observed_profile = hs.mean(0)                                 # venta observada (todas las horas)
hours = np.arange(24)

# stat headline = venta perdida recuperada / venta observada (mismo metodo que el notebook)
instk = st == 0
cat = df["second_category_id"].values
shape = np.zeros((len(df), 24))
for c in np.unique(cat):
    m = cat == c
    shape[m] = np.nan_to_num(np.nanmean(np.where(instk[m], hs[m], np.nan), 0))
w = shape.copy(); w[:, :6] = 0; w[:, 22:] = 0
s = w.sum(1, keepdims=True); s[s == 0] = 1; w = w / s
full = instk[:, COMM].all(1)
lvl = (df.assign(_f=full, _s=df.sale_amount)[full]
         .groupby(["store_id","product_id"])["_s"].mean())
sl = np.array([lvl.get(k, np.nan) for k in zip(df.store_id, df.product_id)])
sl = np.where(np.isnan(sl), df.sale_amount.values, sl)
rec = np.where(instk, hs, sl[:, None] * w)
demand = np.maximum(rec[:, COMM].sum(1) + hs[:, :6].sum(1) + hs[:, 22:].sum(1), df.sale_amount.values)
share = (demand - df.sale_amount.values).sum() / df.sale_amount.sum()

# ---- estilo ----
INK="#14202e"; SUB="#5b6b7a"; TRUE="#0e9f6e"; OBS="#94a3b8"; GAP="#e23d4b"; BG="#ffffff"
plt.rcParams.update({"font.family":"DejaVu Sans","axes.edgecolor":"#d7dde3"})
fig, ax = plt.subplots(figsize=(12, 6.27), dpi=100); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
fig.subplots_adjust(left=0.07, right=0.97, top=0.74, bottom=0.13)

ax.fill_between(hours[6:22], observed_profile[6:22], instock_profile[6:22],
                color=GAP, alpha=0.16, zorder=1)
ax.plot(hours, instock_profile, color=TRUE, lw=3.2, zorder=3,
        label="Demanda real  (horas con stock)")
ax.plot(hours, observed_profile, color=OBS, lw=2.6, ls=(0,(5,2)), zorder=2,
        label="Venta observada  (censurada por quiebres)")

ax.set_xlim(0, 23); ax.set_ylim(0, instock_profile.max()*1.18)
ax.set_xticks(range(0, 24, 2)); ax.set_xticklabels([f"{h:02d}h" for h in range(0,24,2)], fontsize=11, color=SUB)
ax.set_yticks([]);
for s in ["top","right","left"]: ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color("#d7dde3")
ax.tick_params(length=0)
ax.grid(axis="x", color="#eef1f4", lw=1)
ax.legend(loc="upper left", frameon=False, fontsize=12.5, labelcolor=INK,
          bbox_to_anchor=(0.005, 1.005))

# flecha + callout sobre la brecha
peak = 6 + int(np.argmax((instock_profile[6:22]-observed_profile[6:22])))
ymid = (instock_profile[peak]+observed_profile[peak])/2
ax.annotate("La demanda que\ntu ERP no ve",
            xy=(peak, ymid), xytext=(peak+3.4, ymid*0.62),
            color=GAP, fontsize=14, fontweight="bold", ha="left", va="center",
            arrowprops=dict(arrowstyle="-|>", color=GAP, lw=2))

# ---- titulares (sobre el area del axes) ----
fig.text(0.07, 0.93, "La venta que tu ERP no ve", fontsize=31, fontweight="bold", color=INK)
fig.text(0.07, 0.862, "Demanda censurada por quiebres de stock",
         fontsize=14, color=SUB)
fig.text(0.07, 0.815, "FreshRetailNet-50K · 50.000 series · 90 días · resolución por hora",
         fontsize=11.5, color=SUB)
fig.text(0.97, 0.945, f"≈{share*100:.0f}%", fontsize=40, fontweight="bold", color=GAP, ha="right", va="top")
fig.text(0.97, 0.845, "de la demanda queda\ninvisible en el quiebre",
         fontsize=12, color=SUB, ha="right", va="top")

fig.text(0.07, 0.035, "Oscar Ponce · oscarponce.com", fontsize=11, color=SUB)
fig.text(0.97, 0.035, "CRISP-ML(Q)  ·  recuperación de demanda latente", fontsize=11, color=SUB, ha="right")

import os; os.makedirs("assets", exist_ok=True)
fig.savefig("assets/hero_censored_demand.png", dpi=100, facecolor=BG)
print("saved assets/hero_censored_demand.png  (1200x627)")
print(f"stat: lost share at commercial hours = {share:.3f}")
