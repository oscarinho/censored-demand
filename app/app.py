"""
Censored Demand — la venta que el sistema no ve
Demo interactiva: hace visible la demanda perdida por quiebres de stock.
Dataset: FreshRetailNet-50K (muestra de 20k días tienda×producto, datos horarios).
Run local:  streamlit run app/app.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title="Censored Demand — la venta que el sistema no ve",
                   page_icon="📦", layout="wide")

GOLD, RED, GREEN, MUTE = "#C9A86A", "#D96B5F", "#5FB07A", "#9aa0aa"

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando datos…")
def load_data() -> pd.DataFrame:
    here = Path(__file__).resolve()
    cands = [here.parent.parent / "data" / "freshretailnet_train_sample.parquet",
             Path("data/freshretailnet_train_sample.parquet"),
             here.parent / "freshretailnet_train_sample.parquet"]
    for c in cands:
        if c.exists():
            return pd.read_parquet(c)
    st.error("No encuentro `data/freshretailnet_train_sample.parquet`. "
             "Ejecuta `python scripts/get_and_profile.py` primero.")
    st.stop()

def recover(hsale, hstock):
    """Imputa las horas censuradas (stockout) con la tasa media de venta de las horas con stock."""
    hs = np.asarray(hsale, dtype=float)
    ss = np.asarray(hstock, dtype=int)
    instock = ss == 0
    rate = hs[instock].mean() if instock.any() else 0.0
    rec = np.where(ss == 1, rate, hs)
    return hs, rec, int((ss == 1).sum())

@st.cache_data(show_spinner="Reconstruyendo demanda latente…")
def enrich(df: pd.DataFrame) -> pd.DataFrame:
    obs, rec, oos = [], [], []
    for hs, ss in zip(df["hours_sale"], df["hours_stock_status"]):
        h, r, n = recover(hs, ss)
        obs.append(float(h.sum())); rec.append(float(r.sum())); oos.append(n)
    out = df.copy()
    out["obs_daily"], out["rec_daily"], out["oos_hours"] = obs, rec, oos
    out["lost"] = out["rec_daily"] - out["obs_daily"]
    return out

df = load_data()
d = enrich(df)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("### 📦 Censored Demand — la venta que el sistema no ve")
st.caption("En retail, `unidades vendidas` ≠ demanda. Cuando la percha se vacía, la venta es 0 — "
           "pero la demanda existió. Esta demo la hace visible. "
           "Caso completo: [oscarponce.com/labs/censored-demand](https://oscarponce.com/labs/censored-demand)")

# ── KPIs del benchmark completo (notebook, 4.5M series) ───────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Días con quiebre", "44%", help="Días tienda×producto con ≥1 hora sin stock (benchmark completo)")
k2.metric("Venta perdida oculta", "≈19%", help="Sobre el volumen observado (benchmark completo)")
k3.metric("Unidades invisibles", "866,515", help="Demanda estimada que el registro nunca vio")
k4.metric("Sesgo del pronóstico", "−0.13 → −0.01", help="Entrenar sobre venta observada vs demanda recuperada")
st.caption("KPIs del **benchmark completo** (4.5M series, 10 familias de modelos) — ver el notebook. "
           "Abajo puedes explorar series individuales de una muestra para ver el método de recuperación en acción.")
st.divider()

# ── Series explorer ───────────────────────────────────────────────────────────
left, right = st.columns([1, 2.2], gap="large")

with left:
    st.markdown("**Explora una serie (tienda × producto × día)**")
    only_oos = st.toggle("Solo días con quiebre", value=True)
    pool = d[d["oos_hours"] > 0] if only_oos else d
    min_lost = st.slider("Mínimo de venta perdida (unidades)", 0, int(max(1, d["lost"].max())), 0)
    pool = pool[pool["lost"] >= min_lost]
    if pool.empty:
        st.warning("No hay series con esos filtros."); st.stop()
    st.caption(f"{len(pool):,} series candidatas")
    idx = st.selectbox(
        "Elige una serie",
        options=pool.index[:500],
        format_func=lambda i: f"Tienda {d.at[i,'store_id']} · Prod {d.at[i,'product_id']} · {d.at[i,'dt']} "
                              f"· {d.at[i,'oos_hours']}h sin stock",
    )

row = d.loc[idx]
hs, rec, n_oos = recover(row["hours_sale"], row["hours_stock_status"])
ss = np.asarray(row["hours_stock_status"], dtype=int)
hours = list(range(24))

with right:
    fig = go.Figure()
    # observado
    fig.add_trace(go.Bar(x=hours, y=hs, name="Venta observada", marker_color=GREEN))
    # demanda recuperada en horas censuradas (la parte invisible)
    imputed = np.where(ss == 1, rec, 0)
    fig.add_trace(go.Bar(x=hours, y=imputed, name="Demanda recuperada (censurada)", marker_color=RED))
    # sombrear horas de stockout
    for h in hours:
        if ss[h] == 1:
            fig.add_vrect(x0=h-0.5, x1=h+0.5, fillcolor=RED, opacity=0.06, line_width=0)
    fig.update_layout(
        barmode="stack", height=380, margin=dict(t=30, b=30, l=10, r=10),
        title="Venta por hora — verde = registrada · rojo = perdida por percha vacía",
        xaxis_title="Hora del día", yaxis_title="Unidades", legend=dict(orientation="h"),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Venta registrada (día)", f"{hs.sum():.1f}")
    c2.metric("Demanda real estimada", f"{rec.sum():.1f}")
    uplift = (rec.sum() - hs.sum()) / max(hs.sum(), 1e-9) * 100
    c3.metric("Venta perdida", f"+{rec.sum()-hs.sum():.1f}", f"{uplift:.0f}% más demanda")

st.info("⚠️ El método mostrado acá (imputar las horas sin stock con la tasa de las horas con stock) es una "
        "**ilustración simple** del concepto. La recuperación rigurosa del lab usa 10 familias de modelos y baja "
        "el sesgo del pronóstico de −0.13 a −0.01. La demanda real es **inobservable** — la recuperada siempre es "
        "una estimación, nunca se evalúa contra la verdad. Ese es el límite honesto del método.")
