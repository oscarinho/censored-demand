"""
Descarga FreshRetailNet-50K y guarda una muestra + un perfil inicial.
Corre donde haya internet (tu Mac / Claude Code), desde labs/censored-demand/:
    pip install datasets pandas pyarrow
    python scripts/get_and_profile.py
Deja:
    data/freshretailnet_train_sample.parquet   (muestra para que Claude la lea)
    data/PROFILE.txt                            (resumen impreso)
"""
import os, sys, json
import pandas as pd
import numpy as np

os.makedirs("data", exist_ok=True)
out = open("data/PROFILE.txt", "w")
def log(*a):
    line = " ".join(str(x) for x in a)
    print(line); out.write(line + "\n")

try:
    from datasets import load_dataset
except ImportError:
    sys.exit("Falta 'datasets'. Corre: pip install datasets pandas pyarrow")

log("Descargando Dingdong-Inc/FreshRetailNet-50K ...")
ds = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
log("Splits:", {k: len(v) for k, v in ds.items()})

split = "train" if "train" in ds else list(ds.keys())[0]
df = ds[split].to_pandas()
log("\n=== SHAPE ===", df.shape)
log("\n=== COLUMNAS / DTYPES ===")
for c in df.columns:
    log(f"  {c:24s} {str(df[c].dtype):12s} ej: {repr(df[c].iloc[0])[:60]}")

# Guardar muestra manejable para análisis posterior
sample = df.sample(min(20000, len(df)), random_state=42)
sample.to_parquet("data/freshretailnet_train_sample.parquet")
log("\nMuestra guardada:", sample.shape, "-> data/freshretailnet_train_sample.parquet")

# Profiling de censura / stockout — defensivo (busca columnas por nombre)
log("\n=== SEÑALES DE CENSURA / STOCKOUT ===")
cols = {c.lower(): c for c in df.columns}
def find(*keys):
    return [orig for low, orig in cols.items() if any(k in low for k in keys)]

for label, keys in [
    ("stock/stockout", ("stock", "oos", "out_of")),
    ("venta/sale",     ("sale", "sold", "qty", "demand")),
    ("descuento",      ("disc",)),
    ("clima",          ("temp", "humid", "prec", "weather", "rain")),
    ("fecha",          ("date", "dt", "day", "hour")),
]:
    hits = find(*keys)
    log(f"  {label:16s}: {hits}")

# Si hay columnas tipo lista (24h), reportarlo
list_cols = [c for c in df.columns if isinstance(df[c].iloc[0], (list, np.ndarray))]
log("\nColumnas tipo lista (probable serie horaria 24h):", list_cols)

out.close()
print("\nListo. Revisa data/PROFILE.txt y data/freshretailnet_train_sample.parquet")
