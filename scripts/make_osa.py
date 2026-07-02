"""Segundo asset LinkedIn (1200x627), mismo estilo que el hero.
Traduce la venta perdida (~19% de la observada) a disponibilidad real (OSA ~84%).
OSA = observada / (observada + perdida) = 1 / (1 + share).
Usa las cifras canónicas publicadas (share=0.19) para ser consistente con el hero/sitio."""
import numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

SHARE = 0.19                      # venta perdida / observada (headline del hero/sitio)
OSA   = 1/(1+SHARE)               # disponibilidad real
GAP   = 1-OSA                     # brecha de servicio (~16%)

INK="#14202e"; SUB="#5b6b7a"; TRUE="#0e9f6e"; OBSRED="#e23d4b"; BG="#ffffff"
plt.rcParams.update({"font.family":"DejaVu Sans"})
fig=plt.figure(figsize=(12,6.27),dpi=100); fig.patch.set_facecolor(BG)
ax=fig.add_axes([0.07,0.27,0.90,0.26]); ax.set_facecolor(BG)
ax.set_xlim(0,100); ax.set_ylim(0,1); ax.axis("off")

osa_pct=OSA*100; gap_pct=GAP*100
# barra: capturado (verde) + invisible (rojo)
ax.add_patch(plt.Rectangle((0,0),osa_pct,1,fc=TRUE,ec="none"))
ax.add_patch(plt.Rectangle((osa_pct,0),gap_pct,1,fc=OBSRED,ec="none"))
# contorno del 100% = demanda real del cliente
ax.add_patch(plt.Rectangle((0,0),100,1,fill=False,ec="#cfd6dd",lw=1.6))
ax.plot([100,100],[-0.15,1.25],color="#cfd6dd",lw=1.4)

# etiquetas dentro/abajo
ax.text(osa_pct/2,0.5,f"Lo que alcanzas a vender · {osa_pct:.0f}%",ha="center",va="center",
        color="white",fontsize=13.5,fontweight="bold")
ax.text(osa_pct+gap_pct/2,0.5,f"Invisible\npercha vacía\n{gap_pct:.0f}%",ha="center",va="center",
        color="white",fontsize=10.5,fontweight="bold",linespacing=1.25)
ax.text(0,-0.55,"0%",ha="left",va="top",color=SUB,fontsize=11)
ax.text(100,-0.55,"Demanda real del cliente · 100%",ha="right",va="top",color=SUB,fontsize=11)

# titulares (mismo tratamiento que el hero)
fig.text(0.07,0.93,"Tu disponibilidad real\nno es la que crees",fontsize=27,fontweight="bold",color=INK,va="top",linespacing=1.05)
fig.text(0.07,0.72,"Cuando ~19% de la demanda se pierde en el quiebre, operas —en volumen— a",
         fontsize=13.5,color=SUB)
fig.text(0.07,0.682,"~84% de disponibilidad real, aunque el ERP reporte 100%.",
         fontsize=13.5,color=SUB)
fig.text(0.07,0.625,"FreshRetailNet-50K · 50.000 series · 90 días · resolución por hora",
         fontsize=11,color=SUB)

fig.text(0.97,0.93,f"≈{osa_pct:.0f}%",fontsize=38,fontweight="bold",color=INK,ha="right",va="top")
fig.text(0.97,0.83,"disponibilidad real\n(OSA potencial)",fontsize=12,color=SUB,ha="right",va="top")

fig.text(0.07,0.06,"Oscar Ponce · oscarponce.com",fontsize=11,color=SUB)
fig.text(0.97,0.06,"CRISP-ML(Q)  ·  recuperación de demanda latente",fontsize=11,color=SUB,ha="right")

fig.savefig("assets/osa_gap_censored_demand.png",dpi=100,facecolor=BG)
print(f"saved assets/osa_gap_censored_demand.png  OSA={osa_pct:.1f}%  gap={gap_pct:.1f}%")
