"""Build the LinkedIn picture (French) and the social-preview image from the live baseline.

    uv run --with matplotlib python marketing/make_images.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from belsim.core import engine

ROOT = Path(__file__).resolve().parents[1]
BLACK, YELLOW, RED, PAPER, INK, MUTED = "#111111", "#FDDA24", "#EF3340", "#FAF7F0", "#151515", "#6B6B6B"

font = next((f for f in ("Archivo", "Helvetica Neue", "Arial", "DejaVu Sans")
             if any(f.lower() in x.name.lower() for x in font_manager.fontManager.ttflist)), "DejaVu Sans")
plt.rcParams["font.family"] = font

res = engine.run(None)
years = res.years
debt = [s.fiscal["debt_pct_gdp"] for s in res.states]
bal = [s.fiscal["balance_pct_gdp"] for s in res.states]
interest = [s.fiscal["interest_pct_gdp"] for s in res.states]
d0, d1 = debt[0], debt[-1]


def fr(x, nd=0):
    return f"{x:.{nd}f}".replace(".", ",")


def chart(ax, label_60, lw=7):
    ax.set_facecolor(PAPER)
    ax.fill_between(years, debt, 0, color=RED, alpha=0.10, linewidth=0)
    ax.plot(years, debt, color=RED, linewidth=lw, solid_capstyle="round")
    ax.axhline(60, color=BLACK, linewidth=1.6, linestyle=(0, (4, 4)))
    ax.text(years[-1], 66, label_60, ha="right", va="bottom", fontsize=15, color=BLACK)
    ax.scatter([years[0], years[-1]], [d0, d1], s=260, color=RED, zorder=5, edgecolor=PAPER, linewidth=3)
    ax.set_xlim(years[0] - 1, years[-1] + 1)
    ax.set_ylim(0, max(debt) * 1.12)
    ax.set_xticks([2024, 2030, 2040, 2050, 2060, 2070])
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=17, colors=MUTED, length=0, pad=8)
    for sp in ax.spines.values():
        sp.set_visible(False)


def flag(fig, y0=0.0, h=0.022):
    for i, c in enumerate((BLACK, YELLOW, RED)):
        fig.patches.append(plt.Rectangle((i / 3, y0), 1 / 3, h, transform=fig.transFigure, color=c, zorder=10))


# ---- LinkedIn, French, 1200 x 1200 ----
fig = plt.figure(figsize=(12, 12), dpi=100, facecolor=PAPER)
flag(fig, 0.978)
fig.text(0.06, 0.905, "Sans rien changer,", fontsize=50, weight="bold", color=INK)
fig.text(0.06, 0.835, "la dette belge triplerait presque.", fontsize=50, weight="bold", color=RED)
fig.text(0.06, 0.785, f"Dette publique en % du PIB, à politique inchangée : {fr(d0)} % en 2024, {fr(d1)} % en 2070.",
         fontsize=19, color=INK)
ax = fig.add_axes([0.06, 0.25, 0.88, 0.50])
chart(ax, "Norme européenne : 60 %")
ax.text(years[0] + 0.6, d0 + 22, f"{fr(d0)} %", fontsize=40, weight="bold", color=INK, ha="left")
ax.text(years[-1] - 0.6, d1 - 78, f"{fr(d1)} %", fontsize=56, weight="bold", color=RED, ha="right", va="top")
fig.text(0.06, 0.165, "Et vous, vous feriez quoi ?", fontsize=38, weight="bold", color=INK)
fig.text(0.06, 0.115, "Un simulateur du budget belge : toutes les recettes, toutes les dépenses,",
         fontsize=19, color=INK)
fig.text(0.06, 0.083, "les 3 Régions, les programmes de 12 partis. Testez vos propres mesures.", fontsize=19, color=INK)
fig.text(0.06, 0.035, "Projection d'un modèle exploratoire, calé sur les comptes 2024-2025 et les perspectives officielles. "
         "Pas un chiffrage officiel.", fontsize=13, color=MUTED)
out = ROOT / "marketing" / "linkedin_fr.png"
fig.savefig(out, dpi=100, facecolor=PAPER)
plt.close(fig)

# ---- social preview, 1200 x 630 ----
fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=PAPER)
flag(fig, 0.962, 0.038)
fig.text(0.05, 0.80, "Belgium's budget, simulated", fontsize=44, weight="bold", color=INK)
fig.text(0.05, 0.71, "2024–2070  ·  FR · NL · EN", fontsize=22, color=MUTED)
ax = fig.add_axes([0.05, 0.21, 0.9, 0.45])
chart(ax, "60 %", lw=6)
ax.text(years[0] + 0.6, d0 + 30, f"{d0:.0f} %", fontsize=30, weight="bold", color=INK, ha="left")
ax.text(years[-1] - 0.6, d1 - 95, f"{d1:.0f} %", fontsize=40, weight="bold", color=RED, ha="right", va="top")
fig.text(0.05, 0.045, "Public debt, % of GDP, no policy change. Exploratory model, not an official costing.",
         fontsize=14, color=MUTED)
fig.savefig(ROOT / "web" / "og.png", dpi=100, facecolor=PAPER)
plt.close(fig)

print(f"debt {d0:.1f} -> {d1:.1f} % of GDP; balance {bal[0]:.1f} -> {bal[-1]:.1f}; interest {interest[0]:.1f} -> {interest[-1]:.1f}")
print(out)
