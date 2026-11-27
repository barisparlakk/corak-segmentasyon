"""Day 3, step 1: VHI baseline and how well VHI agrees with SPEI-3.

Baselines (scored with s13_eval, identical cell-months for every method):
  always none   - predicts "none" everywhere (what doing nothing scores)
  VHI Kogan     - clipped VHI with the fixed Kogan cut-offs 20/30/40 (the literature rule)
  VHI tuned     - the same VHI with the 3 cut-offs re-fitted to SPEI-3 on the TRAIN set only
                  (grid, step 2.5, macro F1). Shows how much is the index vs the cut-offs.
Agreement (TRAINING YEARS and TRAINING CELLS only, so feature decisions never see 2021-2022):
  - correlation of cell-month mean VHI with SPEI-3, overall, by month and by stratum
  - lag: VHI at month t against SPEI-3 at t+1, t, t-1, t-2 (does vegetation answer late?)
  - season means per year (between-year signal)

Outputs: outputs/tables/baseline_scores.csv, baseline_confusion.txt, baseline_strata.csv,
         vhi_spei_agreement.txt, data/preds/baseline_cell_month.csv,
         outputs/figures/vhi_spei_agreement.png
"""
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from config import CLIMATE_DIR, DATA, FIG_DIR, ROOT
from s13_eval import MIN_PIX, NAMES, confusion, load_rows, majority, metrics, score_table, stratum_table, vote

TAB = ROOT / "outputs" / "tables"
PRED = DATA / "preds"
TAB.mkdir(parents=True, exist_ok=True)
PRED.mkdir(parents=True, exist_ok=True)

rows = load_rows()
rows["vhi_kogan"] = np.digitize(rows.VHI_clip, [20, 30, 40])
rows["always_none"] = 3

# ---- tuned cut-offs, train set only
tr = rows[rows.set == "train"]
g = tr.groupby(["cell_id", "year", "month"], sort=True)
gid, ng = g.ngroup().values, g.ngroups
y_cm = g.cls.first().values
grid = np.arange(10, 62.5, 2.5)
best = (-1, None)
for cuts in itertools.combinations(grid, 3):
    p, n = vote(gid, ng, np.digitize(tr.VHI_clip.values, cuts))
    ok = n >= MIN_PIX
    f1 = metrics(y_cm[ok], p[ok])["macro_F1"]
    if f1 > best[0]:
        best = (f1, cuts)
TUNED = best[1]
print(f"tuned VHI cut-offs (train set, macro F1 {best[0]:.3f}): {TUNED}")
rows["vhi_tuned"] = np.digitize(rows.VHI_clip, TUNED)

# ---- scores
tables, cms = [], {}
for col, name in [("always_none", "always none"), ("vhi_kogan", "VHI Kogan"), ("vhi_tuned", "VHI tuned")]:
    cm = majority(rows, col)
    cms[name] = cm
    tables.append(score_table(cm, name))
scores = pd.concat(tables, ignore_index=True)
scores.to_csv(TAB / "baseline_scores.csv", index=False)
pd.set_option("display.width", 200)
print(scores[["method", "set", "n_cell_months", "n_year_months", "macro_F1", "F1_lo", "F1_hi", "QWK", "QWK_lo",
              "QWK_hi", "accuracy"] + [f"recall_{n}" for n in NAMES]].round(3).to_string(index=False))

with open(TAB / "baseline_confusion.txt", "w") as fh:
    for name in ["VHI Kogan", "VHI tuned"]:
        for s in ["temporal test", "strict test"]:
            txt = f"{name} - {s}\n{confusion(cms[name], s).to_string()}\n\n"
            fh.write(txt)
            print(txt)
strata = pd.concat([stratum_table(rows, "vhi_kogan", "VHI Kogan"), stratum_table(rows, "vhi_tuned", "VHI tuned")])
strata.to_csv(TAB / "baseline_strata.csv", index=False)
print(strata[["method", "stratum", "set", "n_cell_months", "macro_F1", "QWK"]].round(3).to_string(index=False))

out = cms["VHI Kogan"].rename(columns={"pred": "pred_kogan"})
out["pred_tuned"] = cms["VHI tuned"].pred.values
out.to_csv(PRED / "baseline_cell_month.csv", index=False)

# ---- agreement, training years and training cells only
a = rows[rows.set == "train"]
cmv = a.groupby(["cell_id", "year", "month"]).agg(VHI=("VHI_clip", "mean"), spei3=("spei3", "first")).reset_index()
lines = [f"cell-month mean VHI vs SPEI-3, training years x training cells (n = {len(cmv)})",
         f"  overall r = {cmv.VHI.corr(cmv.spei3):.3f}"]
lines.append("  by month: " + str(cmv.groupby("month").apply(lambda d: round(d.VHI.corr(d.spei3), 3)).to_dict()))
st = a.groupby(["stratum", "cell_id", "year", "month"]).agg(VHI=("VHI_clip", "mean"), spei3=("spei3", "first"))
lines.append("  by stratum: " + str(st.groupby("stratum").apply(lambda d: round(d.VHI.corr(d.spei3), 3)).to_dict()))

sp = xr.open_dataset(CLIMATE_DIR / "spei3.nc")
cells = pd.read_csv(CLIMATE_DIR / "cells_split.csv")
t = sp.time.to_index()
S = sp.spei3.values[:, cells.iy, cells.ix]
full = pd.DataFrame({"cell_id": np.tile(cells.cell_id.values, len(t)), "ym": np.repeat(t.year * 12 + t.month - 1, len(cells)),
                     "s": S.ravel()})
cmv["ym"] = cmv.year * 12 + cmv.month - 1
lag_r = {}
for k in [-1, 0, 1, 2]:                              # k > 0: SPEI-3 from k months BEFORE the VHI month
    m = cmv.merge(full.assign(ym=full.ym + k), on=["cell_id", "ym"])
    lag_r[k] = m.VHI.corr(m.s)
lines.append("  lag (k = months SPEI-3 precedes VHI; -1 = SPEI-3 one month later): "
             + str({k: round(v, 3) for k, v in lag_r.items()}))
yr = cmv.groupby("year")[["VHI", "spei3"]].mean()
lines.append(f"  season means per training year (n = {len(yr)}): r = {yr.VHI.corr(yr.spei3):.3f}")
lines.append(yr.round(2).T.to_string())
cmv["anom_month"] = cmv.groupby(["year", "month"]).VHI.transform("mean")
within = (cmv.VHI - cmv.anom_month).corr(cmv.spei3 - cmv.groupby(["year", "month"]).spei3.transform("mean"))
lines.append(f"  within-month (between-cell) r = {within:.3f}  "
             "(how well VHI ranks cells against each other in the same month)")
txt = "\n".join(lines)
(TAB / "vhi_spei_agreement.txt").write_text(txt + "\n")
print(txt)

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
ax[0].hexbin(cmv.spei3, cmv.VHI, gridsize=50, mincnt=1, cmap="viridis")
for c in [20, 30, 40]:
    ax[0].axhline(c, color="w", lw=0.8, ls="--")
for c in [-1.5, -1.0, -0.5]:
    ax[0].axvline(c, color="w", lw=0.8, ls="--")
ax[0].set_xlabel("SPEI-3"); ax[0].set_ylabel("cell-month mean VHI"); ax[0].set_title("Training cell-months")
bym = cmv.groupby("month").apply(lambda d: d.VHI.corr(d.spei3))
ax[1].bar(bym.index, bym.values, color="#3b7d4f"); ax[1].set_xlabel("month"); ax[1].set_ylabel("r(VHI, SPEI-3)")
ax[1].set_title("Agreement by month")
ax[2].scatter(yr.spei3, yr.VHI, color="#a0522d")
for y, r in yr.iterrows():
    ax[2].annotate(str(y), (r.spei3, r.VHI), fontsize=8, xytext=(3, 3), textcoords="offset points")
ax[2].set_xlabel("season mean SPEI-3"); ax[2].set_ylabel("season mean VHI"); ax[2].set_title("Training years")
fig.tight_layout(); fig.savefig(FIG_DIR / "vhi_spei_agreement.png", dpi=130)
