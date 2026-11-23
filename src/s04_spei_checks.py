"""Step 9: is the SPEI-3 trustworthy?

Checks
  A. Reference period 1991-2020: mean ~0 and SD ~1 for every cell and calendar month.
  B. Class frequencies in the reference period close to the normal distribution
     (30.9% below -0.5, 15.9% below -1.0, 6.7% below -1.5).
  C. Independent implementation (python `spei` package, maximum-likelihood fit) agrees.
  D. Independent data (TerraClimate, Penman-Monteith PET) agrees.
  E. Growing-season class counts per year, needed to choose the test years.

Outputs: outputs/figures/spei_checks.png, data/climate/class_counts_by_year.csv
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as scs
import spei as si
import xarray as xr

from config import CLASS_NAMES, CLIMATE_DIR, FIG_DIR, REF_END, REF_START, SEASON_MONTHS, SPEI_CUTS

FIG_DIR.mkdir(parents=True, exist_ok=True)
sp = xr.open_dataset(CLIMATE_DIR / "spei3.nc")
cells = pd.read_csv(CLIMATE_DIR / "cells.csv")
prov = cells[cells.frac_in >= 0.5]
iy, ix = prov.iy.values, prov.ix.values

S = sp.spei3.values[:, iy, ix]                      # (time, cells in province)
t = sp.time.to_index()
ref = (t.year >= REF_START) & (t.year <= REF_END)


def classify(v):
    return np.digitize(v, SPEI_CUTS)                 # 0 severe, 1 moderate, 2 mild, 3 none


# ---- A
means, sds = [], []
for m in range(1, 13):
    x = S[ref & (t.month == m)]
    means.append(np.nanmean(x, axis=0)); sds.append(np.nanstd(x, axis=0))
means, sds = np.concatenate(means), np.concatenate(sds)
print(f"A. reference mean per cell-month: {means.mean():+.3f} (range {means.min():+.2f} to {means.max():+.2f})")
print(f"   reference SD per cell-month:   {sds.mean():.3f} (range {sds.min():.2f} to {sds.max():.2f})")

# ---- B
xr_ = S[ref & (t.month >= 3)]  # skip Jan-Feb where the first window is incomplete in 1991
for cut, theo in zip([-0.5, -1.0, -1.5], [30.9, 15.9, 6.7]):
    print(f"B. share below {cut}: {100 * np.mean(xr_ < cut):.1f}%  (theory {theo}%)")

# ---- C  independent implementation on 5 cells spread over the province
rng = np.random.default_rng(0)
pick = rng.choice(len(prov), 5, replace=False)
rs = []
for k in pick:
    d = pd.Series(sp.D.values[:, iy[k], ix[k]], index=t)
    other = si.spei(d, dist=scs.fisk, timescale=3, fit_freq="MS").reindex(t)
    mine = pd.Series(S[:, k], index=t)
    ok = other.notna() & mine.notna()
    rs.append(np.corrcoef(mine[ok], other[ok])[0, 1])
print(f"C. correlation with `spei` package (5 cells): {np.round(rs, 3)}")

# ---- D  independent data
St = sp.spei3_tc.values[:, iy, ix]
tt = sp.time_tc.to_index()
common = t.isin(tt)
a, b = np.nanmean(S[common], axis=1), np.nanmean(St, axis=1)
okd = ~np.isnan(a) & ~np.isnan(b)
r_prov = np.corrcoef(a[okd], b[okd])[0, 1]
cls_agree = np.mean(classify(S[common][2:]) == classify(St[2:]))
print(f"D. province-mean correlation with TerraClimate SPEI-3: {r_prov:.3f}; "
      f"same class (all cells, months): {100 * cls_agree:.1f}%")

# ---- E  growing-season class counts per year (cell-months)
season = t.month.isin(SEASON_MONTHS)
rows = []
for y in range(2014, 2027):
    sel = season & (t.year == y)
    c = classify(S[sel]).ravel()
    counts = {CLASS_NAMES[3 - k]: int((c == k).sum()) for k in range(4)}
    counts.update(year=y, mean_spei3=round(float(np.nanmean(S[sel])), 2),
                  jja_spei3=round(float(np.nanmean(S[sel & t.month.isin([6, 7, 8])])), 2))
    rows.append(counts)
tab = pd.DataFrame(rows)[["year", "mean_spei3", "jja_spei3"] + CLASS_NAMES]
tab.to_csv(CLIMATE_DIR / "class_counts_by_year.csv", index=False)
print("E. growing-season (Apr-Oct) cell-months per class:")
print(tab.to_string(index=False))

# ---- figure
fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
ax[0].plot(t, np.nanmean(S, axis=1), lw=1, label="SPEI-3 (CHIRPS + ERA5-Land Hargreaves)")
ax[0].plot(tt, np.nanmean(St, axis=1), lw=1, alpha=0.7, label="SPEI-3 (TerraClimate)")
for c in SPEI_CUTS:
    ax[0].axhline(c, color="grey", lw=0.6, ls="--")
ax[0].axhline(0, color="black", lw=0.6)
ax[0].set_ylabel("Province-mean SPEI-3"); ax[0].legend(loc="lower left", fontsize=8)
ax[0].set_title("Malatya province: SPEI-3, 1991-2025 (dashed lines = class cut-offs)")
ann = sp.P.groupby("time.year").sum().values[:, iy, ix].mean(axis=1)
yrs = np.unique(t.year)
ax[1].bar(pd.to_datetime([f"{y}-07-01" for y in yrs]), ann, width=300, color="tab:blue")
ax[1].axhline(ann[(yrs >= REF_START) & (yrs <= REF_END)].mean(), color="black", lw=0.8)
ax[1].set_ylabel("Annual P (mm, CHIRPS)")
fig.tight_layout()
fig.savefig(FIG_DIR / "spei_checks.png", dpi=130)
print("saved", FIG_DIR / "spei_checks.png")
