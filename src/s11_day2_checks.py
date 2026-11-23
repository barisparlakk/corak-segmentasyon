"""Day 2, step 5: labels per cell-month, data-coverage checks and the irrigation flag.

  1. Valid Landsat data by month and elevation band (snow / cloud gaps).
  2. Drought-class counts per scored set AFTER cloud gaps: a cell-month is evaluable when
     at least MIN_PIX of its pixels have a valid VHI.
  3. Irrigation ("summer-green") flag: a pixel's mean July-August NDVI over TRAINING years.
     In Malatya's dry summer, rainfed crops and pasture are brown by July, so a pixel that
     stays green is almost certainly irrigated (orchards, maize, sugar beet). The cut-off is
     the Otsu threshold of that distribution (the natural valley between the two groups).
     Also reports which WorldCover classes the summer-green pixels fall in, i.e. whether
     orchards are mapped as cropland or as tree/grass.

Outputs: data/climate/labels_cell_month.csv, data/points/points_flags.csv,
         outputs/figures/summer_ndvi_hist.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from config import CLASS_NAMES, CLIMATE_DIR, DATA, FIG_DIR, SEASON_MONTHS, SPEI_CUTS

MIN_PIX = 5
TEST_YEARS = json.loads((CLIMATE_DIR / "split.json").read_text())["test_years"]
pts = pd.read_csv(DATA / "points" / "points.csv")
cells = pd.read_csv(CLIMATE_DIR / "cells_split.csv")
f = pd.read_csv(DATA / "features" / "features.csv.gz")
f = f.merge(pts[["pid", "cell_id", "lc", "elev"]], on="pid")

# ---- labels per cell-month (growing season)
sp = xr.open_dataset(CLIMATE_DIR / "spei3.nc")
t = sp.time.to_index()
sel = t.month.isin(SEASON_MONTHS) & (t.year >= 2014)
S = sp.spei3.values[sel][:, cells.iy, cells.ix]
lab = pd.DataFrame({
    "cell_id": np.tile(cells.cell_id.values, sel.sum()),
    "year": np.repeat(t.year[sel].values, len(cells)),
    "month": np.repeat(t.month[sel].values, len(cells)),
    "spei3": S.ravel(),
})
lab["cls"] = np.digitize(lab.spei3, SPEI_CUTS)             # 0 severe, 1 moderate, 2 mild, 3 none
lab["label"] = lab.cls.map({0: "severe", 1: "moderate", 2: "mild", 3: "none"})
lab = lab.merge(cells[["cell_id", "split"]], on="cell_id")
lab.to_csv(CLIMATE_DIR / "labels_cell_month.csv", index=False)

# ---- 1. valid data by month and elevation band
f["eband"] = pd.cut(f.elev, [0, 1000, 1500, 2000, 4000], labels=["<1000", "1000-1500", "1500-2000", ">2000"])
print("1. share of pixel-months with valid VHI, by elevation band (rows) and month (columns)")
print(f.pivot_table(index="eband", columns="month", values="VHI", aggfunc=lambda s: s.notna().mean(),
                    observed=True).round(2).to_string())

# ---- 2. class counts per scored set after cloud gaps
nvalid = f.groupby(["cell_id", "year", "month"]).VHI.apply(lambda s: s.notna().sum()).rename("nvalid")
lab = lab.merge(nvalid, left_on=["cell_id", "year", "month"], right_index=True, how="left").fillna({"nvalid": 0})
lab["evaluable"] = lab.nvalid >= MIN_PIX
lab["is_test_year"] = lab.year.isin(TEST_YEARS)
sets = {"train": (lab.split == "train") & ~lab.is_test_year,
        "temporal test": (lab.split == "train") & lab.is_test_year,
        "spatial test": (lab.split == "test") & ~lab.is_test_year,
        "strict test": (lab.split == "test") & lab.is_test_year}
rows = []
for name, m in sets.items():
    r = {"set": name}
    for c in CLASS_NAMES[::-1]:
        tot = (m & (lab.label == c)).sum()
        ev = (m & (lab.label == c) & lab.evaluable).sum()
        r[c] = f"{ev}/{tot}"
    rows.append(r)
print(f"\n2. evaluable / total cell-months per class (evaluable = at least {MIN_PIX} valid pixels)")
print(pd.DataFrame(rows).to_string(index=False))

# ---- 3. summer-green (irrigation) flag from training years
tr = f[~f.year.isin(TEST_YEARS) & f.month.isin([7, 8])]
summer = tr.groupby("pid").NDVI.mean().rename("summer_ndvi")


def otsu(x, bins=200):
    x = x[np.isfinite(x)]
    h, e = np.histogram(x, bins=bins)
    c = (e[:-1] + e[1:]) / 2
    w0 = np.cumsum(h); w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = (np.sum(h * c) - np.cumsum(h * c)) / np.maximum(w1, 1)
    return c[np.argmax(w0 * w1 * (m0 - m1) ** 2)]


thr = float(otsu(summer.values))
flags = pts.merge(summer, left_on="pid", right_index=True, how="left")
flags["summer_green"] = flags.summer_ndvi > thr
# Summer-green alone also catches natural forest (WorldCover tree) and high meadows, which are not
# irrigated. Irrigation proxy = summer-green cropland/grassland below 1,500 m. Not a validated
# irrigation map: used only to report results separately (stratification), never as a feature.
flags["irrigated_proxy"] = flags.summer_green & flags.lc.isin([30, 40]) & (flags.elev < 1500)
flags["stratum"] = np.where(flags.lc.isin([10, 20]), "tree/shrub",
                            np.where(flags.irrigated_proxy, "likely irrigated", "rainfed/pasture"))
flags.to_csv(DATA / "points" / "points_flags.csv", index=False)
print(f"\n3. summer-green threshold (Otsu, training years, July-August mean NDVI): {thr:.3f}")
print("   share summer-green:", round(flags.summer_green.mean(), 3))
print("   by WorldCover class (10 tree, 30 grass, 40 cropland):",
      flags.groupby("lc").summer_green.mean().round(3).to_dict())
print("   summer-green pixels by class:", flags[flags.summer_green].lc.value_counts().to_dict())
print("   strata:", flags.stratum.value_counts().to_dict())
print("   by elevation band:", flags.groupby(pd.cut(flags.elev, [0, 1000, 1500, 2000, 4000]), observed=True)
      .summer_green.mean().round(3).to_dict())

fig, ax = plt.subplots(figsize=(7, 4))
for lc, name in [(30, "grassland"), (40, "cropland"), (10, "tree")]:
    ax.hist(flags.loc[flags.lc == lc, "summer_ndvi"].dropna(), bins=80, alpha=0.5, label=name)
ax.axvline(thr, color="black", ls="--", label=f"Otsu {thr:.2f}")
ax.set_xlabel("mean July-August NDVI (training years)"); ax.set_ylabel("pixels"); ax.legend()
ax.set_title("Summer greenness of the sampled pixels")
fig.tight_layout(); fig.savefig(FIG_DIR / "summer_ndvi_hist.png", dpi=130)
