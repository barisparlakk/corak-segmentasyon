"""Day 5, step 3: does the per-scene air-temperature correction (s08c) fix the seams?

Compares features.csv.gz (original LST_TOA) with features_corr.csv.gz (corrected) on TRAINING
cells and TRAINING years only. Same pixel-months in both (rows valid in both).

1. Seam diagnostic: within each year-month, the correlation across cells between cell-mean TCI
   and cell-mean dTa (the overpass-day air-temperature excess, s08c). Seams caused by
   acquisition dates make this strongly negative; a working correction brings it towards 0.
2. Agreement with SPEI-3 (cell-month means): overall r, and within-month r (each year-month's
   mean removed), for TCI and VHI.
3. The decision uses the CV of s15 (run separately with CORAK_FEATURES=features_corr.csv.gz):
   ADOPTION RULE (fixed before results): adopt the corrected LST if VHI tuned's mean CV macro F1
   rises by >= 0.01 AND it is higher in >= 6 of the 11 folds. Test years are not used.

Output: outputs/tables/lst_correction_eval.txt
"""
import numpy as np
import pandas as pd

from config import CLIMATE_DIR, DATA, ROOT

TAB = ROOT / "outputs" / "tables"
lab = pd.read_csv(CLIMATE_DIR / "labels_cell_month.csv")[["cell_id", "year", "month", "spei3", "split"]]
pts = pd.read_csv(DATA / "points" / "points_flags.csv")[["pid", "cell_id"]]
dta = pd.concat([pd.read_csv(f, usecols=["pid", "year", "month", "DTA"])
                 for f in sorted((DATA / "features").glob("lstcorr_*.csv.gz"))])
test_years = [2021, 2022]
cols = ["pid", "year", "month", "TCI", "VCI", "VHI", "VHI_clip"]
a = pd.read_csv(DATA / "features" / "features.csv.gz", usecols=cols)
b = pd.read_csv(DATA / "features" / "features_corr.csv.gz", usecols=cols)
j = a.merge(b, on=["pid", "year", "month"], suffixes=("_orig", "_corr")).merge(dta, on=["pid", "year", "month"])
j = j.dropna(subset=["VHI_orig", "VHI_corr"]).merge(pts, on="pid").merge(lab, on=["cell_id", "year", "month"])
j = j[(j.split == "train") & ~j.year.isin(test_years)]
cm = j.groupby(["cell_id", "year", "month"]).mean(numeric_only=True).reset_index()

lines = [f"training cells x training years: {len(j)} pixel-months, {len(cm)} cell-months\n"]


def within(d, x, y):
    g = d.groupby(["year", "month"])
    return np.corrcoef(d[x] - g[x].transform("mean"), d[y] - g[y].transform("mean"))[0, 1]


ym = cm.groupby(["year", "month"])
seam = pd.DataFrame({n: ym.apply(lambda d: d[f"TCI_{v}"].corr(d.DTA)) for n, v in [("original", "orig"), ("corrected", "corr")]})
lines.append("1. Seam diagnostic: per year-month correlation across cells of TCI with dTa")
lines.append(f"   median r: original {seam.original.median():+.2f}, corrected {seam.corrected.median():+.2f}; "
             f"year-months with r < -0.5: original {(seam.original < -0.5).sum()}, corrected {(seam.corrected < -0.5).sum()} "
             f"of {len(seam)}")
lines.append(f"   May 2021 is a test month and not included. Largest |r| year-months (original):")
top = seam.reindex(seam.original.abs().sort_values(ascending=False).index).head(6).round(2)
lines += ["     " + l for l in top.to_string().splitlines()]

lines.append("\n2. Agreement with SPEI-3, cell-month means (VCI does not change: a sanity check)")
for v in ["TCI", "VCI", "VHI_clip"]:
    for s in ["orig", "corr"]:
        c = f"{v}_{s}"
        lines.append(f"   {v:8s} {s:4s}: overall r {cm[c].corr(cm.spei3):+.3f}, within-month r {within(cm, c, 'spei3'):+.3f}")

txt = "\n".join(lines)
print(txt)
(TAB / "lst_correction_eval.txt").write_text(txt + "\n")
