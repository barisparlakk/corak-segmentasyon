"""Day 3, step 3: which SPEI accumulation does vegetation follow? (diagnostic, TRAINING YEARS ONLY)

The spec fixed SPEI-3 on the assumption that it "matches how fast vegetation responds"; this
checks it. SPEI-k (k = 1, 3, 6, 9, 12) is computed exactly as SPEI-3 in s03 (same water
balance, generalized logistic L-moment fit, 1991-2020 reference) and correlated with the
cell-month mean VHI over training cells x training years (test years are not looked at).
Output: outputs/tables/spei_timescale.txt
"""
import numpy as np
import pandas as pd
import xarray as xr

from config import CLIMATE_DIR, ROOT, SPEI_CUTS
from s13_eval import load_rows

ds = xr.open_dataset(CLIMATE_DIR / "spei3.nc")      # holds the monthly water balance D = P - PET
from s03_spei import glo_spei  # noqa: E402

t = ds.time.to_index()
cells = pd.read_csv(CLIMATE_DIR / "cells_split.csv")
rows = load_rows()
a = rows[rows.set == "train"]
cmv = a.groupby(["cell_id", "year", "month"]).agg(VHI=("VHI_clip", "mean"), NDVI_z=("NDVI_z", "mean"),
                                                  LST_z=("LST_TOA_z", "mean")).reset_index()
lines = [f"cell-month means, training cells x training years (n = {len(cmv)})",
         "k   r(VHI)  r(NDVI_z)  r(LST_TOA_z)  r(season-mean VHI, per year)"]
for k in [1, 3, 6, 9, 12]:
    acc = ds.D.rolling(time=k).sum().values
    s = glo_spei(acc, t.month.values, t.year.values)[:, cells.iy, cells.ix]
    lab = pd.DataFrame({"cell_id": np.tile(cells.cell_id.values, len(t)), "year": np.repeat(t.year, len(cells)),
                        "month": np.repeat(t.month, len(cells)), "s": s.ravel()})
    m = cmv.merge(lab, on=["cell_id", "year", "month"])
    yr = m.groupby("year")[["VHI", "s"]].mean()
    if k == 6:   # labels for the SPEI-6 side check (s15_models.py A spei6), same class cut-offs
        split = pd.read_csv(CLIMATE_DIR / "labels_cell_month.csv")[["cell_id", "year", "month", "split"]]
        l6 = split.merge(lab.rename(columns={"s": "spei6"}), on=["cell_id", "year", "month"])
        l6["cls"] = np.digitize(l6.spei6, SPEI_CUTS)
        l6.to_csv(CLIMATE_DIR / "labels_spei6_cell_month.csv", index=False)
    lines.append(f"{k:<3} {m.VHI.corr(m.s):.3f}   {m.NDVI_z.corr(m.s):.3f}      {m.LST_z.corr(m.s):.3f}"
                 f"         {yr.VHI.corr(yr.s):.3f}")
txt = "\n".join(lines)
(ROOT / "outputs" / "tables" / "spei_timescale.txt").write_text(txt + "\n")
print(txt)
