"""Step 10-11: fix the test years and the spatial test blocks BEFORE any satellite data is seen.

Test years (chosen from class_counts_by_year.csv, 2019-2025 so Model A and B share them):
  - one clearly dry year and one near-normal year,
  - together they must contain every drought class, otherwise per-class F1/IoU is undefined.

Spatial blocks: 5 x 5 climate cells (0.25 degree, about 28 x 22 km). About 20% of the
province's cells go to test, spread over elevation strata. Cells within one cell of a
test block are a buffer, used for neither training nor testing.
Several candidate layouts are compared and the one losing the fewest cells to the buffer
is kept.

Outputs: data/climate/split.json, data/climate/cells_split.csv, outputs/figures/split_map.png
"""
import json

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import AOI_GEOJSON, CLASS_NAMES, CLIMATE_DIR, FIG_DIR

BLOCK = 5
TEST_SHARE = 0.20
TEST_YEARS = [2021, 2022]        # 2021: driest season of 2019-2025; 2022: closest to normal

# ---- test years: verify the class-coverage rule
tab = pd.read_csv(CLIMATE_DIR / "class_counts_by_year.csv").set_index("year")
cov = tab.loc[TEST_YEARS, CLASS_NAMES].sum()
print("Test years", TEST_YEARS, "class cell-months:", cov.to_dict())
assert (cov > 0).all(), "test years must contain every class"

# ---- blocks
cells = pd.read_csv(CLIMATE_DIR / "cells.csv")
cells = cells[cells.frac_in > 0].copy()
cells["brow"], cells["bcol"] = cells.row // BLOCK, cells.col // BLOCK
cells["block"] = cells.brow.astype(str) + "_" + cells.bcol.astype(str)
blocks = cells.groupby("block").agg(n=("cell_id", "size"), elev=("elev", "mean"),
                                    brow=("brow", "first"), bcol=("bcol", "first"))
full = blocks[blocks.n >= 15]                          # only well-filled blocks can be test blocks
print(f"Cells: {len(cells)} | blocks: {len(blocks)} | blocks with >=15 cells: {len(full)}")

target = TEST_SHARE * len(cells)
rc = set(zip(cells.row, cells.col))


def evaluate(test_blocks):
    test = cells.block.isin(test_blocks)
    trc = set(zip(cells[test].row, cells[test].col))
    near = cells.apply(lambda c: any((c.row + dr, c.col + dc) in trc
                                     for dr in (-1, 0, 1) for dc in (-1, 0, 1)), axis=1)
    buf = near & ~test
    return test, buf


def stratified_layout(seed, n_strata):
    rng = np.random.default_rng(seed)
    strata = pd.qcut(full.elev.rank(method="first"), n_strata, labels=False)
    chosen = []
    for s in range(n_strata):
        cand = full[strata == s].index.tolist()
        rng.shuffle(cand)
        for b in cand:  # no two test blocks touching each other (keeps them spread out)
            br, bc = full.loc[b, ["brow", "bcol"]]
            if all(abs(br - full.loc[o, "brow"]) > 1 or abs(bc - full.loc[o, "bcol"]) > 1 for o in chosen):
                chosen.append(b)
                break
    return chosen


def contiguous_layout(seed, n_groups):
    """Pairs of neighbouring blocks: fewer test-block edges, so a smaller buffer."""
    rng = np.random.default_rng(seed)
    strata = pd.qcut(full.elev.rank(method="first"), n_groups, labels=False)
    chosen = []
    for s in range(n_groups):
        cand = full[strata == s].index.tolist()
        rng.shuffle(cand)
        for b in cand:
            br, bc = full.loc[b, ["brow", "bcol"]]
            nb = [o for o in full.index if o != b and o not in chosen
                  and abs(full.loc[o, "brow"] - br) + abs(full.loc[o, "bcol"] - bc) == 1]
            far = all(abs(br - full.loc[o, "brow"]) > 2 or abs(bc - full.loc[o, "bcol"]) > 2 for o in chosen)
            if nb and far:
                chosen += [b, nb[0]]
                break
    return chosen


results = []
n_scatter = max(2, round(target / full.n.mean()))
for seed in range(30):
    for kind, fn, arg in [("scattered", stratified_layout, n_scatter), ("paired", contiguous_layout, max(1, n_scatter // 2))]:
        tb = fn(seed, arg)
        test, buf = evaluate(tb)
        results.append({"kind": kind, "seed": seed, "blocks": tb, "test": int(test.sum()),
                        "buffer": int(buf.sum()), "train": int((~test & ~buf).sum()),
                        "elev_lo": cells[test].elev.min(), "elev_hi": cells[test].elev.max()})
res = pd.DataFrame(results)
res["share"] = res.test / len(cells)
ok = res[(res.share.between(0.15, 0.25)) & (res.elev_hi - res.elev_lo > 800)]
print(ok.groupby("kind")[["test", "buffer", "train"]].mean().round(0))
best = ok.sort_values(["buffer", "seed"]).iloc[0]
print("Chosen layout:", best.kind, "seed", best.seed, "| test cells", best.test,
      "| buffer", best.buffer, "| train", best.train,
      f"| test elevation {best.elev_lo:.0f}-{best.elev_hi:.0f} m")

test, buf = evaluate(best.blocks)
cells["split"] = np.where(test, "test", np.where(buf, "buffer", "train"))
cells.to_csv(CLIMATE_DIR / "cells_split.csv", index=False)
split = {"test_years": TEST_YEARS, "block_size_cells": BLOCK, "layout": best.kind,
         "seed": int(best.seed), "test_blocks": list(best.blocks),
         "counts": {k: int(v) for k, v in cells.split.value_counts().items()}}
(CLIMATE_DIR / "split.json").write_text(json.dumps(split, indent=2))

# ---- map
g = gpd.read_file(CLIMATE_DIR / "cells.gpkg")
g = g.merge(cells[["cell_id", "split"]], on="cell_id")
aoi = gpd.read_file(AOI_GEOJSON)
fig, ax = plt.subplots(figsize=(8, 6))
colors = {"train": "#9ecae1", "buffer": "#d9d9d9", "test": "#e6550d"}
for k, c in colors.items():
    g[g.split == k].plot(ax=ax, color=c, edgecolor="white", linewidth=0.3, label=k)
aoi.boundary.plot(ax=ax, color="black", linewidth=1)
ax.set_title(f"Spatial split ({best.kind}, {BLOCK}x{BLOCK}-cell blocks); test years {TEST_YEARS}")
handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
ax.legend(handles, colors.keys(), loc="lower right")
fig.tight_layout(); fig.savefig(FIG_DIR / "split_map.png", dpi=130)
print("saved split.json, cells_split.csv, split_map.png")
