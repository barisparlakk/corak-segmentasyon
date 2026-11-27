"""Day 4, step 3: quick-look figure of the map products, one per year.

Top row: VHI drought classes (30 m, shown at 240 m), April-October. Bottom row: SPEI-3 classes on
the 0.05 deg climate grid, same months. Same colours as the QGIS styles.
Usage: python s19_quicklook.py [YEAR ...]   -> outputs/figures/maps_<year>.png
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap

from config import FIG_DIR, ROOT, SEASON_MONTHS

MAPS = ROOT / "outputs" / "maps"
YEARS = [int(a) for a in sys.argv[1:]] or [2021, 2022, 2025]
cmap = ListedColormap(["#7f0000", "#e34a33", "#fdbb84", "#31a354"])
cmap.set_bad("white")
MONTHS = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]

for year in YEARS:
    fig, ax = plt.subplots(2, 7, figsize=(21, 6.2))
    for j, m in enumerate(SEASON_MONTHS):
        p = MAPS / f"vhi_{year}_{m:02d}.tif"
        if p.exists():
            with rasterio.open(p) as ds:
                a = ds.read(1, out_shape=(ds.height // 8, ds.width // 8),
                            resampling=rasterio.enums.Resampling.nearest).astype(float)
            a[a == 255] = np.nan
            cls = np.digitize(a, [20, 30, 40]).astype(float)
            cls[np.isnan(a)] = np.nan
            ax[0, j].imshow(np.ma.masked_invalid(cls), cmap=cmap, norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4),
                            interpolation="nearest")
            ax[0, j].set_title(f"{MONTHS[j]} {year}  VHI (median {np.nanmedian(a):.0f})", fontsize=10)
        p = MAPS / f"spei3_{year}_{m:02d}.tif"
        if p.exists():
            with rasterio.open(p) as ds:
                s = ds.read(1).astype(float)
            s[s == -9999] = np.nan
            cls = np.digitize(s, [-1.5, -1.0, -0.5]).astype(float)
            cls[np.isnan(s)] = np.nan
            ax[1, j].imshow(np.ma.masked_invalid(cls), cmap=cmap, norm=BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4),
                            interpolation="nearest")
            ax[1, j].set_title(f"{MONTHS[j]} {year}  SPEI-3 (mean {np.nanmean(s):+.1f})", fontsize=10)
        for a_ in ax[:, j]:
            a_.set_xticks([]); a_.set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in cmap.colors]
    fig.legend(handles, ["severe", "moderate", "mild", "none"], loc="lower center", ncol=4, frameon=False)
    fig.suptitle(f"Malatya {year}: satellite VHI classes (top) vs climate SPEI-3 classes (bottom)", fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    out = FIG_DIR / f"maps_{year}.png"
    fig.savefig(out, dpi=110)
    print("saved", out)

# ---- seasonal overview: every year, VHI season classes (top) and SPEI-3 season classes (bottom)
from config import STUDY_END, STUDY_START  # noqa: E402

years = [y for y in range(STUDY_START, STUDY_END + 1) if (MAPS / f"vhi_season_{y}.tif").exists()]
if years:
    fig, ax = plt.subplots(2, len(years), figsize=(2.3 * len(years), 5.6))
    norm = BoundaryNorm([-.5, .5, 1.5, 2.5, 3.5], 4)
    for j, y in enumerate(years):
        with rasterio.open(MAPS / f"vhi_season_{y}.tif") as ds:
            a = ds.read(1, out_shape=(ds.height // 10, ds.width // 10),
                        resampling=rasterio.enums.Resampling.nearest).astype(float)
        a[a == 255] = np.nan
        c = np.digitize(a, [20, 30, 40]).astype(float); c[np.isnan(a)] = np.nan
        ax[0, j].imshow(np.ma.masked_invalid(c), cmap=cmap, norm=norm, interpolation="nearest")
        ax[0, j].set_title(f"{y}\nVHI {np.nanmean(a):.0f}", fontsize=9)
        with rasterio.open(MAPS / f"spei3_season_{y}_class.tif") as ds:
            s = ds.read(1).astype(float)
        s[s == 255] = np.nan
        ax[1, j].imshow(np.ma.masked_invalid(s), cmap=cmap, norm=norm, interpolation="nearest")
        with rasterio.open(MAPS / f"spei3_season_{y}.tif") as ds:
            sv = ds.read(1).astype(float)
        ax[1, j].set_title(f"SPEI-3 {np.nanmean(np.where(sv == -9999, np.nan, sv)):+.2f}", fontsize=9)
        for a_ in ax[:, j]:
            a_.set_xticks([]); a_.set_yticks([])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c_) for c_ in cmap.colors]
    fig.legend(handles, ["severe", "moderate", "mild", "none"], loc="lower center", ncol=4, frameon=False)
    fig.suptitle("Malatya growing season (Apr-Oct): satellite VHI (top, 30 m, vegetation only) vs SPEI-3 (bottom)")
    fig.tight_layout(rect=[0, 0.06, 1, 0.93], h_pad=2.5)
    fig.savefig(FIG_DIR / "maps_seasons.png", dpi=110)
    print("saved", FIG_DIR / "maps_seasons.png")
