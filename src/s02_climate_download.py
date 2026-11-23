"""Step 6-7: download monthly climate for every 0.05 degree cell over Malatya.

Main source (1991 to 2026-09, consistent for the whole period):
  - precipitation: CHIRPS pentads summed per month (mm)
  - Tmax / Tmin / Tmean: ERA5-Land daily aggregates averaged per month (deg C),
    bilinearly resampled onto the CHIRPS grid
Cross-check source (1991-2024): TerraClimate pr and pet (Penman-Monteith), same grid.
TerraClimate ends in 2024-12, which is why it is not the main source.

Output: data/climate/climate_monthly.nc, data/climate/cells.csv
"""
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.geometry import box

from config import (AOI_GEOJSON, CLIM_END, CLIM_LAST_MONTH, CLIM_RES, CLIM_START, CLIM_X0, CLIM_Y0,
                    CLIMATE_DIR, CRS_UTM, ee_init)

warnings.filterwarnings("ignore", category=DeprecationWarning)
ee = ee_init()
CLIMATE_DIR.mkdir(parents=True, exist_ok=True)

# ---- grid over the province, snapped to the CHIRPS cell edges, 1-cell margin
aoi = gpd.read_file(AOI_GEOJSON)
xmin, ymin, xmax, ymax = aoi.total_bounds
c0 = int(np.floor((xmin - CLIM_X0) / CLIM_RES)) - 1
c1 = int(np.floor((xmax - CLIM_X0) / CLIM_RES)) + 1
r0 = int(np.floor((CLIM_Y0 - ymax) / CLIM_RES)) - 1
r1 = int(np.floor((CLIM_Y0 - ymin) / CLIM_RES)) + 1
ncol, nrow = c1 - c0 + 1, r1 - r0 + 1
x0 = CLIM_X0 + c0 * CLIM_RES
y0 = CLIM_Y0 - r0 * CLIM_RES
GRID = {
    "dimensions": {"width": ncol, "height": nrow},
    "affineTransform": {"scaleX": CLIM_RES, "shearX": 0, "translateX": x0,
                        "shearY": 0, "scaleY": -CLIM_RES, "translateY": y0},
    "crsCode": "EPSG:4326",
}
lons = x0 + (np.arange(ncol) + 0.5) * CLIM_RES
lats = y0 - (np.arange(nrow) + 0.5) * CLIM_RES
print(f"Grid: {nrow} rows x {ncol} cols, rows {r0}-{r1}, cols {c0}-{c1}")

months = pd.date_range(f"{CLIM_START}-01-01", f"{CLIM_LAST_MONTH}-01", freq="MS")


def pull(image, names):
    """Download a multi-band image on GRID as a dict of 2-D arrays."""
    arr = ee.data.computePixels({"expression": image, "fileFormat": "NUMPY_NDARRAY", "grid": GRID})
    return {n: np.asarray(arr[n], dtype="float64") for n in names}


def monthly_stack(fn, label, year_list):
    """fn(start_date) -> single-band image; pulled in yearly chunks to stay under request limits."""
    out = {}
    for yr in year_list:
        ms = [m for m in months if m.year == yr]
        imgs = [fn(ee.Date(m.strftime("%Y-%m-%d"))).rename(f"{label}_{m:%Y%m}") for m in ms]
        img = ee.Image.cat(imgs)
        out.update(pull(img, [f"{label}_{m:%Y%m}" for m in ms]))
    return np.stack([out[f"{label}_{m:%Y%m}"] for m in months if m.year in year_list])


chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD").select("precipitation")
era = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
tc = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE")


def p_month(d):
    return chirps.filterDate(d, d.advance(1, "month")).sum()


def t_month(band):
    def f(d):
        return (era.filterDate(d, d.advance(1, "month")).select(band).mean()
                .subtract(273.15).resample("bilinear"))
    return f


def tc_month(band):
    def f(d):
        return tc.filterDate(d, d.advance(1, "month")).select(band).first().toFloat().resample("bilinear")
    return f


# ---- partial-month guard: every month must have all 6 CHIRPS pentads and every ERA5-Land day,
# otherwise a monthly sum/mean would silently come from incomplete data
counts = ee.List([ee.List([chirps.filterDate(ee.Date(m.strftime("%Y-%m-%d")), ee.Date(m.strftime("%Y-%m-%d")).advance(1, "month")).size(),
                           era.filterDate(ee.Date(m.strftime("%Y-%m-%d")), ee.Date(m.strftime("%Y-%m-%d")).advance(1, "month")).size()])
                  for m in months]).getInfo()
bad = [(m.strftime("%Y-%m"), c) for m, c in zip(months, counts) if c[0] != 6 or c[1] != m.days_in_month]
assert not bad, f"incomplete months (pentads, days): {bad}"
print(f"Partial-month guard passed for {len(months)} months ({months[0]:%Y-%m} to {months[-1]:%Y-%m})")

years = list(range(CLIM_START, CLIM_END + 1))
tc_years = [y for y in years if y <= 2024]

data = {}
for name, fn, yrs in [("P", p_month, years),
                      ("Tmax", t_month("temperature_2m_max"), years),
                      ("Tmin", t_month("temperature_2m_min"), years),
                      ("Tmean", t_month("temperature_2m"), years),
                      ("tc_pr", tc_month("pr"), tc_years),
                      ("tc_pet_raw", tc_month("pet"), tc_years)]:
    print("downloading", name)
    data[name] = monthly_stack(fn, name, yrs)

time = months
tc_time = months[months.year <= 2024]
ds = xr.Dataset(
    {k: (("time", "lat", "lon"), v) for k, v in data.items() if not k.startswith("tc_")},
    coords={"time": time, "lat": lats, "lon": lons},
)
for k in ["tc_pr", "tc_pet_raw"]:
    ds[k] = xr.DataArray(data[k], dims=("time_tc", "lat", "lon"),
                         coords={"time_tc": tc_time, "lat": lats, "lon": lons})

# ---- per-cell table: ids, share of the cell inside Malatya, mean elevation
cells = []
for i, la in enumerate(lats):
    for j, lo in enumerate(lons):
        cells.append({"row": r0 + i, "col": c0 + j, "iy": i, "ix": j, "lon": lo, "lat": la,
                      "geometry": box(lo - CLIM_RES / 2, la - CLIM_RES / 2, lo + CLIM_RES / 2, la + CLIM_RES / 2)})
cells = gpd.GeoDataFrame(cells, crs="EPSG:4326")
cu = cells.to_crs(CRS_UTM)
au = aoi.to_crs(CRS_UTM).geometry.iloc[0]
cells["frac_in"] = cu.intersection(au).area / cu.area

dem = (ee.Image("CGIAR/SRTM90_V4").select("elevation")
       .reduceResolution(ee.Reducer.mean(), maxPixels=65535)
       .rename("elev"))
elev = pull(dem, ["elev"])["elev"]
cells["elev"] = elev[cells.iy, cells.ix]
cells["cell_id"] = cells.row * 100000 + cells.col

ds.to_netcdf(CLIMATE_DIR / "climate_monthly.nc")
cells.drop(columns="geometry").to_csv(CLIMATE_DIR / "cells.csv", index=False)
cells.to_file(CLIMATE_DIR / "cells.gpkg", driver="GPKG")
inside = cells[cells.frac_in > 0]
print(f"Cells touching Malatya: {len(inside)} (>=50% inside: {(cells.frac_in >= 0.5).sum()})")
print(f"Elevation of those cells: {inside.elev.min():.0f}-{inside.elev.max():.0f} m")
