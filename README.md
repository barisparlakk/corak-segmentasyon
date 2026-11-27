# ÇORAK: satellite drought detection for Malatya province

Can 30 m Landsat (and Sentinel-2) imagery detect agricultural drought across all of Malatya, Türkiye?
The ground truth is the climate drought index SPEI-3. A 5-day solo project, 23–27 November 2026.

- **Target:** SPEI-3 drought class per 0.05° climate cell and month (none > −0.5, mild −0.5 to −1.0, moderate −1.0 to −1.5, severe < −1.5), growing season April–October, 2014–2026.
- **Baseline:** the Vegetation Health Index (VHI = ½ VCI + ½ TCI) with Kogan's cut-offs 40 / 30 / 20.
- **Models:** XGBoost on per-pixel anomaly features. Model A uses Landsat 8/9, 2014–2026. Model B adds Sentinel-2, 2019–2026.
- **Evaluation:** test years 2021 and 2022 plus spatial test blocks. Both were fixed before any satellite data was seen.

The full specification, every decision and every result, with dates, is in [`docs/spec.md`](docs/spec.md).

## Main results

| Finding | Numbers |
|---|---|
| VHI follows SPEI-3 between seasons | Province seasonal means: r = 0.86 over the training years and 0.83 over all 13 years. Cell-month r = 0.53 |
| VHI does not separate cells within a month | Within-month r between cells = 0.01. Kogan VHI on the strict test set (test cells × 2021–22): macro F1 0.27, against 0.13 for "always none" |
| No machine-learning model beats VHI | The pre-registered XGBoost (v1) lost to "always none" on 2021–22 (macro F1 0.11 temporal and 0.10 strict test). In the redesigned v2, leave-one-year-out CV over 11 training years gives VHI with tuned cut-offs 0.30 and the best model 0.29. No candidate passed the pass rule declared in advance, so the test years were not re-scored |
| Sentinel-2 adds nothing | Model B equals the Landsat-only model on the same years (2019+) |
| SPEI-6 is not a better target | VHI correlates slightly better with SPEI-6 (0.57 vs 0.53), but its classes are matched worse in CV |
| Per-scene temperature correction | Removes most of the Landsat scene seams in the monthly maps, but does not improve class agreement. Not adopted |

The main conclusion is that the limit is the label: a 5 km climate index compared with 30 m vegetation. It is not the model. At the seasonal scale, VHI is a usable drought signal. Within a month, it does not reproduce SPEI-3's cell-to-cell pattern.

**Known effects** (checked, see the spec):
- **April 2025 frost:** it lowered tree greenness in April–May 2025, which looks like drought in VHI.
- **February 2023 earthquakes:** no lasting vegetation change shows at the sample pixels.

## Products

All products are Cloud-Optimized GeoTIFFs in `outputs/maps/` (about 1.2 GB, not in git; rebuild with `s17`–`s21`):

| File | Content |
|---|---|
| `vhi_YYYY_MM.tif` | Monthly VHI, 30 m, EPSG:32637, April–October 2014–2026 (91 maps), vegetation pixels only |
| `vhi_season_YYYY.tif` | Growing-season mean VHI per year (13 maps). The main product |
| `spei3_YYYY_MM.tif`, `spei6_YYYY_MM.tif` | SPEI-3 / SPEI-6 on the 0.05° climate grid, clipped to the province |
| `*_class.tif` | SPEI-3 drought classes, monthly and seasonal |
| `corak_maps.qgz` | QGIS project with every layer grouped and styled, and an OpenStreetMap base map |

Class rasters carry an embedded colour table, so QGIS and ArcGIS show the drought classes without any style file. A QGIS `.qml` style sits next to every raster.

Monthly maps show straight-edged blocks along Landsat path edges. This is weather on the acquisition date in the uncorrected thermal band; the seasonal maps mostly smooth it out.

Quick looks: [`outputs/figures/maps_seasons.png`](outputs/figures/maps_seasons.png) and `outputs/figures/maps_<year>.png`.

## Repository layout

```
src/              pipeline scripts s01–s22 and config.py (run from src/)
docs/spec.md      specification and results log
docs/sunum/       advisor presentation (Turkish) snapshots and figures
data/aoi/         province boundary (FAO GAUL 2025)
data/climate/     climate cells, train/test split, SPEI-3/6 labels (cell-month)
data/points/      fixed vegetation sample pixels and strata
data/preds/       cell-month predictions of the baseline and v1 model
outputs/tables/   scores, CV folds, agreement checks
outputs/models/   model files and CV decisions
outputs/figures/  figures
```

Not in git (rebuilt by the scripts): `data/features/` (pixel-month tables), `data/ref/` (per-pixel VHI references), `outputs/maps/`, and the `*.nc` / `*.gpkg` / `*.tif` files.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| Labels | `s01_aoi.py` | Province boundary; checks that the 30 m grid matches Landsat |
| | `s02_climate_download.py` | Monthly CHIRPS precipitation and ERA5-Land temperature per 0.05° cell, 1991–2026 (TerraClimate as a cross-check) |
| | `s03_spei.py`, `s04_spei_checks.py` | SPEI-3 (Hargreaves PET, generalized logistic fit by L-moments, 1991–2020 reference) and its checks |
| | `s05_split.py`, `s06_split_checks.py` | Freezes the test years and the spatial test blocks (`data/climate/split.json`) |
| Features | `s07_points.py` | Fixed vegetation sample pixels per climate cell (WorldCover 2021) |
| | `s08_landsat.py`, `s08b_landsat_lst.py`, `s09_sentinel2.py` | Monthly Landsat 8/9 indices, TOA-based land surface temperature, Sentinel-2 indices at those pixels |
| | `s10_indices.py`, `s11_day2_checks.py` | VCI / TCI / VHI and anomaly features (leave-one-year-out climatology); coverage checks |
| Evaluation | `s13_eval.py` | Shared loading and scoring (cell-month majority vote, macro F1, QWK, bootstrap intervals) |
| | `s14_baseline.py` | VHI baseline and VHI–SPEI-3 agreement |
| | `s15_models.py` (`A`, `A19`, `B`) | XGBoost candidates in leave-one-year-out CV against reference rules, with the pass rule declared in advance. `s15_models_v1.py` is the original, pre-registered run |
| | `s16_timescale.py` | Which SPEI window (1–12 months) vegetation follows |
| Maps | `s17_vhi_climatology.py` | Per-pixel VHI references (min/max over the training years) |
| | `s18_maps.py`, `s20_finalize_maps.py` | Monthly and seasonal VHI and SPEI maps, vegetation mask, colours |
| | `s19_quicklook.py`, `s21_qgis_project.py` | Figures and the QGIS project |
| Checks | `s08c_lst_corrected.py`, `s22_lst_correction_eval.py` | Per-scene air-temperature correction of the thermal band and its evaluation (not adopted) |
| Slides | `s12_deck_maps.py` | Maps for the presentation |

## Running it

Requirements:
- Python 3.12.
- A Google Earth Engine project with access to the Earth Engine API; set `GEE_PROJECT` in `src/config.py`.
- On macOS, XGBoost needs OpenMP: `brew install libomp`.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install earthengine-api numpy pandas xarray netCDF4 scipy spei scikit-learn xgboost rasterio geopandas shapely matplotlib
earthengine authenticate
cd src
python s01_aoi.py        # then s02 ... s22 in order; each script says what it writes
```

Versions used:

| Package | Version |
|---|---|
| earthengine-api | 1.7.53 |
| numpy | 2.5.5 |
| pandas | 3.1.1 |
| xarray | 2026.9.0 |
| scipy | 1.18.1 |
| spei | 0.8.3 |
| scikit-learn | 1.9.1 |
| xgboost | 3.4.1 |
| rasterio | 1.5.2 (GDAL 3.12.2) |
| geopandas | 1.2.0 |
| matplotlib | 3.11.2 |

Earth Engine Community tier note: the full run used up the 150 EECU-hour monthly quota. `s18_maps.py` therefore fetches the maps in small blocks with retries instead of batch exports.

`s21_qgis_project.py` runs with QGIS's own Python, for example on macOS:

```bash
QGIS=/Applications/QGIS-final-4_2_3.app/Contents
QT_QPA_PLATFORM=offscreen PYTHONPATH=$QGIS/Resources/python3.12/site-packages $QGIS/MacOS/python s21_qgis_project.py
```

## Data sources

- Landsat 8/9 Collection 2 Level-2 and Level-1 TOA (USGS).
- Sentinel-2 L2A with Cloud Score+ (ESA / Google).
- CHIRPS precipitation (UCSB CHC).
- ERA5-Land (ECMWF / Copernicus).
- TerraClimate (cross-check only).
- ESA WorldCover 2021.
- FAO GAUL 2025 boundaries.

All are accessed through Google Earth Engine.
