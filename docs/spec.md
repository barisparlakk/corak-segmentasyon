# ÇORAK specification (fixed on Mon 23 Nov 2026)

Every later step reads from this sheet and from `src/config.py`. Changing anything here after Tuesday invalidates the evaluation.

## Study area and grids

| Item | Value | Why |
|---|---|---|
| Study area | Malatya province, FAO GAUL 2025 level 1 (`data/aoi/malatya.geojson`), 12,251 km² | Official boundary, current release |
| Satellite grid | EPSG:32637 (UTM 37N), 30 m, pixel corners at 15 m (mod 30) | Whole province is in zone 37. The offset matches the Landsat C2 scenes, so Landsat is never resampled |
| Climate (label) grid | CHIRPS native 0.05° grid, EPSG:4326. Cell `row = floor((50 − lat)/0.05)`, `col = floor((lon + 180)/0.05)`, `cell_id = row·100000 + col` | Each label cell is identified exactly the same way locally and in Earth Engine |
| Climate cells | 597 touch the province, 507 are at least 50% inside. Elevation 680–2,401 m | |

## Calendar

| Item | Value |
|---|---|
| Satellite seasons | 2014–2026, April–October, monthly composites. 2026 covers April–September only, because CHIRPS ends at September 2026 |
| Model A | Landsat 8/9 features, 2014–2026 |
| Model B | Landsat 8/9 + Sentinel-2 features, 2019–2026 (6 training seasons: 2019, 2020, 2023, 2024, 2025, 2026) |
| Climate record | January 1991 to September 2026. Every month is checked for completeness (6 CHIRPS pentads and all ERA5-Land days) |
| SPEI reference period | 1991–2020 (WMO climate normal) |

## Climate data and SPEI-3

| Item | Value | Why |
|---|---|---|
| Precipitation | CHIRPS pentads summed per month | Complete through September 2026. Annual mean 474 mm over the province |
| PET | Hargreaves (FAO-56) from ERA5-Land daily Tmax, Tmin and Tmean | Needs only temperature. One consistent source for 1991–2025 |
| Why not TerraClimate | It ends in December 2024. Kept as an independent cross-check | |
| 2026 | Very wet per CHIRPS: 558 mm in January–September against a normal of 314 mm. Every 2026 cell-month is "none" except one "mild". It adds wet-season training examples but no drought examples | |
| SPEI fit | Generalized logistic, unbiased L-moments, per cell and calendar month, on 1991–2020 | Same method as the reference R `SPEI` package. Valid for negatively skewed dry months |
| Accumulation | 3 months | Matches how fast vegetation responds |

**Validity checks** (`src/s04_spei_checks.py`, `outputs/figures/spei_checks.png`):

| Check | Expected | Result |
|---|---|---|
| Reference mean per cell-month | 0 | +0.007 |
| Reference SD per cell-month | 1 | 0.98 |
| Share below −0.5 / −1.0 / −1.5 | 30.9% / 15.9% / 6.7% | 31.0% / 16.2% / 6.6% |
| Correlation with the `spei` package (maximum likelihood) | high | 0.994–0.996 |
| Province-mean correlation with TerraClimate SPEI-3 | high | 0.84 |

## Drought classes

| Class | SPEI-3 (target) | VHI (baseline) |
|---|---|---|
| None | ≥ −0.5 | ≥ 40 |
| Mild | −1.0 to −0.5 | 30–40 |
| Moderate | −1.5 to −1.0 | 20–30 |
| Severe | < −1.5 | < 20 |

Every Landsat pixel takes the SPEI class of the climate cell containing it, for the same month.

## Train/test split (fixed before any satellite data was seen)

- **Test years:** 2021 and 2022.
  - 2021 is the driest season of 2019–2025 (June–August SPEI-3 = −2.13).
  - 2022 is near-normal (season mean −0.16). 2019 is closer to zero but has no severe cell-months, so it would leave the class unscored.
  - Together they contain all four classes: 2,846 / 1,102 / 792 / 2,358 cell-months (none / mild / moderate / severe).
  - Both lie inside 2019–2025, so Models A and B are scored on the same seasons.
- **Test blocks:** two pairs of adjacent 5×5-cell blocks (`47_874`, `46_874`, `47_871`, `46_871`, block = `row//5 _ col//5`).
  - They cover 92 cells (15%) and span 691–2,230 m elevation.
  - They were chosen by elevation-stratified sampling. Of 60 candidate layouts, this one loses the fewest cells to the buffer (paired layouts lose 55 on average versus 90 for scattered ones).
- **Buffer:** 50 cells within one cell of a test block. They are used for neither training nor testing.
- **Training:** 455 cells.
- **Files:** `data/climate/split.json`, `data/climate/cells_split.csv`, `outputs/figures/split_map.png`.

**Split checks** (`src/s06_split_checks.py`):

Cell-months per class in each scored set (growing season, 2014 to September 2026):

| Set | Severe | Moderate | Mild | None |
|---|---|---|---|---|
| Train | 5,638 | 4,910 | 4,346 | 19,686 |
| Temporal test | 1,981 | 757 | 961 | 2,671 |
| Spatial test | 1,110 | 1,035 | 917 | 3,930 |
| Strict test | 501 | 120 | 219 | 448 |

- **Class coverage:** every class is present in every set. Moderate is thin in the strict set (120).
- **Land cover:** test cells resemble training cells in vegetated share (0.93 vs 0.90) and mean elevation (1,379 vs 1,409 m). They have less cropland (8% vs 15%), so report metrics per land-cover type as well.
- **Variance:** 96% of SPEI-3 variance in the growing season is between months; only 5% is between cells within a month.
  - The labels are close to province-wide, so the effective sample is about 90 season-months, not tens of thousands of pixels.
  - The temporal test is the meaningful one. The spatial test mainly checks that the satellite features transfer across terrain.

**Reported results:**
1. Test years in training blocks (temporal generalization).
2. Test blocks in training years (spatial generalization).
3. Test blocks in test years (strictest). This is the headline result.

**Leakage rules:**
- VCI/TCI min/max and any feature scaling are computed from training years only.
- The SPEI reference period (1991–2020) is part of the label definition, not a fitted model parameter.
- Label resolution is about 5 km, so metrics are computed per cell-month by majority vote of pixel predictions, not per pixel.

## Masks and strata (built tomorrow in Earth Engine)

| Layer | Definition |
|---|---|
| Vegetation mask | ESA WorldCover v200 (2021) classes 10 tree, 20 shrubland, 30 grassland, 40 cropland. Excludes built-up, bare, snow, water and wetland |
| Cloud masks | Landsat: QA_PIXEL bits for cloud, shadow and snow. Sentinel-2: Cloud Score+ |
| Elevation strata | Copernicus DEM: < 1,000 m, 1,000–1,500 m, 1,500–2,000 m, > 2,000 m |
| Irrigation flag | Candidate: cropland pixels that stay green (high NDVI) in July–August. To be decided on Tuesday |

## Open decisions for Tuesday

- **VCI/TCI scaling.** With min/max from training years only, the held-out drought year 2021 will fall outside 0–100. The plan is to clip to 0–100 for VHI classes and keep the unclipped values as model features. Model B has only six training seasons for its own min/max, so it reuses Model A's Landsat training-year range.

## Known risks to check

- **2025 spring frost.** A frost in April 2025 is reported to have badly damaged Malatya's apricot crop (from memory; to be verified). Low NDVI that season may come from frost rather than drought. 2025 is a training year, not a test year, but check its residuals.
- **February 2023 earthquakes.** Built-up and debris areas changed. The WorldCover 2021 mask predates this. Recheck heavily damaged urban areas.
- **CHIRPS rain gauges.** CHIRPS uses few gauges in eastern Türkiye. TerraClimate agreement (r = 0.84) is the current safeguard.
