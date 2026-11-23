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

## Day 2: satellite features (Tue)

### Fixed pixels (`src/s07_points.py`)
- **What:** 14,779 pixels, 25 per climate cell (593 cells; a few small border cells have fewer), stratified random sample with seed 42.
- **Where they sit:** exactly on Landsat 30 m pixel centres (checked). Each pixel has at least 2/3 WorldCover vegetation cover. The cell_id of each pixel matches `config.cell_rc` (checked).
- **Files:** `data/points/points.csv` and the Earth Engine asset `projects/corak-515608/assets/corak_points`.
- **WorldCover classes of the pixels:** grassland 78%, cropland 16%, tree 6%.

### Monthly values, March–October 2014–2026 (`s08`, `s08b`, `s09`)

| Source | Values | Notes |
|---|---|---|
| Landsat 8/9 Level-2 | NDVI, EVI, SAVI, NDMI | Per-scene scale/offset from metadata. QA_PIXEL cloud, shadow, snow and cirrus masked, plus saturation. Monthly median |
| Landsat 8/9 Level-1 TOA | LST_TOA | Used for TCI and as a feature, for every pixel (see below) |
| Sentinel-2 L2A, 2019–2026 | NDVI, EVI, NDMI, NDRE | Cloud Score+ ≥ 0.6, SCL snow masked. Reflectance is averaged over each 30 m Landsat footprint before computing indices. Monthly median |

**Pilot checks:**
- Values are identical to a single scene's own pixels (100%).
- No rows were lost.
- Lowland cropland NDVI peaks in May. July LST is about 46 °C.

**Why the temperature is LST_TOA, not Landsat ST_B10:**
- **The gap:** Level-2 surface temperature is empty wherever the ASTER GED emissivity database has no data (`ST_EMIS` is masked). Over Malatya that is a north–south strip, roughly 38.0–38.65° E, holding 1,348 pixels (9%), in every year.
- **The method:** LST_TOA is the single-channel method applied to TOA band-10 brightness temperature, with NDVI-threshold emissivity (Sobrino et al. 2004). It is available for 100% of pixels.
- **Agreement:** against ST_B10 where both exist, r = 0.988, with a stable bias of −5.1 °C (no atmospheric correction) and SD 1.9 °C.
- **Why the bias doesn't matter:** TCI compares each pixel with its own history, so a stable bias cancels.
- **ST_B10:** kept in the tables for comparison only, never used as a feature.

**Coverage** (share of pixel-months with valid VHI):

| Elevation | April | May | June | July | August | September | October |
|---|---|---|---|---|---|---|---|
| < 1,000 m | 0.99 | 0.98 | 1.00 | 1.00 | 1.00 | 0.99 | 0.97 |
| > 2,000 m | 0.46 | 0.90 | 0.99 | 1.00 | 0.99 | 0.97 | 0.91 |

April above 2,000 m is the only weak spot, mostly because of snow.

### Indices and features (`s10_indices.py`, `data/features/features.csv.gz`)
- **Size:** 1,344,889 pixel-months (April–October).
- **VCI, TCI, VHI:** per pixel and calendar month, with min/max from training years only (all years except 2021 and 2022) and at least 5 valid years. VHI exists for 96.7% of pixel-months.
- **Out-of-range values:** 17.5% of test-year VCI values fall outside 0–100, as expected because their extremes were not in the reference. VHI classes use the clipped VHI (`VHI_clip`); models get the unclipped values.
- **Model features:** z-scores per pixel and calendar month against training-year mean and SD (NDVI, EVI, SAVI, NDMI, LST_TOA, and the Sentinel-2 indices), plus the previous month's z (lag 1, which is why March is extracted).
  - Sentinel-2 z-scores use its 6 training seasons.
  - Observation counts (NOBS) are for diagnostics only.

### Class coverage after cloud gaps (`s11_day2_checks.py`)
A cell-month is evaluable when at least 5 of its pixels have a valid VHI.

| Set | Severe | Moderate | Mild | None |
|---|---|---|---|---|
| Train | 5,506 / 5,638 | 4,818 / 4,910 | 4,292 / 4,346 | 19,039 / 19,686 |
| Temporal test | 1,922 / 1,981 | 750 / 757 | 929 / 961 | 2,615 / 2,671 |
| Spatial test | 1,110 / 1,110 | 1,019 / 1,035 | 917 / 917 | 3,871 / 3,930 |
| Strict test | 474 / 501 | 120 / 120 | 219 / 219 | 448 / 448 |

Every class stays scoreable in every set.

### Irrigation proxy and reporting strata (`data/points/points_flags.csv`)
- **Summer-green:** mean July–August NDVI over training years above 0.345 (Otsu threshold). 22% of pixels qualify, but this also catches 96% of WorldCover tree pixels (natural forest, orchards) and some high meadows.
- **Irrigation proxy:** summer-green **and** cropland or grassland **and** below 1,500 m.
  - This is not a validated irrigation map.
  - It is used only to report results separately, never as a model feature.
- **Strata:** rainfed/pasture 12,169 pixels, likely irrigated 1,773 (12%), tree/shrub 837.

### First look at the signal (training years, cell-month means vs SPEI-3)

| Stratum | VHI | VCI | TCI | LST_TOA z |
|---|---|---|---|---|
| All | 0.51 | 0.37 | 0.45 | −0.46 |
| Rainfed/pasture | 0.52 | 0.39 | 0.45 | −0.46 |
| Likely irrigated | 0.41 | 0.21 | 0.43 | −0.45 |
| Tree/shrub | 0.33 | 0.10 | 0.40 | −0.41 |

- **Greenness is decoupled where water is not limiting.** In irrigated land and trees, VCI barely follows drought, because irrigation and deep roots keep plants green.
- **Temperature still responds everywhere.** TCI holds at about 0.40–0.45 in every stratum.
- This is the expected physical picture and supports combining both in VHI and in the models.

### Real maps for the presentation (`s12_deck_maps.py`, `outputs/figures/maps/`)
Elevation, slope, Sentinel-2 true colour and NDVI (June–mid-July 2025), WorldCover land cover, and a Malatya city close-up. All in UTM 37N, clipped to the province.

## Known risks to check

- **2025 spring frost.** A frost in April 2025 is reported to have badly damaged Malatya's apricot crop (from memory; to be verified). Low NDVI that season may come from frost rather than drought. 2025 is a training year, not a test year, but check its residuals.
- **February 2023 earthquakes.** Built-up and debris areas changed. The WorldCover 2021 mask predates this. Recheck heavily damaged urban areas.
- **CHIRPS rain gauges.** CHIRPS uses few gauges in eastern Türkiye. TerraClimate agreement (r = 0.84) is the current safeguard.
