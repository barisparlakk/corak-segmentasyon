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

## Day 2: satellite features (plan day 2, run Mon 23 Nov)

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
- **VCI, TCI, VHI:** per pixel and calendar month, with min/max from training years only (all years except 2021 and 2022) and at least 5 valid years. Changed on day 3: a training year now uses the *other* training years (see Day 3). VHI exists for 96.7% of pixel-months.
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

## Day 3: baseline and first models (plan day 3, run Mon 23 Nov)

Scripts: `s13_eval.py` (shared scoring), `s14_baseline.py`, `s15_models_v1.py`, `s15_models.py` (v2).
Tables in `outputs/tables/`, cell-month predictions in `data/preds/`.

### Scoring rules (identical for every method)
- **Rows:** pixel-months with a valid VHI. Buffer cells are never used.
- **Scored cell-months:** at least 5 pixels with a valid VHI.
- **Cell-month class:** majority vote of the pixel classes. Ties go to the tied class closest to the mean pixel class; a remaining tie goes to the drier class.
- **Metrics:** macro F1, quadratic-weighted kappa (QWK), accuracy and recall per class. 95% intervals come from a bootstrap over year-months.
- **Small test sample:** the temporal test has only 14 year-months, so its intervals are wide.

### VHI baseline on the test sets (`baseline_scores.csv`)

| Method | Temporal test F1 / QWK | Strict test F1 / QWK | Spatial test F1 / QWK |
|---|---|---|---|
| Always none | 0.15 / 0 | 0.13 / 0 | 0.18 / 0 |
| VHI Kogan 20/30/40 | 0.24 / 0.02 | 0.27 / 0.13 | 0.33 / 0.32 |
| VHI, cut-offs refitted on train (20/35/50) | 0.22 / 0.00 | 0.22 / 0.10 | 0.36 / 0.35 |

- **Spatial test:** it shares its months with training, so its scores are optimistic.
- **Kappa:** every test-year interval includes 0.
- **Recall:** VHI recognises severe drought best (severe recall 0.52 temporal, 0.38 strict).
- **2022 is the key finding.** The satellites see 2022 as nearly as stressed as 2021 (season median VHI 29 vs 25), while SPEI-3 calls it near-normal or wet until August.
  - This sinks every satellite method in that year, VHI included.
  - One possible reason is carry-over from the 2021 drought (soil and groundwater, perennial plants), which a 3-month window cannot see. This is untested. SPEI-6 did not improve class agreement in training-year CV, so it is not confirmed there either.

### VHI–SPEI agreement (training years and training cells only, `vhi_spei_agreement.txt`)
- **Cell-month correlation:** r = 0.53. It is highest in April, June and October (0.63–0.68).
- **Lag:** VHI matches SPEI-3 of the same month best (r = 0.53). Lagged SPEI does worse: 0.43 at one month earlier, 0.45 at two months earlier, 0.40 at one month later.
- **Season mean per year:** r = 0.86. Satellites see dry and wet years clearly.
- **Within-month correlation between cells:** r = 0.01.
  - Inside one month, VHI does not rank which cells are drier.
  - Within-month label variance is small (about 4% of the total), so this r would be low in any case.
  - The thermal scene seams found on day 4 (see below) may also contribute. This is a hypothesis, not tested per Landsat path.
  - In effect, the province has one drought state per month: about 76 independent training samples, not 1.2 million rows.

### Model A v1, the pre-registered run (`*_v1_*`, `s15_models_v1.py`)
- **Setup:**
  - XGBoost classifier.
  - Features: indices, z-anomalies, previous-month z, VCI/TCI/VHI, month, elevation, land cover.
  - Tuning: leave-one-year-out CV.
- **Test result:** temporal F1 0.11 / QWK −0.03, strict F1 0.11 / QWK 0.02. This is worse than "always none".
- **Likely causes:**
  - **The calendar plus class weights.** In training years, August–September are often severe (25–32%), and class weighting lifts that prior above "none". The model predicts severe drought every August–September in both 2021 and 2022.
  - **The two test years look alike to the satellites but not to SPEI.** Median NDVI z is −0.75 in 2021 and −0.56 in 2022; LST z is +0.61 and +0.50.
  - So near-identical predictions would follow even without a calendar effect.
  - Dropping `month` and the raw levels later moved CV only from about 0.28 to 0.29, so the calendar is not the whole story.
- **Status:** this test-year result stands as the one pre-registered score.

### Changes after v1 (decided on training years only)
1. **Leave-one-year-out climatology (s10).** A training year's min/max/mean/SD now come from the other training years, so training anomalies are built exactly like test-year ones.
   - Training-year VCI is now outside 0–100 in 18% of cases, against 17.5% in test years. Before this change it was 0% by construction.
   - Test-year features are unchanged. v1 features are kept as `features_v1.csv.gz`.
2. **Anomaly-only features:** no month and no raw levels.
3. **CV against references, fold by fold.** References: always none, calendar (most common class per month), VHI Kogan, and VHI with refitted cut-offs.
   - Candidates, declared in advance: C1 XGB classifier with anomalies and cell means; C2 XGB regressor of SPEI-3 with cell means; C3 the same regressor without cell means.
   - **Pass rule, declared before running:** mean CV macro F1 at least 0.03 above the best reference, and better in at least 2/3 of folds.

### v2 cross-validation result (`*_v2_cv_summary.csv`): no model passes

Mean leave-one-year-out macro F1 / QWK:

| Method | Model A, 11 folds | A features 2019+, 6 folds | Model B (+Sentinel-2), 6 folds |
|---|---|---|---|
| Always none | 0.21 / 0 | 0.22 / 0 | 0.22 / 0 |
| Calendar | 0.25 / 0.10 | 0.22 / 0.15 | 0.22 / 0.15 |
| VHI Kogan | 0.29 / 0.20 | 0.29 / 0.24 | 0.29 / 0.24 |
| **VHI tuned (best reference)** | **0.30 / 0.22** | **0.29 / 0.24** | **0.29 / 0.24** |
| C1 classifier + cell means | 0.29 / 0.24 | 0.27 / 0.17 | 0.26 / 0.18 |
| C2 regressor + cell means | 0.26 / 0.20 | 0.24 / 0.12 | 0.25 / 0.13 |
| C3 regressor, pixel only | 0.25 / 0.16 | 0.23 / 0.10 | 0.24 / 0.10 |

- **The models are indistinguishable from VHI.**
  - None passes the declared rule, so no model was re-scored on 2021–2022.
  - C1's kappa is slightly above VHI tuned for Model A (0.24 vs 0.22), but every gap is inside the fold-to-fold spread (SD about 0.05 in F1).
- **Random Forest was not run.** It uses the same features, and the bottleneck is the label, not the learner.
- **Sentinel-2 adds nothing.** Model B equals the Landsat-only model on the same years.
- **Conclusion for monthly SPEI-3 classes at 5 km:** machine learning on these satellite features does not add skill over VHI. The limits are the label and the monthly composites, not the classifier:
  - only about 76 independent year-months
  - no shared within-month spatial signal
  - vegetation that is decoupled from SPEI by irrigation, trees and wet-year greening
  - acquisition-date artefacts in monthly Landsat composites, mostly in the thermal band (day 4)

### Random Forest in the same CV (Fri 27 Nov, `s15_models.py A rf`, tables `A_v2_rf_cv_*`)
- **Candidates, declared before running:** C4 (RF classifier) and C5 (RF regressor of SPEI-3, cut like C2). Both use the anomaly and cell-mean features of C1/C2.
  - 300 trees, min_samples_leaf 100, max_features sqrt, max_samples 0.7, seed 0. C4 uses class-balanced weights.
  - Same references and the same pass rule as v2.
- **Result (Model A, 11 folds):** C4 0.281 (beats VHI tuned in 5/11 folds), C5 0.259 (3/11), VHI tuned 0.302, VHI Kogan 0.285. The references are identical to the v2 run.
- **No candidate passes, so the test years were not scored.** Random Forest is slightly below XGBoost C1 (0.290). The "RF/XGBoost" core of the plan is now fully tested, and neither beats VHI.

### Which SPEI window does vegetation follow? (`s16_timescale.py`, training years only)

| SPEI window (months) | 1 | 3 | 6 | 9 | 12 |
|---|---|---|---|---|---|
| r with cell-month VHI | 0.49 | 0.53 | **0.57** | 0.54 | 0.49 |
| r with NDVI z | 0.21 | 0.37 | **0.57** | 0.57 | 0.54 |
| r with LST z | **−0.52** | −0.46 | −0.38 | −0.34 | −0.29 |
| r of season means per year (n = 11) | 0.70 | 0.86 | **0.91** | 0.87 | 0.81 |

- **Greenness follows a 6-month window best.** Surface temperature follows the shortest window best.
- **SPEI-3 was never the best match.** It was fixed on day 1 as an assumption ("matches how fast vegetation responds").
- **Switching to SPEI-6 is a spec change.** It would move the target, so it is a decision for the user, not made here. No test-year values were looked at for this check.

**SPEI-6 classes in the same CV (Tue 24 Nov, `s15_models.py A spei6`, training years only, never test-scored).** The user chose to keep SPEI-3 and run SPEI-6 as a side check.
- Mean leave-one-year-out macro F1 / QWK:
  - VHI tuned 0.27 / 0.19
  - VHI Kogan 0.26 / 0.18
  - C1 0.23 / 0.09, C2 0.24 / 0.09, C3 0.26 / 0.10
  - always none 0.26 / 0
- No model passes.
- Every method agrees with the SPEI-6 classes *less* than with the SPEI-3 classes, even though the plain correlation is higher. The class boundaries matter more than the correlation.
- **Decision: the target stays SPEI-3.**

## Day 4: map products (plan day 4, run Tue 24 Nov)

Scripts:
- `s17_vhi_climatology.py`: per-pixel training-year min/max reference.
- `s18_maps.py`: monthly VHI and SPEI maps.
- `s20_finalize_maps.py`: mask, colours, seasonal maps.
- `s19_quicklook.py`: figures.

Output goes to `outputs/maps/` (about 1.2 GB, not in git). Figures are `outputs/figures/maps_<year>.png` and `maps_seasons.png`.

### Products (all Cloud-Optimized GeoTIFF)

| File | Content | Grid |
|---|---|---|
| `vhi_YYYY_MM.tif` | Monthly VHI 0–100 (uint8, 255 = no data), April–October 2014–2026, 91 maps | 30 m UTM 37N, Landsat-aligned, 5,539 × 4,727 px |
| `vhi_season_YYYY.tif` | Mean of the monthly VHI, April–October, needing at least 4 valid months | same |
| `spei3_YYYY_MM.tif`, `spei6_YYYY_MM.tif` | SPEI value (float32, −9999 = no data) | 0.05° CHIRPS grid, EPSG:4326, the 597 cells touching the province |
| `spei3_season_YYYY.tif` | Mean monthly SPEI-3, April–October | same |
| `*_class.tif` | SPEI-3 classes as uint8 (0 severe, 1 moderate, 2 mild, 3 none) | same |

- **Styling:**
  - Every uint8 raster carries an embedded colour table with the drought-class colours, so QGIS and ArcGIS both show the classes without a style file.
  - A QGIS `.qml` with the same name sits next to every raster.
  - The float SPEI rasters need a style in ArcGIS. The `_class` versions do not.
- **Vegetation mask:** the same rule as the sample points. At least 2/3 of the 30 m pixel must be WorldCover tree, shrub, grass or cropland; this covers 46% of the bounding box. Water, built-up land and bare rock are no data, because VHI was only ever evaluated on vegetation.
- **Reference:** the per-pixel min/max come from all training years (s17).
  - For 2021–2022 this is exactly the point-table reference.
  - For training years it includes the mapped year itself, so 2014–2020 and 2023–2026 maps do not equal the leave-one-year-out point features. This is the standard operational choice for VHI maps.
  - The reference was computed block by block (512 px) and cached in `data/ref/`. The Earth Engine asset exports (s17) were queued but never ran (see quota below).

### Checks
- **Maps against the point table, 2021 and 2022:**
  - 98,581 and 102,920 point-months are valid in both.
  - |map − table| has median 0.24, 99th percentile 0.50 and maximum 0.7, which is integer rounding of the map and of the int16 reference.
  - The same pixels are valid in both: 0 point-months valid in only one of them.
- **Grid:** pixel corners sit at 15 m (mod 30), so it is the Landsat grid with no resampling.
- **COG:** layout COG, internal overviews 2/4/8/16, embedded colour tables readable.
- **Seasonal agreement:** province-mean seasonal VHI (vegetation pixels) against seasonal SPEI-3 gives r = 0.83 over all 13 years and 0.86 over training years (`outputs/tables/season_vhi_spei.csv`).
  - The lowest VHI seasons are 2014, 2017, 2021 and 2025 (VHI 28–33). Three of them are the driest SPEI-3 seasons (−1.0 to −1.5); 2014 is only mildly dry (−0.44).
  - The wet seasons 2023 and 2026 have the highest VHI (69–71).
  - 2022 is the outlier, with VHI 31 against SPEI-3 −0.16.
- **QGIS check (QGIS 4.2.3, Tue 24 Nov):** 8 sample maps (seasonal and monthly VHI, SPEI-3/6 values and classes) opened through PyQGIS. All are valid, with the right CRS, no-data value and overviews.
  - The check found that the VHI and SPEI value `.qml` files were invalid XML, because of an unescaped `<` in labels such as "severe (VHI < 20)". QGIS ignored them and fell back to the embedded colour table.
  - Fixed in `s18_maps.qml()` by escaping the labels. All 400 `.qml` files were regenerated and parse as XML, and all 8 samples now load with their styles.
  - The rendered maps were inspected: at full zoom they show fields and centre pivots, and the dam lake and the city are masked.
- **QGIS project:** `outputs/maps/corak_maps.qgz` (built by `s21_qgis_project.py` with QGIS's own Python). It uses relative paths and has 55 layers:
  - OpenStreetMap base map;
  - seasonal VHI and SPEI-3 season classes for every year;
  - monthly VHI and SPEI-3 classes for the test years 2021 and 2022.

### Monthly scene seams (found on day 4)
- **What they look like:** straight-edged blocks in monthly maps. In May 2021 a parallelogram is "severe" while its neighbours are "none".
- **The thermal half carries them:**
  - Across the May 2021 seam, VCI stays at 20–35 while TCI drops from about 100 in the west to about −10 in the east.
  - The number of clear scenes also changes there (1 to 3 per month).
- **Cause:** TOA surface temperature has no atmospheric correction and varies with the acquisition date within the month. So each Landsat path carries its own bias, which does not cancel against the per-pixel reference.
- **Consequences:**
  - Monthly 30 m VHI maps are reliable as province-level signals, but within-month spatial contrasts partly reflect acquisition dates.
  - The seasonal mean damps the seams, and the seasonal maps are the product the evidence supports (r = 0.86 between years).
  - The monthly maps stay as the scored, unchanged Kogan baseline.
- **Possible fix:** correct each scene's LST before compositing, for example against ERA5-Land air temperature at the acquisition time. This would change the VHI definition, so it could only be evaluated in training-year CV.

### Seam fix tried: per-scene air-temperature correction (Thu 26 Nov) — not adopted
- **Pilot (May 2021):** the western path was imaged on cool days and the eastern path on a hot day.
  - Western path, 7 and 23 May: ERA5-Land 2 m air temperature at overpass −4 to −6 °C against the month's 08 UTC mean.
  - Eastern path, 16 May: +3 to +4 °C.
  - That roughly 10 °C gap matches the LST jump across the seam.
- **Correction (`s08c_lst_corrected.py`), fixed before any evaluation, with no fitted parameter:**
  - For each scene, LST_corr = LST_TOA − dTa, where dTa = ERA5-Land T2m at the overpass hour minus that month's mean T2m at 08 UTC, resampled bilinearly. Lakes take the scene's provincial mean dTa.
  - Then the monthly median, as before.
  - Masks and scene counts are identical to `lsttoa_*` in all 13 years (checked): the same valid pixel-months.
- **Features:** `s10_indices.py corr` writes `features_corr.csv.gz`. The scoring scripts read it with `CORAK_FEATURES=features_corr.csv.gz`, and s15 outputs get the tag `_lstcorr`.
- **Adoption rule (fixed before results; separate from the s15 model pass rule):** adopt the corrected LST (VHI definition and maps) if, in Model A's 11-fold training-year CV, VHI tuned's mean macro F1 rises by at least 0.01 AND is higher in at least 6 of 11 folds. Test years are not used.
- **Results, training cells × training years (`lst_correction_eval.txt`, `A_v2_lstcorr_cv_*`):**

  | Check | Original LST | Corrected LST |
  |---|---|---|
  | Year-months where TCI follows the overpass weather (r(TCI, dTa) < −0.5 across cells) | 27 of 76 | 5 of 76 |
  | Median r(TCI, dTa) | −0.36 | +0.13 |
  | VHI vs SPEI-3, cell-month r overall | 0.526 | 0.552 |
  | VHI vs SPEI-3, within-month r | 0.010 | 0.041 |
  | CV macro F1, VHI tuned (deciding) | 0.302 | 0.293 (higher in 5/11 folds) |
  | CV macro F1, VHI Kogan | 0.285 | 0.284 |
  | CV macro F1, best model (C1) | 0.290 | 0.297; no candidate passes the s15 rule |

- **Direct seam check, May 2021 (no labels used):** points within 5 km of the path edge (n = 6,568).
  - TCI jumps by −34.4 across the edge (east minus west) with the original LST, and by +9.0 with the corrected LST.
  - VCI is flat across the edge (−0.6).
  - So the correction removes most of the seam but overshoots, consistent with the median r(TCI, dTa) of +0.13: k = 1 is a little too strong. k was not tuned, by design.
- **Seasonal agreement, training cells × training years:**

  | Scale | Original LST | Corrected LST |
  |---|---|---|
  | Between years (year means) | r = 0.853 | r = 0.810 |
  | Per cell and season | r = 0.735 | r = 0.715 |

  The correction slightly weakens the scale at which VHI works best.
- **Decision:**
  - The scored VHI baseline and the VHI definition stay as they are. The correction fails the adoption rule: VHI tuned falls by 0.009 and is higher in only 5/11 folds. The models move within noise.
  - Seam-corrected maps were not built. Monthly maps would lose most of their seams, but the seasonal maps (the main product) would agree slightly less with SPEI-3.
  - Five year-months still show r(TCI, dTa) < −0.5 after correction (for example June 2026 −0.75, October 2019 −0.64).
  - `s08c`, `features_corr.csv.gz` and the `_lstcorr` tables are kept as the record of this check.
- **Interpretation (a hypothesis, not tested further):** removing most of the date-driven part of TCI did not improve within-month class agreement (r 0.010 to 0.041). This is consistent with the day-3 view that the label scale (5 km SPEI-3 against 30 m vegetation) is the main limit. But only this one correction (k = 1, 11 km ERA5-Land, residual seams) was tried.

### Earth Engine quota (Tue 24 Nov)
- **Restricted Mode:** the project used up the Community tier's 150 EECU-hours for November.
  - Concurrency is reduced and batch exports queue indefinitely.
  - The quota resets on 1 December, after the deadline.
- **Decision (user):** stay restricted, build all maps locally, and drop the Earth Engine App. Delivery is COG files for QGIS/ArcGIS.
- **Costs of working restricted:**
  - Monthly maps took about 75 s each.
  - One request hung for 25 minutes. Requests now have a 5-minute deadline and are retried.

## Known risks to check

- **2025 spring frost: confirmed, small and short.**
  - **Event:** blossoms were killed by frost on 10–12 April 2025 and Malatya's apricot harvest collapsed. Sources: Daily Sabah and FreshPlaza (2025 reports).
  - **Check (Thu 26 Nov), training cells and training years, cell-months:**
    - VHI and NDVI z were each regressed on SPEI-3 per month and land-cover group, with each year left out of its own fit.
    - Mean NDVI z residual for trees (WorldCover 10) in 2025: −1.04 in April and −0.83 in May, the 2nd-lowest of 11 years. Cropland: −0.34 and +0.14. Grass/shrub: −0.37 and +0.20.
    - The tree-minus-crop gap is −0.70 (April) and −0.97 (May), the most negative of all 11 years. The next worst May gap is −0.55, in 2015.
    - Trees are back to normal from June. VHI shows the same pattern: tree residual −18.6 in May against +0.7 for cropland.
  - **Effect:** tree/shrub is about 5% of pixel-months, so few cell-month votes can change. No change to the pipeline, because no model was frozen. Frost-damaged trees in April–May 2025 will look like drought in the VHI maps; mention this when showing 2025.
- **February 2023 earthquakes: no step change found at the sample points.**
  - **Check (Thu 26 Nov):** summer (June–August) NDVI, 2023–2026 mean minus 2014–2020 mean, for 14,779 points.
  - **Drops above 0.15:** 1.15% of points. The placebo comparison (2018–20 against 2014–17, before the earthquakes) gives 0.83%.
  - **The largest cluster is the Karakaya reservoir shoreline** north of the city, not earthquake damage. NDVI there goes below 0 in high-water years such as 2019, 2024 and 2026.
  - **Away from water, the drops are gradual:** for 38 points within 15 km of the city, the largest one-year fall is spread over 2017–2026, and only 2 points have it in 2023.
  - **Water pixel-months (NDVI < 0):** 0.18% of pixel-months, with a median VHI of 50, so they are not read as drought.
    - Dropping them changes the VHI Kogan class of 27 of 48,035 scored cell-months: 24 train, 3 spatial test, none in the temporal or strict test sets.
  - **Maps:** the vegetation mask is WorldCover 2021, so shoreline pixels that flood in other years stay in the maps.
- **CHIRPS rain gauges.** CHIRPS uses few gauges in eastern Türkiye. TerraClimate agreement (r = 0.84) is the current safeguard.
