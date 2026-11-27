"""Day 4, step 1: per-pixel VHI reference (min/max) for the whole province, as Earth Engine assets.

For each calendar month April-October, over the TRAINING YEARS (2014-2026 without 2021-2022):
  monthly median NDVI  (Landsat 8/9 C2 L2, same masking/scaling as s08_landsat.prep)
  monthly median LST   (Landsat 8/9 C2 TOA band 10 with NDVI emissivity, s08b_landsat_lst.prep)
  -> per-pixel min and max over the years, kept only where at least 5 years are valid
One asset per month: projects/corak-515608/assets/vhi_ref_MM with bands
  NDVI_min, NDVI_max (x 10000) and LST_min, LST_max (deg C x 100), int16, 30 m UTM 37N grid.
Maps (s18) and the Earth Engine App reuse these, so VHI for any month is a cheap computation.

Note: this is the operational reference (all training years). At the sample points it equals the
test-year reference used in s10, so 2021-2022 map values must match the point table exactly.

Usage: python s17_vhi_climatology.py          (starts the export tasks)
       python s17_vhi_climatology.py status   (shows task states)
"""
import json
import sys

from config import AOI_GEOJSON, CRS_UTM, GEE_PROJECT, SEASON_MONTHS, STUDY_END, STUDY_START, UTM_TRANSFORM, ee_init
import s08_landsat
import s08b_landsat_lst

ee = ee_init()
TEST_YEARS = json.loads((AOI_GEOJSON.parents[1] / "climate" / "split.json").read_text())["test_years"]
TRAIN_YEARS = [y for y in range(STUDY_START, STUDY_END + 1) if y not in TEST_YEARS]
MIN_YEARS = 5
aoi = ee.Geometry(json.loads(AOI_GEOJSON.read_text())["geometry"])


def monthly(year, month):
    """Monthly median NDVI (L2) and LST (TOA) for one year-month, as an image with 2 bands."""
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, "month")
    ndvi = s08_landsat.L89.filterDate(start, end).map(s08_landsat.prep).select("NDVI").median()
    lst = s08b_landsat_lst.TOA.filterDate(start, end).map(s08b_landsat_lst.prep).median()
    return ee.Image.cat(ndvi.rename("NDVI"), lst.rename("LST"))


def reference(month):
    ic = ee.ImageCollection([monthly(y, month) for y in TRAIN_YEARS])
    n = ic.count()
    mn, mx = ic.min(), ic.max()
    img = ee.Image.cat(mn.select("NDVI").multiply(10000).rename("NDVI_min"),
                       mx.select("NDVI").multiply(10000).rename("NDVI_max"),
                       mn.select("LST").multiply(100).rename("LST_min"),
                       mx.select("LST").multiply(100).rename("LST_max"))
    ok_n = n.select("NDVI").gte(MIN_YEARS)
    ok_l = n.select("LST").gte(MIN_YEARS)
    img = img.updateMask(ee.Image.cat(ok_n, ok_n, ok_l, ok_l))
    return img.round().toInt16().clip(aoi)


def asset_id(month):
    return f"projects/{GEE_PROJECT}/assets/vhi_ref_{month:02d}"


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        for t in ee.data.getTaskList()[:20]:
            if t.get("description", "").startswith("vhi_ref"):
                print(t["description"], t["state"], t.get("error_message", ""))
        sys.exit()
    for m in SEASON_MONTHS:
        task = ee.batch.Export.image.toAsset(
            image=reference(m), description=f"vhi_ref_{m:02d}", assetId=asset_id(m), region=aoi,
            crs=CRS_UTM, crsTransform=UTM_TRANSFORM, maxPixels=1e10, pyramidingPolicy={".default": "mean"})
        task.start()
        print("started", asset_id(m))
