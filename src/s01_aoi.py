"""Step 2-3: Malatya boundary, and check that the 30 m grid lines up with Landsat.

Output: data/aoi/malatya.geojson
Prints: province area, bounding box, Landsat pixel-corner offset.
"""
import json
import warnings

import geopandas as gpd

from config import AOI_DIR, AOI_GEOJSON, CRS_UTM, GAUL_ASSET, PROVINCE, ee_init

warnings.filterwarnings("ignore", category=DeprecationWarning)
ee = ee_init()

AOI_DIR.mkdir(parents=True, exist_ok=True)

fc = ee.FeatureCollection(GAUL_ASSET).filter(ee.Filter.eq("GAUL1_NAME", PROVINCE))
props = fc.first().toDictionary(["GAUL0_NAME", "GAUL1_NAME", "GAUL1_CODE"]).getInfo()
print("Boundary record:", props, "| features:", fc.size().getInfo())

geojson = fc.geometry().getInfo()
AOI_GEOJSON.write_text(json.dumps({"type": "Feature", "properties": props, "geometry": geojson}))

aoi = gpd.read_file(AOI_GEOJSON)
area_km2 = aoi.to_crs(CRS_UTM).area.iloc[0] / 1e6
print(f"Province area: {area_km2:,.0f} km2")
print("Bounding box (lon/lat):", [round(v, 3) for v in aoi.total_bounds])

# Landsat alignment: read an actual Landsat 8/9 L2 scene's transform over Malatya
geom = fc.geometry()
scene = (ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterBounds(geom.centroid(1))
         .filterDate("2024-07-01", "2024-08-01").first())
proj = scene.select("SR_B4").projection().getInfo()
tx = proj["transform"]
print("Landsat scene CRS:", proj.get("crs"), "| transform:", tx)
print("Pixel-corner offset (x mod 30, y mod 30):", tx[2] % 30, tx[5] % 30)
