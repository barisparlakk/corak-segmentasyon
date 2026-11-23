"""Real Malatya maps for the advisor presentation (PNG, UTM 37N, province clipped, transparent outside).

  elevation.png   Copernicus DEM, hypsometric colours blended with hillshade
  slope.png       slope in degrees, derived from the DEM
  truecolor.png   Sentinel-2 L2A median, 1 June - 15 July 2025, Cloud Score+ masked
  ndvi.png        NDVI of the same composite
  city.png        true-colour close-up around Malatya city (same composite)
  landcover.png   ESA WorldCover 2021 (official class colours)

Output: outputs/figures/maps/*.png
"""
import json
import urllib.request
import warnings

from config import AOI_GEOJSON, CRS_UTM, ROOT, ee_init

warnings.filterwarnings("ignore", category=DeprecationWarning)
ee = ee_init()
OUT = ROOT / "outputs" / "figures" / "maps"
OUT.mkdir(parents=True, exist_ok=True)

aoi = ee.Geometry(json.loads(AOI_GEOJSON.read_text())["geometry"])
fc = ee.FeatureCollection([ee.Feature(aoi)])
region = aoi.bounds(1).buffer(3000).bounds(1)
outline = ee.Image().byte().paint(fc, 0, 4).visualize(palette=["16323A"])


def save(img, name, reg=region, dim=1400, clip=True, line=True):
    if clip:
        img = img.updateMask(ee.Image.constant(1).clip(aoi).mask())
    if line:
        img = ee.ImageCollection([img, outline]).mosaic()
    url = img.getThumbURL({"region": reg, "dimensions": dim, "format": "png", "crs": CRS_UTM})
    urllib.request.urlretrieve(url, OUT / f"{name}.png")
    print("saved", name)


dem = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1").select("DEM").mosaic().setDefaultProjection(
    ee.Projection("EPSG:4326").atScale(30))
hs = ee.Terrain.hillshade(dem.reproject(ee.Projection(CRS_UTM).atScale(90)))
hypso = dem.visualize(min=650, max=2600, palette=["4F7942", "9DB86B", "E7DFA0", "D9B27C", "B07B4F", "8A5A44", "EFE8E0"])
elev = hypso.toFloat().multiply(hs.divide(255).multiply(0.55).add(0.45)).uint8()
save(elev, "elevation")

slope = ee.Terrain.slope(dem.reproject(ee.Projection(CRS_UTM).atScale(90)))
save(slope.visualize(min=0, max=35, palette=["F5F1E8", "E7C9A8", "D08A5B", "A8461F", "5E2410"]), "slope")

s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(aoi).filterDate("2025-06-01", "2025-07-16")
      .linkCollection(ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"), ["cs_cdf"])
      .map(lambda i: i.updateMask(i.select("cs_cdf").gte(0.6))).median())
tc = s2.visualize(bands=["B4", "B3", "B2"], min=150, max=2600, gamma=1.15)
save(tc, "truecolor")
ndvi = s2.normalizedDifference(["B8", "B4"])
# water (NDVI < 0) in blue, then bare/dry brown -> dense green
save(ndvi.visualize(min=-0.2, max=0.65,
                    palette=["2B6A99", "2B6A99", "9C6B3C", "D9C38A", "E8E3A0", "A6C76A", "4E9A3A", "1F5F2A"]), "ndvi")

city = ee.Geometry.Rectangle([38.17, 38.27, 38.45, 38.44])
save(tc, "city", reg=city, dim=1400, clip=False, line=False)

wc = ee.ImageCollection("ESA/WorldCover/v200").first()
classes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
pal = ["006400", "FFBB22", "FFFF4C", "F096FF", "FA0000", "B4B4B4", "F0F0F0", "0064C8", "0096A0", "00CF75", "FAE6A0"]
save(wc.remap(classes, list(range(len(classes)))).visualize(min=0, max=len(classes) - 1, palette=pal), "landcover")
