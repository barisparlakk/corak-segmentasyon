"""Day 4, step 5: QGIS project for the map products (run with QGIS's own Python).

Writes outputs/maps/corak_maps.qgz with relative paths, layers grouped and only one map visible:
  OpenStreetMap base map (needs internet)
  VHI season (Apr-Oct), one layer per year          - 2021 visible
  SPEI-3 season classes, one layer per year
  VHI monthly 2021 / 2022 (the test years)
  SPEI-3 monthly classes 2021 / 2022
Styles come from the .qml files beside each map (QGIS loads them automatically).

  QT_QPA_PLATFORM=offscreen PYTHONPATH=<QGIS.app>/Contents/Resources/python3.12/site-packages \
      <QGIS.app>/Contents/MacOS/python s21_qgis_project.py
"""
from pathlib import Path

from qgis.core import (Qgis, QgsApplication, QgsCoordinateReferenceSystem, QgsProject, QgsRasterLayer, QgsReferencedRectangle)

MAPS = Path(__file__).resolve().parents[1] / "outputs" / "maps"
YEARS = range(2014, 2027)
SEASON_MONTHS = range(4, 11)
SHOW = "vhi_season_2021"

app = QgsApplication([], False)
app.initQgis()
proj = QgsProject.instance()
proj.setCrs(QgsCoordinateReferenceSystem("EPSG:32637"))
proj.setFilePathStorage(Qgis.FilePathType.Relative)
root = proj.layerTreeRoot()


def add(group, stem, title):
    path = MAPS / f"{stem}.tif"
    if not path.exists():
        return None
    lyr = QgsRasterLayer(str(path), title)
    assert lyr.isValid(), path
    proj.addMapLayer(lyr, False)
    node = group.addLayer(lyr)
    node.setItemVisibilityChecked(stem == SHOW)
    return lyr


groups = [
    ("VHI season (Apr-Oct, 30 m, vegetation only)", [(f"vhi_season_{y}", f"VHI season {y}") for y in YEARS]),
    ("SPEI-3 season classes (0.05 deg)", [(f"spei3_season_{y}_class", f"SPEI-3 season {y}") for y in YEARS]),
]
for y in (2021, 2022):
    groups.append((f"VHI monthly {y} (test year)", [(f"vhi_{y}_{m:02d}", f"VHI {y}-{m:02d}") for m in SEASON_MONTHS]))
    groups.append((f"SPEI-3 monthly classes {y}", [(f"spei3_{y}_{m:02d}_class", f"SPEI-3 {y}-{m:02d}")
                                                   for m in SEASON_MONTHS]))
first = None
for i, (name, layers) in enumerate(groups):
    g = root.addGroup(name)
    g.setExpanded(i == 0)
    for stem, title in layers:
        lyr = add(g, stem, title)
        if stem == SHOW:
            first = lyr

osm = QgsRasterLayer("type=xyz&url=https://tile.openstreetmap.org/{z}/{x}/{y}.png&zmax=19&zmin=0",
                     "OpenStreetMap", "wms")
proj.addMapLayer(osm, False)
root.addLayer(osm)

ext = first.extent()
ext.grow(5000)
proj.viewSettings().setDefaultViewExtent(QgsReferencedRectangle(ext, first.crs()))
out = MAPS / "corak_maps.qgz"
assert proj.write(str(out)), proj.error()
print("wrote", out, "with", len(proj.mapLayers()), "layers")
app.exitQgis()
