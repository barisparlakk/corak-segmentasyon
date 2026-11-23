"""Shared settings for ÇORAK. Every script imports from here so the
study area, grids and calendar are defined exactly once."""
from pathlib import Path

GEE_PROJECT = "corak-515608"

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
AOI_DIR = DATA / "aoi"
CLIMATE_DIR = DATA / "climate"
FIG_DIR = ROOT / "outputs" / "figures"

# Study area: Malatya province (FAO GAUL 2025, level 1)
GAUL_ASSET = "FAO/GAUL/2025/level1"
PROVINCE = "Malatya"
COUNTRY = "Türkiye"  # value of GAUL0_NAME; checked in s01_aoi.py
AOI_GEOJSON = AOI_DIR / "malatya.geojson"

# Analysis grid for satellite data: UTM 37N, 30 m (Landsat-aligned)
CRS_UTM = "EPSG:32637"
PIXEL_M = 30
# Landsat C2 pixel corners sit at x, y = 15 (mod 30) in EPSG:32637 (read from a real
# LC09 scene over Malatya in s01_aoi.py). Every 30 m export uses this transform.
UTM_TRANSFORM = [30, 0, 15, 0, -30, 15]

# Climate (label) grid = native CHIRPS grid, 0.05 degree, EPSG:4326.
# Cell (row, col) is derived ONLY from this transform, here and in Earth Engine.
CLIM_RES = 0.05
CLIM_X0 = -180.0
CLIM_Y0 = 50.0


def cell_rc(lon, lat):
    """Row/col of the climate cell containing (lon, lat)."""
    import numpy as np
    col = np.floor((np.asarray(lon) - CLIM_X0) / CLIM_RES).astype(int)
    row = np.floor((CLIM_Y0 - np.asarray(lat)) / CLIM_RES).astype(int)
    return row, col


# Calendar
CLIM_START, CLIM_END = 1991, 2026          # climate download
CLIM_LAST_MONTH = "2026-09"               # last complete month in CHIRPS (checked in s02)
REF_START, REF_END = 1991, 2020           # SPEI reference period (WMO normal)
STUDY_START, STUDY_END = 2014, 2026       # satellite growing seasons (2026: April-September only)
SEASON_MONTHS = list(range(4, 11))        # April-October
S2_START = 2019                           # Sentinel-2 model (Model B)

# Drought classes from SPEI-3 (upper bounds are exclusive)
CLASS_NAMES = ["none", "mild", "moderate", "severe"]
SPEI_CUTS = [-1.5, -1.0, -0.5]            # severe < -1.5 <= moderate < -1.0 <= mild < -0.5 <= none
VHI_CUTS = [20, 30, 40]                   # Kogan: severe < 20 <= moderate < 30 <= mild < 40 <= none


def ee_init():
    import ee
    ee.Initialize(project=GEE_PROJECT)
    return ee
