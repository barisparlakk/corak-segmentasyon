"""Day 2, step 4: drought indices and anomaly features from the fixed-pixel time series.

All reference statistics use TRAINING YEARS ONLY (every study year except the test years),
per pixel and calendar month, and need at least MIN_YEARS valid years.

Baseline (Kogan):
  VCI = 100 (NDVI - NDVImin) / (NDVImax - NDVImin)
  TCI = 100 (LSTmax - LST)  / (LSTmax - LSTmin), with LST = LST_TOA for every pixel (s08b)
  VHI = 0.5 VCI + 0.5 TCI            -> VHI classes use the value clipped to 0-100
Model features (robust to a single contaminated year fixing min or max):
  z-scores of NDVI, EVI, SAVI, NDMI, LST_TOA (and Sentinel-2 indices when present) against the
  pixel's own training-year mean and SD for that calendar month, plus the previous month's z.

Output: data/features/features.csv.gz (one row per pixel-month, April-October)
"""
import json

import numpy as np
import pandas as pd

from config import CLIMATE_DIR, DATA, SEASON_MONTHS

FEAT = DATA / "features"
TEST_YEARS = json.loads((CLIMATE_DIR / "split.json").read_text())["test_years"]
MIN_YEARS = 5
LS_VARS = ["NDVI", "EVI", "SAVI", "NDMI", "LST_TOA"]   # LST_TOA: see s08b (L2 ST has an emissivity gap)
S2_VARS = ["S2_NDVI", "S2_EVI", "S2_NDMI", "S2_NDRE"]


def load(prefix):
    files = sorted(FEAT.glob(f"{prefix}_*.csv.gz"))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else None


def add_stats(df, var, train):
    g = df[train].groupby(["pid", "month"])[var]
    st = pd.DataFrame({"mean": g.mean(), "sd": g.std(), "min": g.min(), "max": g.max(), "n": g.count()})
    st.loc[st.n < MIN_YEARS, ["mean", "sd", "min", "max"]] = np.nan
    st.columns = [f"{var}_{c}" for c in st.columns]
    return df.merge(st, left_on=["pid", "month"], right_index=True, how="left")


def build():
    ls = load("landsat").merge(load("lsttoa"), on=["pid", "year", "month"], how="left")
    train = ~ls.year.isin(TEST_YEARS)
    for v in LS_VARS:
        ls = add_stats(ls, v, train)
        ls[f"{v}_z"] = (ls[v] - ls[f"{v}_mean"]) / ls[f"{v}_sd"]
    ls["VCI"] = 100 * (ls.NDVI - ls.NDVI_min) / (ls.NDVI_max - ls.NDVI_min)
    ls["TCI"] = 100 * (ls.LST_TOA_max - ls.LST_TOA) / (ls.LST_TOA_max - ls.LST_TOA_min)
    ls["VHI"] = 0.5 * ls.VCI + 0.5 * ls.TCI
    ls["VHI_clip"] = 0.5 * ls.VCI.clip(0, 100) + 0.5 * ls.TCI.clip(0, 100)

    s2 = load("s2")
    zvars = [f"{v}_z" for v in LS_VARS]
    if s2 is not None:
        tr2 = ~s2.year.isin(TEST_YEARS)
        for v in S2_VARS:
            s2 = add_stats(s2, v, tr2)
            s2[f"{v}_z"] = (s2[v] - s2[f"{v}_mean"]) / s2[f"{v}_sd"]
        keep = ["pid", "year", "month", "S2_NOBS"] + S2_VARS + [f"{v}_z" for v in S2_VARS]
        ls = ls.merge(s2[keep], on=["pid", "year", "month"], how="left")
        zvars += [f"{v}_z" for v in S2_VARS]

    # previous month's anomaly (March exists only for this purpose)
    ls = ls.sort_values(["pid", "year", "month"])
    for v in zvars:
        ls[f"{v}_lag1"] = ls.groupby(["pid", "year"])[v].shift(1)

    out = ls[ls.month.isin(SEASON_MONTHS)]
    drop = [c for c in out.columns if c.endswith(("_mean", "_sd", "_min", "_max", "_n"))]
    out = out.drop(columns=drop)
    out.to_csv(FEAT / "features.csv.gz", index=False)
    return out


if __name__ == "__main__":
    f = build()
    print(f"rows {len(f)} | pixels {f.pid.nunique()} | years {sorted(f.year.unique())}")
    tr = ~f.year.isin(TEST_YEARS)
    print("VHI valid share:", round(f.VHI.notna().mean(), 3))
    out = (f.VCI < -1e-6) | (f.VCI > 100 + 1e-6)          # tolerance: a pixel's own max is 100 +- rounding
    print("VCI outside 0-100, training years:", round(out[tr].mean(), 3), "| test years:", round(out[~tr].mean(), 3))
    print("NDVI_z SD, training years (should be about 1):", round(f.NDVI_z[tr].std(), 2))
