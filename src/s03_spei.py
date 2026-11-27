"""Step 8: SPEI-3 for every climate cell, 1991-2025.

Method (Vicente-Serrano et al. 2010):
  1. PET by Hargreaves (FAO-56 eq. 52) from ERA5-Land Tmax/Tmin/Tmean and
     extraterrestrial radiation Ra (FAO-56 eq. 21) at each cell's latitude.
  2. Water balance D = P - PET, summed over 3 months (SPEI-3).
  3. For each cell and calendar month, fit a generalized logistic distribution
     (log-logistic family) to the 1991-2020 reference values by L-moments.
  4. SPEI = standard-normal quantile of the fitted cumulative probability.
Cross-check: the same fit on TerraClimate (P - Penman-Monteith PET), 1991-2024.

Output: data/climate/spei3.nc
"""
import numpy as np
import xarray as xr

from scipy.stats import norm

from config import CLIMATE_DIR, REF_END, REF_START

ds = xr.open_dataset(CLIMATE_DIR / "climate_monthly.nc")


def ra_mm_per_day(lat_deg, doy):
    """Extraterrestrial radiation, FAO-56 eq. 21, converted to mm/day of evaporation."""
    phi = np.deg2rad(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(-np.tan(phi) * np.tan(dec))
    ra = (24 * 60 / np.pi) * 0.0820 * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
    return 0.408 * ra


# ---- 1. Hargreaves PET (mm/month)
time = ds.time.to_index()
mid_doy = (time + np.timedelta64(14, "D")).day_of_year.values
days = time.days_in_month.values
lat2d = np.broadcast_to(ds.lat.values[:, None], (ds.sizes["lat"], ds.sizes["lon"]))
ra = ra_mm_per_day(lat2d[None], mid_doy[:, None, None])
trange = np.clip(ds.Tmax.values - ds.Tmin.values, 0, None)
pet = 0.0023 * ra * (ds.Tmean.values + 17.8) * np.sqrt(trange) * days[:, None, None]
pet = np.clip(pet, 0, None)
ds["PET"] = (("time", "lat", "lon"), pet)
ds["D"] = ds.P - ds.PET


def glo_spei(acc, months, years):
    """acc: (time, y, x) accumulated balance. For each calendar month, fit a generalized
    logistic distribution (log-logistic family) to the reference years by unbiased
    L-moments (Hosking 1990), as the reference R `SPEI` package does. Unlike the plain
    3-parameter log-logistic PWM fit, this stays valid for negatively skewed samples,
    which dry summer months in Malatya produce."""
    out = np.full(acc.shape, np.nan)
    for m in range(1, 13):
        sel = months == m
        ref = sel & (years >= REF_START) & (years <= REF_END) & ~np.isnan(acc).all(axis=(1, 2))  # Jan-Feb 1991 have no full 3-month window
        x = np.sort(acc[ref], axis=0)                      # (n, y, x), ascending
        n = x.shape[0]
        i = np.arange(1, n + 1)[:, None, None]
        b0 = x.mean(axis=0)
        b1 = (((i - 1) / (n - 1)) * x).mean(axis=0)
        b2 = (((i - 1) * (i - 2) / ((n - 1) * (n - 2))) * x).mean(axis=0)
        l1, l2, l3 = b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0
        k = -l3 / l2                                        # GLO shape = -L-skewness
        small = np.abs(k) < 1e-6
        ks = np.where(small, 1.0, k)
        alpha = np.where(small, l2, l2 * np.sin(ks * np.pi) / (ks * np.pi))
        xi = np.where(small, l1, l1 - alpha * (1 / ks - np.pi / np.sin(ks * np.pi)))
        z = (acc[sel] - xi) / alpha
        with np.errstate(invalid="ignore", divide="ignore"):
            arg = 1 - ks * z
            y = np.where(small, z, -np.log(arg) / ks)
            # outside the distribution's bound: upper tail if k>0, lower tail if k<0
            y = np.where(~small & (arg <= 0), np.where(k > 0, 50.0, -50.0), y)
        cdf = 1 / (1 + np.exp(-y))
        out[sel] = norm.ppf(np.clip(cdf, 1e-6, 1 - 1e-6))
    out[np.isnan(acc)] = np.nan
    return np.clip(out, -3.09, 3.09)


def spei3(d, t):
    acc = d.rolling({d.dims[0]: 3}).sum().values
    idx = t.to_index()
    return glo_spei(acc, idx.month.values, idx.year.values)


if __name__ == "__main__":
    spei = spei3(ds.D, ds.time)
    tc_d = ds.tc_pr - 0.1 * ds.tc_pet_raw          # TerraClimate pet scale factor 0.1 (checked: ~1135 mm/yr)
    spei_tc = spei3(tc_d, ds.time_tc)

    out = xr.Dataset(
        {"spei3": (("time", "lat", "lon"), spei), "P": ds.P, "PET": ds.PET, "D": ds.D},
        coords={"time": ds.time, "lat": ds.lat, "lon": ds.lon},
    )
    out["spei3_tc"] = xr.DataArray(spei_tc, dims=("time_tc", "lat", "lon"),
                                   coords={"time_tc": ds.time_tc, "lat": ds.lat, "lon": ds.lon})
    out.to_netcdf(CLIMATE_DIR / "spei3.nc")
    print("Hargreaves PET, province-wide annual mean 1991-2020:",
          round(float(ds.PET.sel(time=slice("1991", "2020")).groupby("time.year").sum().mean()), 0), "mm")
    print("saved spei3.nc")
