#!/usr/bin/env python3
"""Summer Sentinel-2 median composite + Copernicus DEM terrain features for a study box (odc-stac, AWS Earth Search).

Bands: blue, green, red, nir, swir16, swir22 (20 m), scaled to reflectance with each item's own offset;
cloud/shadow/snow pixels are masked with the L2A scene classification before the median.
Terrain: elevation, slope (degrees) and local relief (elevation minus 1 km mean) from Copernicus DEM GLO-30.
Output: one GeoTIFF stack <out>.tif with band descriptions.

Usage: composite.py --bbox=-104.0,45.9,-103.3,46.6 --years 2022,2023,2024 --months 6,7,8,9 --out data/nd_stack
"""
import argparse
import os

import numpy as np
import odc.stac
import pystac_client
import rasterio
import xarray as xr
from scipy import ndimage

STAC = "https://earth-search.aws.element84.com/v1"
import os as _os
for _k, _v in {"GDAL_HTTP_TIMEOUT": "60", "GDAL_HTTP_CONNECTTIMEOUT": "20", "GDAL_HTTP_MAX_RETRY": "4",
               "GDAL_HTTP_RETRY_DELAY": "3", "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "AWS_NO_SIGN_REQUEST": "YES"}.items():
    _os.environ.setdefault(_k, _v)          # without a timeout a stalled HTTP range read hangs a worker forever

BANDS = ["blue", "green", "red", "nir", "swir16", "swir22", "scl"]
BAD = [0, 1, 3, 8, 9, 10, 11]          # nodata, saturated, shadow, cloud, cirrus, snow


def retry(fn, tries=5):
    import time
    for i in range(tries):
        try:
            return fn()
        except Exception as e:                      # STAC API drops connections under load
            print(f"  retry {i + 1}: {str(e)[:100]}", flush=True)
            time.sleep(5 * (i + 1))
    raise SystemExit("giving up after retries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", required=True)
    ap.add_argument("--years", default="2022,2023,2024")
    ap.add_argument("--months", default="6,7,8,9")
    ap.add_argument("--max-cloud", type=float, default=20)
    ap.add_argument("--res", type=float, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--one-pass", action="store_true",
                    help="compute all bands in one dask graph (SCL mask read once); faster, no per-band cache")
    a = ap.parse_args()
    bbox = [float(v) for v in a.bbox.split(",")]
    cat = pystac_client.Client.open(STAC)
    items = []
    for y in a.years.split(","):
        for m in a.months.split(","):
            m = int(m); end = 30 if m in (6, 9) else 31
            items += retry(lambda: list(cat.search(collections=["sentinel-2-l2a"], bbox=bbox,
                                     datetime=f"{y}-{m:02d}-01/{y}-{m:02d}-{end}",
                                     query={"eo:cloud_cover": {"lt": a.max_cloud}}).items()))
    print(f"{len(items)} scenes", flush=True)
    lon0, lat0 = bbox[0], bbox[1]
    epsg = 32600 + int((lon0 + 180) // 6) + 1                      # UTM zone of the box (northern hemisphere)
    ds = odc.stac.load(items, bands=BANDS, bbox=bbox, crs=f"EPSG:{epsg}", resolution=a.res, groupby="solar_day",
                       chunks={"x": 1024, "y": 1024}, fail_on_error=False)   # skip unreadable scenes; bands are validated below
    ok = ~ds.scl.isin(BAD)
    layers, names = [], []
    import dask
    dask.config.set(scheduler="threads", num_workers=6)
    if a.one_pass:
        refl = (ds[BANDS[:-1]].where(ok).astype("float32") * 1e-4 - 0.1)
        refl = refl.where(refl > -0.1)
        med = retry(lambda: refl.median("time").compute(), tries=3)
        for b in BANDS[:-1]:
            frac = float(np.isfinite(med[b].values).mean())
            if frac < 0.5:
                raise SystemExit(f"band {b}: only {frac:.0%} valid pixels")
            layers.append(med[b]); names.append(b)
            print(f"  composite {b}: {frac:.0%} valid (one pass)", flush=True)
    for b in ([] if a.one_pass else BANDS[:-1]):
        v = ds[b].where(ok & (ds[b] > 0)).astype("float32") * 1e-4
        # baseline >= 04.00 (from 25 Jan 2022) adds +1000 DN; all scenes here are 2022+, so subtract 0.1
        v = v - 0.1
        part = f"{a.out}_part_{b}.nc"                # finished bands are cached: a later failure does not lose them
        if os.path.exists(part):
            med = xr.open_dataarray(part).load()
            layers.append(med); names.append(b)
            print(f"  composite {b}: cached", flush=True)
            continue
        for attempt in range(3):                      # transient HTTP failures: retry the whole band
            try:
                med = v.median("time").compute()
                med.drop_vars([c for c in med.coords if c not in ("x", "y", "spatial_ref")], errors="ignore").to_netcdf(part)
                break
            except Exception as e:
                print(f"  {b}: attempt {attempt + 1} failed: {str(e)[:120]}", flush=True)
        else:
            raise SystemExit(f"band {b} failed 3 times")
        frac = float(np.isfinite(med.values).mean())
        if frac < 0.5:
            raise SystemExit(f"band {b}: only {frac:.0%} valid pixels")
        layers.append(med); names.append(b)
        print(f"  composite {b}: {frac:.0%} valid", flush=True)
    st = xr.concat(layers, "band").assign_coords(band=names)
    dem_items = retry(lambda: list(cat.search(collections=["cop-dem-glo-30"], bbox=bbox).items()))
    dem = odc.stac.load(dem_items, bands=["data"], bbox=bbox, crs=f"EPSG:{epsg}", resolution=a.res,
                        groupby="solar_day").data.max("time").astype("float32")
    dem = dem.reindex_like(st.isel(band=0), method="nearest")
    z = dem.values
    gy, gx = np.gradient(z, a.res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    relief = z - ndimage.uniform_filter(z, size=int(1000 / a.res))
    arr = np.concatenate([st.values, z[None], slope[None], relief[None]]).astype("float32")
    names += ["elevation", "slope", "relief_1km"]
    gb = st.odc.geobox
    with rasterio.open(a.out + ".tif", "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
                       dtype="float32", crs=str(gb.crs), transform=gb.transform, compress="deflate", nodata=np.nan) as dst:
        dst.write(arr)
        for i, n in enumerate(names, 1):
            dst.set_band_description(i, n)
    print(f"-> {a.out}.tif {arr.shape}", flush=True)


if __name__ == "__main__":
    main()
