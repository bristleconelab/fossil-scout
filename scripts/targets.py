#!/usr/bin/env python3
"""Turn a prospectivity map into field targets: keep only the target formation's outcrop, then list high-score patches
far from every known collection.

Inputs: prospectivity GeoTIFF (train.py), formation polygons (Macrostrat geologic_units/map GeoJSON), PBDB collections.
Outputs (in --out): targets.csv (patch centre, area, mean score, distance to nearest known collection), targets.png map,
and filter metrics: share of known sites inside the outcrop, capture curve before/after the geology filter.

Usage: targets.py --prob out/nd_model/prospectivity_test.tif --formation data/hellcreek_polygons.geojson \
                  --pbdb data/pbdb_hellcreek.csv --name montana --out out/targets_mt [--top 0.05] [--min-km 2]
"""
import argparse
import csv
import json
import os

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.features import rasterize
from scipy import ndimage

from train import precise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob", required=True)
    ap.add_argument("--formation", required=True)
    ap.add_argument("--pbdb", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=float, default=0.05, help="top share of the OUTCROP area kept as targets")
    ap.add_argument("--min-km", type=float, default=2.0)
    ap.add_argument("--min-ha", type=float, default=2.0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    with rasterio.open(a.prob) as s:
        p = s.read(1); tr, crs = s.transform, s.crs
    H, W = p.shape
    poly = gpd.read_file(a.formation).to_crs(crs)
    outcrop = rasterize([(g, 1) for g in poly.geometry if g is not None], out_shape=(H, W), transform=tr).astype(bool)
    valid = np.isfinite(p)
    t = Transformer.from_crs(4326, crs, always_xy=True)
    allpts, prec = [], []
    for r in csv.DictReader(open(a.pbdb)):
        x, y = t.transform(float(r["lng"]), float(r["lat"]))
        c, rw = ~tr * (x, y)
        if 0 <= rw < H and 0 <= c < W:
            allpts.append((x, y))
            if precise(r):
                prec.append((int(rw), int(c)))
    inside = float(np.mean([outcrop[r, c] for r, c in prec])) if prec else float("nan")

    def capture(mask, frac):
        v = p[mask & valid]
        thr = np.quantile(v, 1 - frac)
        return float(np.mean([p[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2].max() >= thr and mask[r, c] for r, c in prec]))

    metrics = {"region": a.name, "precise_sites": len(prec), "sites_inside_outcrop": round(inside, 3),
               "outcrop_share_of_area": round(float(outcrop[valid].mean()), 3),
               "capture_top5_all_area": round(capture(np.ones_like(outcrop), 0.05), 3),
               "capture_top5_outcrop_only": round(capture(outcrop, 0.05), 3)}
    # targets: top share of the outcrop, connected patches, far from every known collection (any precision)
    thr = np.quantile(p[outcrop & valid], 1 - a.top)
    hot = (p >= thr) & outcrop & valid
    hot = ndimage.binary_opening(hot, iterations=1)
    lab, n = ndimage.label(hot)
    px_ha = abs(tr.a * tr.e) / 1e4
    kx = np.array([q[0] for q in allpts]); ky = np.array([q[1] for q in allpts])
    to_ll = Transformer.from_crs(crs, 4326, always_xy=True)
    rows = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        m = lab[sl] == i
        ha = m.sum() * px_ha
        if ha < a.min_ha:
            continue
        rr, cc = np.nonzero(m)
        r0, c0 = rr.mean() + sl[0].start, cc.mean() + sl[1].start
        x, y = tr * (c0 + 0.5, r0 + 0.5)
        dkm = float(np.min(np.hypot(kx - x, ky - y)) / 1000) if len(kx) else 99.0
        if dkm < a.min_km:
            continue
        lon, lat = to_ll.transform(x, y)
        rows.append({"lat": round(lat, 5), "lon": round(lon, 5), "area_ha": round(ha, 1),
                     "mean_score": round(float(p[sl][m].mean()), 3), "nearest_known_km": round(dkm, 1)})
    rows.sort(key=lambda r: (-r["mean_score"] * np.sqrt(r["area_ha"])))
    for k, r in enumerate(rows, 1):
        r["rank"] = k
    with open(os.path.join(a.out, "targets.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "lat", "lon", "area_ha", "mean_score", "nearest_known_km"])
        w.writeheader(); w.writerows(rows)
    metrics["targets"] = len(rows)
    json.dump(metrics, open(os.path.join(a.out, "metrics.json"), "w"), indent=1)
    print(json.dumps(metrics))
    fig, ax = plt.subplots(figsize=(8, 8))
    base = np.where(outcrop, 0.55, 0.85); base[~valid] = 1
    ax.imshow(base, cmap="Greys_r", vmin=0, vmax=1)
    ax.imshow(np.where(hot, 1.0, np.nan), cmap="autumn", alpha=0.9, interpolation="nearest")
    pr = [q for q in prec]
    ax.plot([c for _, c in pr], [r for r, _ in pr], "o", mfc="none", mec="#0b3d91", ms=5, mew=1.1, label="known collections (PBDB)")
    for r in rows[:15]:
        x, y = t.transform(r["lon"], r["lat"]); c, rw = ~tr * (x, y)
        ax.plot(c, rw, "*", color="#00c853", ms=13, mec="black", mew=0.6)
        ax.text(c + 25, rw, str(r["rank"]), fontsize=8, weight="bold")
    ax.plot([], [], "*", color="#00c853", ms=11, mec="black", label=f"new targets > {a.min_km:g} km from any known site (top 15)")
    ax.set_title(f"{a.name}: dark grey = Hell Creek outcrop (Macrostrat), red = top {a.top:.0%} of outcrop by score")
    ax.legend(loc="lower right", fontsize=8); ax.axis("off")
    fig.savefig(os.path.join(a.out, "targets.png"), dpi=95, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
