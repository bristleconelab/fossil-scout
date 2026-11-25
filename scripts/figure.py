#!/usr/bin/env python3
"""Side-by-side map of the training (North Dakota) and transfer (Montana) prospectivity: red = top 5 % of each area by
model score, circles = PBDB collections with precise coordinates.

Usage: figure.py --train out/nd_model/prospectivity.tif --test out/nd_model/prospectivity_test.tif \
                 --pbdb data/pbdb_hellcreek.csv --out docs/img/prospectivity_nd_mt.png
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Transformer

PRECISE = {"3", "4", "6", "seconds"}


def panel(ax, path, pbdb, title, top=0.05):
    with rasterio.open(path) as s:
        p = s.read(1); tr, crs = s.transform, s.crs
    H, W = p.shape
    valid = np.isfinite(p)
    thr = np.quantile(p[valid], 1 - top)
    rgb = np.full((H, W, 3), 0.85)
    rgb[~valid] = 1.0
    rgb[valid & (p >= thr)] = (1.0, 0.0, 0.0)
    ax.imshow(rgb, interpolation="nearest")
    t = Transformer.from_crs(4326, crs, always_xy=True)
    pts = []
    for r in csv.DictReader(open(pbdb)):
        if r["latlng_precision"] not in PRECISE:
            continue
        c, rw = ~tr * t.transform(float(r["lng"]), float(r["lat"]))
        if 1 <= rw < H - 1 and 1 <= c < W - 1 and valid[int(rw), int(c)]:
            pts.append((c, rw))
    ax.plot([c for c, _ in pts], [r for _, r in pts], "o", mfc="none", mec="#0b3d91", ms=6, mew=1.5,
            label=f"known fossil collections (PBDB), n={len(pts)}")
    ax.set_title(title); ax.legend(loc="lower right", fontsize=8); ax.axis("off")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--pbdb", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fig, axs = plt.subplots(1, 2, figsize=(12, 7))
    panel(axs[0], a.train, a.pbdb, "North Dakota: training area")
    panel(axs[1], a.test, a.pbdb, "Montana Hell Creek: never seen in training")
    fig.suptitle("Fossil prospectivity: red = top 5 % of the area by model score; circles = known fossil sites")
    fig.savefig(a.out, dpi=100, bbox_inches="tight"); plt.close(fig)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
