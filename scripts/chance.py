#!/usr/bin/env python3
"""Chance level of the capture curve under the same rule train.py uses for sites.

train.py counts a site as captured when any pixel of its 3x3 window reaches the top x % threshold. A random point
passes that test more often than x %, so the fair baseline is the share of random valid points whose 3x3 window
reaches the threshold. This script measures it on a prospectivity map.

Usage: chance.py --prob out/nd_model/prospectivity_test.tif [--prob out/nd_model/prospectivity.tif] --out docs/chance_baseline.json
"""
import argparse
import json

import numpy as np
import rasterio


def chance(path, n, seed, fracs=(0.05, 0.10, 0.20)):
    with rasterio.open(path) as s:
        p = s.read(1)
    valid = np.isfinite(p)
    q = np.nan_to_num(p, nan=-1)
    rr, cc = np.nonzero(valid[1:-1, 1:-1])
    k = np.random.default_rng(seed).choice(len(rr), size=min(n, len(rr)), replace=False)
    rr, cc = rr[k] + 1, cc[k] + 1
    win = np.max(np.stack([q[rr + dr, cc + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1)]), axis=0)
    out = {}
    for f in fracs:
        thr = np.quantile(p[valid], 1 - f)
        out[f"top{int(f * 100)}pct"] = round(float((win >= thr).mean()), 3)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prob", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--points", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    res = {"rule": "random valid point counts as captured if any pixel of its 3x3 window is in the top x %",
           "points": a.points, "seed": a.seed, "maps": {p: chance(p, a.points, a.seed) for p in a.prob}}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
