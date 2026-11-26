#!/usr/bin/env python3
"""Fossil-locality prospectivity from a Sentinel-2 + terrain stack, trained on Paleobiology Database collections.

Positives: PBDB collections with precise coordinates (latlng_precision 3, 4, 6 or seconds, latlng_basis not
"based on political unit") inside the stack; each gives
its 3x3 pixel neighbourhood (coordinate error is tens of metres). Background: random valid pixels (presence-background).
Model: gradient boosting on bands, indices and terrain. Validation: spatial block cross-validation (blocks of --block-km)
and the capture curve = share of held-out sites that fall in the top 5/10/20 % of the area by predicted score.
Optional --test-stack: apply the model trained on the whole training area to another region and report its capture curve.

Usage: train.py --stack data/nd_stack.tif --pbdb data/pbdb_hellcreek.csv --out out/nd [--test-stack data/mt_stack.tif]
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from pyproj import Transformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

PRECISE = {"3", "4", "6", "seconds"}
COARSE_BASIS = {"based on political unit"}      # coordinate taken from the county: many decimals, wrong place


def precise(r):
    return r["latlng_precision"] in PRECISE and r["latlng_basis"] not in COARSE_BASIS


def features(path):
    with rasterio.open(path) as s:
        a = s.read().astype("float32")
        names = list(s.descriptions)
        tr, crs = s.transform, s.crs
    b = dict(zip(names, a))
    eps = 1e-6
    extra = {
        "ndvi": (b["nir"] - b["red"]) / (b["nir"] + b["red"] + eps),
        "clay": (b["swir16"] - b["swir22"]) / (b["swir16"] + b["swir22"] + eps),   # clay / mudstone exposures
        "iron": b["red"] / (b["blue"] + eps),                                       # oxidised sediments
        "bright": (b["blue"] + b["green"] + b["red"]) / 3,
    }
    X = np.concatenate([a, np.stack(list(extra.values()))])
    return X, names + list(extra), tr, crs


def sites(pbdb, crs, tr, shape):
    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    out = []
    for r in csv.DictReader(open(pbdb)):
        if not precise(r):
            continue
        x, y = t.transform(float(r["lng"]), float(r["lat"]))
        col, row = ~tr * (x, y)
        row, col = int(row), int(col)
        if 1 <= row < shape[0] - 1 and 1 <= col < shape[1] - 1:
            out.append((row, col, r["collection_no"]))
    return out


def capture(score, valid, pts, fracs=(0.05, 0.10, 0.20)):
    v = score[valid]
    res = {}
    for f in fracs:
        thr = np.quantile(v, 1 - f)
        hit = [score[r - 1:r + 2, c - 1:c + 2].max() >= thr for r, c, _ in pts]
        res[f"top{int(f * 100)}pct"] = round(float(np.mean(hit)), 3) if hit else None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stack", required=True)
    ap.add_argument("--pbdb", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-stack")
    ap.add_argument("--block-km", type=float, default=5)
    ap.add_argument("--bg-per-site", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    X, names, tr, crs = features(a.stack)
    C, H, W = X.shape
    valid = np.all(np.isfinite(X), axis=0)
    pts = [p for p in sites(a.pbdb, crs, tr, (H, W)) if valid[p[0], p[1]]]
    print(f"stack {H}x{W}, {C} features, {len(pts)} precise sites inside", flush=True)
    pos_rc = [(r + dr, c + dc) for r, c, _ in pts for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
    vr, vc = np.nonzero(valid)
    k = rng.choice(len(vr), size=min(len(vr), a.bg_per_site * len(pts) * 9), replace=False)
    bg_rc = list(zip(vr[k], vc[k]))
    rc = np.array(pos_rc + bg_rc)
    y = np.r_[np.ones(len(pos_rc)), np.zeros(len(bg_rc))]
    F = X[:, rc[:, 0], rc[:, 1]].T
    bpx = a.block_km * 1000 / abs(tr.a)
    blocks = (rc[:, 0] // bpx).astype(int) * 10000 + (rc[:, 1] // bpx).astype(int)
    site_block = {(r, c): int(r // bpx) * 10000 + int(c // bpx) for r, c, _ in pts}
    ub = np.unique(blocks); rng.shuffle(ub); folds = np.array_split(ub, 5)
    oof_auc, cap = [], []
    for i, fb in enumerate(folds):
        te = np.isin(blocks, fb)
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, random_state=a.seed)
        m.fit(F[~te], y[~te])
        if len(np.unique(y[te])) == 2:
            oof_auc.append(roc_auc_score(y[te], m.predict_proba(F[te])[:, 1]))
        held = [(r, c, n) for r, c, n in pts if site_block[(r, c)] in set(fb)]
        if held:
            rows_needed = sorted({r for r, _, _ in held})
            score = np.full((H, W), -1.0, dtype="float32")
            flat = X.reshape(C, -1).T
            ok = valid.ravel()
            score.ravel()[ok] = m.predict_proba(flat[ok])[:, 1]
            # capture is measured over the held-out blocks only (area of those blocks)
            rr, cc = np.indices((H, W))
            inblock = np.isin((rr // bpx).astype(int) * 10000 + (cc // bpx).astype(int), fb) & valid
            cap.append((len(held), capture(score, inblock, held)))
        print(f"  fold {i}: test sites {len(held)}, AUC {oof_auc[-1] if oof_auc else float('nan'):.3f}", flush=True)
    wsum = sum(n for n, _ in cap)
    cv_capture = {k: round(sum(n * c[k] for n, c in cap) / wsum, 3) for k in cap[0][1]} if cap else {}
    final = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, random_state=a.seed).fit(F, y)
    flat = X.reshape(C, -1).T; ok = valid.ravel()
    score = np.full(H * W, np.nan, dtype="float32"); score[ok] = final.predict_proba(flat[ok])[:, 1]
    score = score.reshape(H, W)
    with rasterio.open(os.path.join(a.out, "prospectivity.tif"), "w", driver="GTiff", height=H, width=W, count=1,
                       dtype="float32", crs=crs, transform=tr, compress="deflate", nodata=np.nan) as d:
        d.write(score, 1)
    report = {"sites": len(pts), "features": names, "cv_auc_mean": round(float(np.mean(oof_auc)), 3),
              "cv_auc_folds": [round(float(v), 3) for v in oof_auc], "cv_capture": cv_capture,
              "random_baseline": {"top5pct": 0.05, "top10pct": 0.10, "top20pct": 0.20}}
    if a.test_stack:
        Xt, _, trt, crst = features(a.test_stack)
        Ct, Ht, Wt = Xt.shape
        vt = np.all(np.isfinite(Xt), axis=0)
        pt = [p for p in sites(a.pbdb, crst, trt, (Ht, Wt)) if vt[p[0], p[1]]]
        st = np.full(Ht * Wt, np.nan, dtype="float32")
        st[vt.ravel()] = final.predict_proba(Xt.reshape(Ct, -1).T[vt.ravel()])[:, 1]
        st = st.reshape(Ht, Wt)
        report["transfer"] = {"stack": a.test_stack, "sites": len(pt), "capture": capture(np.nan_to_num(st, nan=-1), vt, pt)}
        with rasterio.open(os.path.join(a.out, "prospectivity_test.tif"), "w", driver="GTiff", height=Ht, width=Wt, count=1,
                           dtype="float32", crs=crst, transform=trt, compress="deflate", nodata=np.nan) as d:
            d.write(st, 1)
    json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "features"}, indent=1))
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(score, cmap="magma", vmin=0, vmax=np.nanpercentile(score, 99))
    ax.plot([c for _, c, _ in pts], [r for r, _, _ in pts], "c+", ms=5, label="known fossil collections (PBDB)")
    ax.set_title("Fossil prospectivity (bright = likely), with known collections"); ax.legend(loc="lower right"); ax.axis("off")
    fig.savefig(os.path.join(a.out, "prospectivity.png"), dpi=110, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    main()
