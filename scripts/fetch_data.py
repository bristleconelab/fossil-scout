#!/usr/bin/env python3
"""Download the two small public inputs: PBDB Hell Creek collections and Macrostrat Hell Creek outcrop polygons.

PBDB:       paleobiodb.org data service, all collections with formation = Hell Creek (with location fields).
Macrostrat: geologic_units/map for the Hell Creek strat_name_ids 71267, 71268 and 867 (GeoJSON, about 40 MB).

The PBDB snapshot used for the results is committed as data/pbdb_hellcreek.csv; re-downloading it may add new
collections. The Macrostrat polygons are not committed; fetch them before running targets.py.

Usage: fetch_data.py [--pbdb data/pbdb_hellcreek.csv] [--macrostrat data/hellcreek_polygons.geojson] [--skip-pbdb]
"""
import argparse
import json
import os

import requests

PBDB = "https://paleobiodb.org/data1.2/colls/list.csv"
MACROSTRAT = "https://macrostrat.org/api/v2/geologic_units/map"
HELL_CREEK_IDS = "71267,71268,867"      # from /api/v2/defs/strat_names?strat_name_like=Hell%20Creek


def get(url, params, path):
    r = requests.get(url, params=params, timeout=300)
    r.raise_for_status()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(r.content)
    print(f"-> {path} ({len(r.content) / 1e6:.1f} MB)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pbdb", default="data/pbdb_hellcreek.csv")
    ap.add_argument("--macrostrat", default="data/hellcreek_polygons.geojson")
    ap.add_argument("--skip-pbdb", action="store_true", help="keep the committed PBDB snapshot")
    a = ap.parse_args()
    if not a.skip_pbdb:
        get(PBDB, {"formation": "Hell Creek", "show": "loc,locext,geo", "limit": "all"}, a.pbdb)
    get(MACROSTRAT, {"strat_name_id": HELL_CREEK_IDS, "format": "geojson_bare"}, a.macrostrat)
    n = len(json.load(open(a.macrostrat))["features"])
    print(f"   {n} Hell Creek polygons", flush=True)


if __name__ == "__main__":
    main()
