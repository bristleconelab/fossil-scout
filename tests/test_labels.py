"""Label filter: a PBDB coordinate taken from the county is not a fossil site, however many decimals it has.

Run: python -m unittest discover -s tests -v
"""
import csv
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from train import precise  # noqa: E402


class LabelFilterTest(unittest.TestCase):
    def test_precision_and_basis(self):
        self.assertTrue(precise({"latlng_precision": "6", "latlng_basis": "stated in text"}))
        self.assertTrue(precise({"latlng_precision": "seconds", "latlng_basis": "estimated from map"}))
        self.assertFalse(precise({"latlng_precision": "1", "latlng_basis": "stated in text"}))
        self.assertFalse(precise({"latlng_precision": "6", "latlng_basis": "based on political unit"}))

    def test_snapshot_has_no_county_coordinates_among_sites(self):
        rows = list(csv.DictReader(open(ROOT / "data" / "pbdb_hellcreek.csv", encoding="utf-8")))
        kept = [r for r in rows if precise(r)]
        self.assertTrue(kept)
        self.assertFalse([r["collection_no"] for r in kept if r["latlng_basis"] == "based on political unit"])


if __name__ == "__main__":
    unittest.main()
