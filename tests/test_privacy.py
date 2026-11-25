"""Guard: the public repo must not carry personal details, internal hosts, local paths or field-target coordinates.

The word list is base64 so the guard itself does not publish it. Add words by re-encoding the list.
Run: python -m unittest discover -s tests -v
"""
import base64
import re
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORDS = base64.b64decode(
    "dXpiZWsKdGFzaGtlbnQKdG9zaGtlbnQKYnVraGFyYQpyb2JlcnRhCmVkY29tCnZzcGhlcmUKa2VyaW8KaXNtb2lsb3ZkZXZhcmNobGludXgKMDAwOS0wMDA4LTI3ODU="
).decode().split("\n")
PATTERNS = [
    re.compile(r"\b(10\.\d{1,3}|172\.(1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b"),  # private IPv4
    re.compile(r"/[U]sers/[A-Za-z]"),                                                  # local macOS paths
    re.compile(r"/home/[a-z][a-z0-9_-]*/"),                                             # local Linux home paths
    re.compile(r"/private/tmp/claude"),                                                # local scratch paths
    re.compile(r"[A-Za-z0-9._%+-]+@(g[m]ail|yahoo|mail|outlook|icloud)\.[a-z]+", re.I),   # personal mailboxes
]
SKIP_SUFFIX = {".png", ".jpg", ".webp", ".tif", ".nc", ".lock"}
# lists of unexplored field-target patches (targets.py output) stay local: no public treasure map
TARGET_LIST = re.compile(r"(^|/)targets[^/]*\.csv$", re.I)


def tracked_files():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return out.splitlines()


def tracked_text_files():
    for name in tracked_files():
        p = ROOT / name
        if p.suffix.lower() in SKIP_SUFFIX or not p.is_file() or p == Path(__file__).resolve():
            continue
        try:
            yield name, p.read_text()
        except UnicodeDecodeError:
            continue


class PrivacyTest(unittest.TestCase):
    def test_no_private_details_in_tracked_files(self):
        hits = []
        for name, text in tracked_text_files():
            low = text.lower()
            hits += [f"{name}: word #{i}" for i, w in enumerate(WORDS) if w and w in low]
            hits += [f"{name}: {m.group(0)}" for pat in PATTERNS for m in pat.finditer(text)]
        self.assertFalse(hits, "private details in tracked files:\n" + "\n".join(hits[:50]))

    def test_no_target_coordinate_lists_tracked(self):
        hits = [n for n in tracked_files() if TARGET_LIST.search(n)]
        self.assertFalse(hits, f"field-target lists must not be published: {hits}")


if __name__ == "__main__":
    unittest.main()
