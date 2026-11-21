"""Verifies every file listed in SHA256SUMS. Run: python check_hashes.py"""
import hashlib, sys
from pathlib import Path
root = Path(__file__).resolve().parent; ok = bad = 0
for line in (root / "SHA256SUMS").read_text().splitlines():
    h, rel = line.split("  ", 1); p = root / rel
    if p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == h: ok += 1
    else: bad += 1; print("MISMATCH or missing:", rel)
print(f"{ok} files match, {bad} do not.")
sys.exit(1 if bad else 0)
