#!/usr/bin/env python3
"""
Build a self-contained SpatialConnect.zip for QGIS.

Install via:  Plugins → Manage and Install Plugins → Install from ZIP

Usage:
    python build_plugin_zip.py            # creates dist/SpatialConnect-<version>.zip
    python build_plugin_zip.py 0.2.0      # override version
"""

import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
PLUGIN_DIR = ROOT / "plugin"
METADATA = PLUGIN_DIR / "metadata.txt"
OUT_DIR = ROOT / "dist"

# ── version ──────────────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    version = sys.argv[1]
else:
    text = METADATA.read_text()
    m = re.search(r"^version\s*=\s*(.+)$", text, re.MULTILINE)
    if not m:
        sys.exit("ERROR: could not find 'version=' in metadata.txt")
    version = m.group(1).strip()

zip_name = f"SpatialConnect-{version}.zip"
print(f"Building SpatialConnect v{version} ...")

# ── assemble in a temp dir ────────────────────────────────────────────────────
with tempfile.TemporaryDirectory() as tmp:
    staging = Path(tmp) / "SpatialConnect"
    shutil.copytree(
        PLUGIN_DIR,
        staging,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )

    # ── write zip ────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_zip = OUT_DIR / zip_name

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(staging.rglob("*")):
            arcname = Path("SpatialConnect") / file.relative_to(staging)
            zf.write(file, arcname)

print(f"\nDone → {out_zip}")
print()
print("Install in QGIS:")
print("  Plugins → Manage and Install Plugins → Install from ZIP")
