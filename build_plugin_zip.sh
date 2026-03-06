#!/usr/bin/env bash
# Build a self-contained SpatialConnect.zip for QGIS
# Install via:  Plugins → Manage and Install Plugins → Install from ZIP
#
# Usage:
#   ./build_plugin_zip.sh            # creates dist/SpatialConnect-<version>.zip
#   ./build_plugin_zip.sh 0.2.0      # override version

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERSION="${1:-$(grep -m1 '^version=' "$SCRIPT_DIR/plugin/metadata.txt" | cut -d= -f2)}"
OUT_DIR="$SCRIPT_DIR/dist"
TMP_DIR="$(mktemp -d)"
ZIP_NAME="SpatialConnect-${VERSION}.zip"

echo "Building SpatialConnect v${VERSION} ..."

# core/ is already inside plugin/ — just copy the whole folder
cp -r "$SCRIPT_DIR/plugin" "$TMP_DIR/SpatialConnect"

# Clean up pycache
find "$TMP_DIR/SpatialConnect" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find "$TMP_DIR/SpatialConnect" -name "*.pyc" -delete 2>/dev/null || true

mkdir -p "$OUT_DIR"
(cd "$TMP_DIR" && zip -r "$OUT_DIR/$ZIP_NAME" SpatialConnect)
rm -rf "$TMP_DIR"

echo ""
echo "Done → $OUT_DIR/$ZIP_NAME"
echo ""
echo "Install in QGIS:"
echo "  Plugins → Manage and Install Plugins → Install from ZIP"
