"""Tests for the Raster I/O utilities."""

import numpy as np
import pytest
from pathlib import Path

from core.raster_utils import RasterUtils, RasterMeta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_test_meta(rows=10, cols=10, bands=1, nodata=-9999.0):
    """Build a minimal RasterMeta for testing.

    Uses rasterio types when available, falls back to GDAL-compatible
    values (WKT string + 6-tuple geotransform) so tests also run in
    environments that have osgeo.gdal but not rasterio.
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds
        transform = from_bounds(0, 0, 1, 1, cols, rows)
        crs = rasterio.crs.CRS.from_epsg(4326)
    except ImportError:
        # GDAL 6-tuple: (x_min, x_pixel, x_rot, y_max, y_rot, y_pixel_neg)
        transform = (0.0, 1.0 / cols, 0.0, 1.0, 0.0, -1.0 / rows)
        crs = (
            'GEOGCS["WGS 84",DATUM["WGS_1984",'
            'SPHEROID["WGS 84",6378137,298.257223563]],'
            'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]'
        )
    return RasterMeta(
        crs=crs,
        transform=transform,
        width=cols,
        height=rows,
        count=bands,
        dtype="float64",
        nodata=nodata,
    )


# ---------------------------------------------------------------------------
# Write / read roundtrip
# Tests run with rasterio when available, fall back to osgeo.gdal.
# Both are valid backends; at least one must be present.
# ---------------------------------------------------------------------------

class TestRasterIO:

    def test_write_read_roundtrip_2d(self, tmp_path):
        array = np.random.rand(10, 10).astype(np.float64)
        meta = make_test_meta(10, 10)
        out_path = tmp_path / "test.tif"
        RasterUtils.write_raster(str(out_path), array, meta)
        array_back, meta_back = RasterUtils.read_raster(str(out_path))
        np.testing.assert_allclose(array_back, array, rtol=1e-6)

    def test_write_read_roundtrip_3d(self, tmp_path):
        array = np.random.rand(10, 10, 3).astype(np.float64)
        meta = make_test_meta(10, 10, bands=3)
        out_path = tmp_path / "test_3band.tif"
        RasterUtils.write_raster(str(out_path), array, meta)
        array_back, meta_back = RasterUtils.read_raster(str(out_path))
        assert array_back.shape == (10, 10, 3)
        np.testing.assert_allclose(array_back, array, rtol=1e-6)

    def test_nodata_preserved(self, tmp_path):
        array = np.ones((5, 5), dtype=np.float64)
        array[2, 2] = -9999.0
        meta = make_test_meta(5, 5, nodata=-9999.0)
        out_path = tmp_path / "nd.tif"
        RasterUtils.write_raster(str(out_path), array, meta)
        arr_back, meta_back = RasterUtils.read_raster(str(out_path))
        assert arr_back[2, 2] == pytest.approx(-9999.0)
        assert meta_back.nodata == pytest.approx(-9999.0)


# ---------------------------------------------------------------------------
# RasterMeta.to_profile
# ---------------------------------------------------------------------------

class TestRasterMeta:

    def test_to_profile_keys(self):
        meta = make_test_meta()
        profile = meta.to_profile()
        for key in ("driver", "height", "width", "count", "dtype", "crs", "transform"):
            assert key in profile

    def test_nodata_included_in_profile(self):
        meta = make_test_meta(nodata=-1.0)
        profile = meta.to_profile()
        assert profile["nodata"] == pytest.approx(-1.0)
