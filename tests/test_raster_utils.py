"""Tests for the Raster I/O utilities."""

import numpy as np
import pytest
from pathlib import Path

from core.raster_utils import RasterUtils, RasterMeta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_test_meta(rows=10, cols=10, bands=1, nodata=-9999.0):
    try:
        import rasterio
        from rasterio.transform import from_bounds
        transform = from_bounds(0, 0, 1, 1, cols, rows)
        crs = rasterio.crs.CRS.from_epsg(4326)
    except ImportError:
        transform = None
        crs = None
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
# ---------------------------------------------------------------------------

class TestRasterIO:

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("rasterio"),
        reason="rasterio not installed",
    )
    def test_write_read_roundtrip_2d(self, tmp_path):
        array = np.random.rand(10, 10).astype(np.float64)
        meta = make_test_meta(10, 10)
        out_path = tmp_path / "test.tif"
        RasterUtils.write_raster(str(out_path), array, meta)
        array_back, meta_back = RasterUtils.read_raster(str(out_path))
        np.testing.assert_allclose(array_back, array, rtol=1e-6)

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("rasterio"),
        reason="rasterio not installed",
    )
    def test_write_read_roundtrip_3d(self, tmp_path):
        array = np.random.rand(10, 10, 3).astype(np.float64)
        meta = make_test_meta(10, 10, bands=3)
        out_path = tmp_path / "test_3band.tif"
        RasterUtils.write_raster(str(out_path), array, meta)
        array_back, meta_back = RasterUtils.read_raster(str(out_path))
        assert array_back.shape == (10, 10, 3)
        np.testing.assert_allclose(array_back, array, rtol=1e-6)

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("rasterio"),
        reason="rasterio not installed",
    )
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
