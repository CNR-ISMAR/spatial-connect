"""
pytest conftest – synthetic data fixtures.

Minimal synthetic dataset:
- raster_single    – 20×20 float64 GeoTIFF, single band, Gaussian blob
- raster_nodata    – 20×20, nodata=-9999 in the border ring
- matrix_npz       – sparse NxN identity matrix saved as .npz
- matrix_mtx       – sparse NxN identity matrix saved as .mtx
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import eye as speye, save_npz


# ── helpers ─────────────────────────────────────────────────────────────────

def _make_rasterio_meta(rows, cols, nodata=None):
    try:
        import rasterio
        from rasterio.transform import from_bounds
        transform = from_bounds(12.0, 44.0, 12.2, 44.2, cols, rows)
        crs = rasterio.crs.CRS.from_epsg(4326)
    except ImportError:
        transform = None
        crs = None

    from core.raster_utils import RasterMeta
    return RasterMeta(
        crs=crs, transform=transform,
        width=cols, height=rows, count=1,
        dtype="float64", nodata=nodata,
    )


def _write_raster(path: Path, array: np.ndarray, nodata=None):
    from core.raster_utils import RasterUtils
    rows, cols = array.shape
    meta = _make_rasterio_meta(rows, cols, nodata)
    RasterUtils.write_raster(str(path), array, meta, nodata_value=nodata)
    return path


def _gaussian_blob(rows, cols, cx=None, cy=None, sigma=3.0, amplitude=100.0):
    if cx is None:
        cx = cols // 2
    if cy is None:
        cy = rows // 2
    c_grid, r_grid = np.meshgrid(np.arange(cols), np.arange(rows))
    return amplitude * np.exp(
        -((r_grid - cy) ** 2 + (c_grid - cx) ** 2) / (2 * sigma ** 2)
    )


def _simple_sparse(n):
    """Identity-like NxN sparse CSR matrix (useful when we just need a valid matrix file)."""
    return speye(n, format="csr", dtype=np.float64)


# ── dimension fixtures ────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def rows():
    return 20


@pytest.fixture(scope="session")
def cols():
    return 20


@pytest.fixture(scope="session")
def N(rows, cols):
    return rows * cols


# ── raster array fixtures ────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def raster_array(rows, cols):
    """Single-band Gaussian blob ndarray, shape (20, 20)."""
    return _gaussian_blob(rows, cols, sigma=3.0, amplitude=100.0)


@pytest.fixture(scope="session")
def raster_nodata_array(rows, cols):
    """Single-band blob with border ring set to nodata=-9999."""
    arr = _gaussian_blob(rows, cols, sigma=3.0, amplitude=100.0)
    arr[0, :] = arr[-1, :] = arr[:, 0] = arr[:, -1] = -9999.0
    return arr


# ── raster file fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def raster_tif(tmp_path, raster_array):
    return _write_raster(tmp_path / "raster.tif", raster_array)


@pytest.fixture
def raster_nodata_tif(tmp_path, raster_nodata_array):
    return _write_raster(tmp_path / "raster_nd.tif", raster_nodata_array, nodata=-9999.0)


# ── matrix file fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def matrix_npz(tmp_path, N):
    """NxN sparse identity matrix saved as .npz."""
    path = tmp_path / "matrix.npz"
    save_npz(str(path), _simple_sparse(N))
    return path


@pytest.fixture
def matrix_mtx(tmp_path, N):
    """NxN sparse identity matrix saved as MatrixMarket .mtx."""
    from scipy.io import mmwrite
    path = tmp_path / "matrix.mtx"
    mmwrite(str(path), _simple_sparse(N))
    return path
