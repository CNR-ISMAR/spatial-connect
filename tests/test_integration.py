"""
Integration tests - full pipeline from data loading to raster output.

These tests exercise the complete pipeline:
    load raster  ->  load connectivity matrix (.npz / .mtx)
                 ->  propagate  ->  write GeoTIFF  ->  read back

They use the synthetic fixtures defined in conftest.py and run entirely
in tmp_path, requiring no pre-existing data files.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import eye as speye

from core import SpatialPropagator, MatrixLoader, RasterUtils


# ------------------------------------------------------------
# 1.  Raster I/O round-trip
# ------------------------------------------------------------

class TestRasterRoundTrip:

    def test_single_band_preserves_values(self, raster_tif, tmp_path):
        array, meta = RasterUtils.read_raster(raster_tif)
        out = tmp_path / "out.tif"
        RasterUtils.write_raster(str(out), array, meta)
        array2, _ = RasterUtils.read_raster(str(out))
        np.testing.assert_allclose(array2, array, rtol=1e-6)

    def test_nodata_round_trip(self, raster_nodata_tif, tmp_path):
        array, meta = RasterUtils.read_raster(raster_nodata_tif)
        assert meta.nodata == pytest.approx(-9999.0)
        out = tmp_path / "nd_out.tif"
        RasterUtils.write_raster(str(out), array, meta)
        a2, m2 = RasterUtils.read_raster(str(out))
        np.testing.assert_array_equal(a2[0, :], -9999.0)
        np.testing.assert_array_equal(a2[-1, :], -9999.0)


# ------------------------------------------------------------
# 2.  Propagation physics
# ------------------------------------------------------------

class TestPropagationPhysics:

    def test_identity_n_iterations_all_same(self, raster_array, rows, cols):
        """Applying the identity matrix any number of times -> identical output."""
        I = speye(rows * cols, format="csr")
        p = SpatialPropagator(mode="discrete", clip_negative=False)
        for n in (1, 5, 10):
            r = p.run(raster_array, I, iterations=n)
            np.testing.assert_allclose(r.output, raster_array, rtol=1e-10)

    def test_nodata_border_preserved(self, raster_nodata_array, rows, cols):
        """NoData cells must remain -9999 after propagation."""
        I = speye(rows * cols, format="csr")
        p = SpatialPropagator(mode="discrete")
        result = p.run(raster_nodata_array, I, iterations=2, nodata_value=-9999.0)
        np.testing.assert_array_equal(result.output[0, :],  -9999.0)
        np.testing.assert_array_equal(result.output[-1, :], -9999.0)
        np.testing.assert_array_equal(result.output[:, 0],  -9999.0)
        np.testing.assert_array_equal(result.output[:, -1], -9999.0)

    def test_continuous_mode_shape(self, raster_array, rows, cols):
        """Continuous mode must return the same spatial shape."""
        I = speye(rows * cols, format="csr")
        p = SpatialPropagator(mode="continuous")
        result = p.run(raster_array, I, iterations=1)
        assert result.output.shape == raster_array.shape

    def test_run_scenarios_both_formats(self, raster_array, matrix_npz,
                                        matrix_mtx):
        """Loading the same matrix from .npz and .mtx and running scenarios gives
        identical outputs (identity == identity regardless of format)."""
        loader = MatrixLoader()
        scenarios = {
            "from_npz": loader.load(matrix_npz),
            "from_mtx": loader.load(matrix_mtx),
        }
        p = SpatialPropagator(mode="discrete", clip_negative=False)
        results = p.run_scenarios(raster_array, scenarios, iterations=3)
        np.testing.assert_allclose(
            results["from_npz"].output,
            results["from_mtx"].output,
            rtol=1e-10,
        )


# ------------------------------------------------------------
# 3.  Full pipeline: raster in -> propagate -> GeoTIFF out -> read back
# ------------------------------------------------------------

class TestFullPipeline:

    def test_write_and_reload_single_scenario(self, raster_tif, matrix_npz,
                                              tmp_path):
        array, meta = RasterUtils.read_raster(raster_tif)

        mat = MatrixLoader().load(matrix_npz)
        p = SpatialPropagator(mode="discrete", clip_negative=False)
        result = p.run(array, mat, iterations=5)

        out_path = tmp_path / "output_n5.tif"
        RasterUtils.write_raster(str(out_path), result.output, meta)

        reloaded, meta2 = RasterUtils.read_raster(str(out_path))
        assert reloaded.shape == array.shape
        np.testing.assert_allclose(reloaded, result.output, rtol=1e-5)
        assert meta2.crs == meta.crs

    def test_pipeline_nodata_end_to_end(self, raster_nodata_tif, matrix_mtx,
                                        tmp_path):
        array, meta = RasterUtils.read_raster(raster_nodata_tif)
        nodata_val = meta.nodata   # -9999.0

        mat = MatrixLoader().load(matrix_mtx)
        p = SpatialPropagator(mode="discrete")
        result = p.run(array, mat, iterations=3, nodata_value=nodata_val)

        out = tmp_path / "output_nd.tif"
        RasterUtils.write_raster(str(out), result.output, meta)

        reloaded, _ = RasterUtils.read_raster(str(out))
        np.testing.assert_array_equal(reloaded[0, :],  nodata_val)
        np.testing.assert_array_equal(reloaded[-1, :], nodata_val)

    def test_pipeline_continuous_mode(self, raster_tif, matrix_npz, tmp_path):
        array, meta = RasterUtils.read_raster(raster_tif)

        mat = MatrixLoader().load(matrix_npz)
        p = SpatialPropagator(mode="continuous")
        result = p.run(array, mat, iterations=2)

        out = tmp_path / "output_cont.tif"
        RasterUtils.write_raster(str(out), result.output, meta)
        assert out.exists()
        reloaded, _ = RasterUtils.read_raster(str(out))
        assert reloaded.shape == array.shape
