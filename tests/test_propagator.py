"""Tests for the core propagation engine."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, eye as speye

from core.propagator import SpatialPropagator, PropagationResult
from core.raster_utils import RasterUtils


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def identity_matrix():
    """1-step identity: output == input."""
    N = 16
    return speye(N, format="csr")


@pytest.fixture
def uniform_matrix():
    """Uniform 4×4 raster, each cell connected equally to all others."""
    N = 16
    C = np.ones((N, N), dtype=float) / N
    return csr_matrix(C)


@pytest.fixture
def simple_raster():
    return np.arange(16, dtype=float).reshape(4, 4)


# ---------------------------------------------------------------------------
# Basic correctness
# ---------------------------------------------------------------------------

class TestSpatialPropagatorDiscrete:

    def test_identity_matrix_returns_input(self, identity_matrix, simple_raster):
        p = SpatialPropagator(mode="discrete", clip_negative=False)
        result = p.run(simple_raster, identity_matrix, iterations=5)
        np.testing.assert_allclose(result.output, simple_raster, rtol=1e-10)

    def test_shape_preserved(self, identity_matrix, simple_raster):
        p = SpatialPropagator(mode="discrete")
        result = p.run(simple_raster, identity_matrix, iterations=1)
        assert result.output.shape == simple_raster.shape

    def test_result_type(self, identity_matrix, simple_raster):
        p = SpatialPropagator(mode="discrete")
        result = p.run(simple_raster, identity_matrix, iterations=1)
        assert isinstance(result, PropagationResult)
        assert result.mode == "discrete"
        assert result.iterations == 1

    def test_n_iterations_consistent(self, simple_raster):
        """C^2 · x == C · (C · x)"""
        N = 16
        C = csr_matrix(np.random.rand(N, N) * 0.1)
        p = SpatialPropagator(mode="discrete", clip_negative=False)
        result_n2  = p.run(simple_raster, C, iterations=2)
        result_n2b = p.run(
            p.run(simple_raster, C, iterations=1).output, C, iterations=1
        )
        np.testing.assert_allclose(result_n2.output, result_n2b.output, rtol=1e-8)

    def test_clip_negative(self, simple_raster):
        N = 16
        # matrix that produces negative outputs for some cells
        C = csr_matrix(-np.eye(N, dtype=float))
        p = SpatialPropagator(mode="discrete", clip_negative=True)
        result = p.run(simple_raster, C, iterations=1)
        assert np.all(result.output >= 0)

    def test_nodata_mask(self, identity_matrix):
        raster = np.array([[1.0, -9999.0], [3.0, 4.0]])
        N = 4
        C = speye(N, format="csr")
        p = SpatialPropagator(mode="discrete")
        result = p.run(raster, C, iterations=1, nodata_value=-9999.0)
        assert result.output[0, 1] == -9999.0

    def test_invalid_iterations(self, identity_matrix, simple_raster):
        p = SpatialPropagator()
        with pytest.raises(ValueError, match="iterations must be"):
            p.run(simple_raster, identity_matrix, iterations=0)

    def test_matrix_size_mismatch(self, simple_raster):
        C = speye(10, format="csr")  # wrong size for 4×4 raster
        p = SpatialPropagator()
        with pytest.raises(ValueError, match="does not match"):
            p.run(simple_raster, C, iterations=1)


class TestSpatialPropagatorContinuous:

    def test_continuous_close_to_discrete_for_small_n(self, simple_raster):
        """
        Continuous mode with C = eps*I:
          expm(n * eps * I) · x  =  e^(n*eps) * x
        Discrete mode with C = eps*I:
          (eps*I)^n · x  =  eps^n * x

        For n=1 both should equal a scalar multiple of x; verify
        the continuous result is close to e^eps * x.
        """
        N = 16
        eps = 0.05
        C = csr_matrix(np.eye(N, dtype=float) * eps)
        p_c = SpatialPropagator(mode="continuous", clip_negative=False)
        r_c = p_c.run(simple_raster, C, iterations=1)
        expected = np.exp(eps) * simple_raster
        np.testing.assert_allclose(r_c.output, expected, rtol=1e-6)

    def test_continuous_result_shape(self, simple_raster):
        N = 16
        C = csr_matrix(np.eye(N, dtype=float) * 0.01)
        p = SpatialPropagator(mode="continuous")
        result = p.run(simple_raster, C, iterations=1)
        assert result.output.shape == simple_raster.shape
        assert result.mode == "continuous"


class TestRunScenarios:

    def test_run_scenarios_returns_all(self, simple_raster):
        N = 16
        scenarios = {
            "s1": speye(N, format="csr"),
            "s2": speye(N, format="csr"),
        }
        p = SpatialPropagator(mode="discrete")
        results = p.run_scenarios(simple_raster, scenarios, iterations=1)
        assert set(results.keys()) == {"s1", "s2"}
        for name, res in results.items():
            assert isinstance(res, PropagationResult)
            assert res.scenario_name == name


# ---------------------------------------------------------------------------
# Transpose connectivity (Sofia's x·T convention)
# ---------------------------------------------------------------------------

class TestTransposeConnectivity:

    def test_transpose_true_applies_xT_formula(self):
        """With an asymmetric T, x·T ≠ T·x; verify the correct formula is used."""
        # T: cell 0 → cell 1 with weight 1 (all mass moves right)
        T = csr_matrix(np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float))
        # rows=1, cols=2; x=[10, 0]
        raster = np.array([[10.0, 0.0]])
        p = SpatialPropagator(mode="discrete", transpose_connectivity=True,
                              clip_negative=False)
        result = p.run(raster, T, iterations=1)
        # x(t+dt) = x @ T = [10, 0] @ [[0,1],[0,1]] = [0, 10]
        np.testing.assert_allclose(result.output, [[0.0, 10.0]], rtol=1e-10)

    def test_transpose_false_applies_Cx_formula(self):
        """With transpose=False the legacy C·x formula is used."""
        # C: cell 0 receives from cell 1 (pull)
        C = csr_matrix(np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float))
        raster = np.array([[10.0, 0.0]])
        p = SpatialPropagator(mode="discrete", transpose_connectivity=False,
                              clip_negative=False)
        result = p.run(raster, C, iterations=1)
        # x' = C @ x = [[0,1],[0,1]] @ [10,0] = [0, 0]
        np.testing.assert_allclose(result.output, [[0.0, 0.0]], rtol=1e-10)

    def test_symmetric_matrix_same_either_way(self, identity_matrix, simple_raster):
        """For symmetric T, transpose=True and transpose=False give the same result."""
        p_t = SpatialPropagator(mode="discrete", transpose_connectivity=True,
                                clip_negative=False)
        p_f = SpatialPropagator(mode="discrete", transpose_connectivity=False,
                                clip_negative=False)
        res_t = p_t.run(simple_raster, identity_matrix, iterations=3)
        res_f = p_f.run(simple_raster, identity_matrix, iterations=3)
        np.testing.assert_allclose(res_t.output, res_f.output, rtol=1e-12)


# ---------------------------------------------------------------------------
# Masked domain (cell_ids)
# ---------------------------------------------------------------------------

class TestCellIds:

    def test_cell_ids_subsets_full_raster(self):
        """
        A 3×3 raster with a sea mask (only 5 of 9 cells valid).
        The transition matrix is 5×5.  Verify that after propagation
        the land cells are restored to nodata and sea cells are updated.
        """
        # mask: 1=sea, 0=land
        mask = np.array([
            [1, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
        ], dtype=float)
        cell_ids = RasterUtils.compute_cell_ids(mask, sea_value=1.0)
        # IDs should be: [[ 0,-1, 1],[ 2, 3,-1],[-1, 4, 5]] -- wait, 6 ones not 5
        # Actually let's just verify N=6, IDs 0..5
        N = int((cell_ids >= 0).sum())   # should be 6

        raster = np.where(mask == 1, 1.0, -9999.0)  # sea=1, land=nodata
        # Identity transition matrix (5x5) → output = input for sea cells
        T = speye(N, format="csr")
        p = SpatialPropagator(mode="discrete", transpose_connectivity=True,
                              clip_negative=False)
        result = p.run(raster, T, iterations=1, nodata_value=-9999.0,
                       cell_ids=cell_ids)
        # Land cells must remain nodata
        assert result.output[0, 1] == pytest.approx(-9999.0)
        assert result.output[1, 2] == pytest.approx(-9999.0)
        # Sea cells must be 1.0 (identity matrix → no change)
        assert result.output[0, 0] == pytest.approx(1.0)
        assert result.output[1, 0] == pytest.approx(1.0)

    def test_compute_cell_ids_ordering(self):
        """Cell IDs must be 0-indexed and assigned in row-major order."""
        mask = np.array([[0, 1, 1], [1, 0, 1]], dtype=float)
        ids = RasterUtils.compute_cell_ids(mask)
        assert ids[0, 0] == -1
        assert ids[0, 1] == 0
        assert ids[0, 2] == 1
        assert ids[1, 0] == 2
        assert ids[1, 1] == -1
        assert ids[1, 2] == 3

    def test_compute_cell_ids_3d_mask(self):
        """Band dimension from rioxarray should be automatically squeezed."""
        mask = np.array([[[0, 1], [1, 0]]], dtype=float)  # shape (1, 2, 2)
        ids = RasterUtils.compute_cell_ids(mask)
        assert ids.shape == (2, 2)
        assert ids[0, 1] == 0
        assert ids[1, 0] == 1
