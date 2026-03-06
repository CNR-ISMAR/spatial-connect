"""Tests for MatrixLoader (.mtx / .npz only)."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, issparse, save_npz

from core.matrix_loader import MatrixLoader


@pytest.fixture
def loader():
    return MatrixLoader()


@pytest.fixture
def small_sparse(tmp_path):
    """8x8 random non-negative sparse matrix saved as .npz."""
    np.random.seed(42)
    dense = np.abs(np.random.rand(8, 8))
    mat = csr_matrix(dense)
    path = tmp_path / "mat.npz"
    save_npz(str(path), mat)
    return path, dense


# ---------------------------------------------------------------------------
# NPZ loading
# ---------------------------------------------------------------------------

class TestNPZLoading:

    def test_load_npz_sparse(self, loader, small_sparse, tmp_path):
        path, dense = small_sparse
        mat = loader.load(path)
        assert issparse(mat)
        np.testing.assert_allclose(mat.toarray(), dense, rtol=1e-10)

    def test_npz_dtype(self, loader, small_sparse):
        path, _ = small_sparse
        mat = loader.load(path)
        assert mat.dtype == np.float64


# ---------------------------------------------------------------------------
# MTX loading
# ---------------------------------------------------------------------------

class TestMTXLoading:

    def test_load_mtx_sparse(self, loader, tmp_path):
        from scipy.io import mmwrite
        np.random.seed(7)
        dense = np.abs(np.random.rand(5, 5))
        path = tmp_path / "mat.mtx"
        mmwrite(str(path), csr_matrix(dense))
        mat = loader.load(path)
        assert issparse(mat)
        np.testing.assert_allclose(mat.toarray(), dense, rtol=1e-10)

    def test_mtx_matches_npz(self, loader, tmp_path):
        """Both formats must give the same matrix."""
        from scipy.io import mmwrite
        np.random.seed(3)
        dense = np.abs(np.random.rand(6, 6))
        sparse = csr_matrix(dense)
        p_mtx = tmp_path / "C.mtx"
        p_npz = tmp_path / "C.npz"
        mmwrite(str(p_mtx), sparse)
        save_npz(str(p_npz), sparse)
        np.testing.assert_allclose(
            loader.load(p_mtx).toarray(),
            loader.load(p_npz).toarray(),
            rtol=1e-10,
        )

    def test_mtx_coordinate_real_general(self, loader, tmp_path):
        """Hand-written MTX file matching the format of sparse_transition_matrix.mtx."""
        content = (
            "%%MatrixMarket matrix coordinate real general\n"
            "4 4 3\n"
            "1 2 0.4\n"
            "1 3 0.6\n"
            "3 4 1.0\n"
        )
        path = tmp_path / "T.mtx"
        path.write_text(content)
        mat = loader.load(path)
        assert issparse(mat)
        assert mat.shape == (4, 4)
        dense = mat.toarray()
        assert dense[0, 1] == pytest.approx(0.4)
        assert dense[0, 2] == pytest.approx(0.6)
        assert dense[2, 3] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Unsupported extension
# ---------------------------------------------------------------------------

class TestUnsupportedExtension:

    def test_csv_now_unsupported(self, loader, tmp_path):
        path = tmp_path / "matrix.csv"
        path.write_text("1,2\n3,4")
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load(path)

    def test_txt_unsupported(self, loader, tmp_path):
        path = tmp_path / "matrix.txt"
        path.write_text("1 2\n3 4")
        with pytest.raises(ValueError, match="Unsupported"):
            loader.load(path)


# ---------------------------------------------------------------------------
# load_scenarios
# ---------------------------------------------------------------------------

class TestLoadScenarios:

    def test_load_scenarios_mtx(self, loader, tmp_path):
        from scipy.io import mmwrite
        mat = csr_matrix(np.eye(4, dtype=float))
        p1 = tmp_path / "s1.mtx"
        p2 = tmp_path / "s2.mtx"
        mmwrite(str(p1), mat)
        mmwrite(str(p2), mat * 2)
        scenarios = loader.load_scenarios({"s1": p1, "s2": p2})
        assert set(scenarios.keys()) == {"s1", "s2"}
        np.testing.assert_allclose(
            scenarios["s2"].toarray(), (mat * 2).toarray(), rtol=1e-10
        )
