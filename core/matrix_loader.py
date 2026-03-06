"""
MatrixLoader – load connectivity / transition matrices.

Supported formats
-----------------
* MatrixMarket  (.mtx) – the format produced by the Lagrangian particle model
  (e.g. ``sparse_transition_matrix.mtx``)
* NumPy sparse  (.npz) – scipy.sparse.save_npz output
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
from scipy.sparse import csr_matrix, issparse, load_npz

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class MatrixLoader:
    """Load a connectivity / transition matrix from file.

    Parameters
    ----------
    dtype : numpy dtype
        Target dtype.  Default ``np.float64``.
    """

    SUPPORTED_EXTENSIONS = {".npz", ".mtx"}

    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, source: Union[str, Path, np.ndarray]) -> csr_matrix:
        """
        Load a transition matrix from *source*.

        Accepts
        -------
        * file path (str or Path) – ``.mtx`` or ``.npz``
        * ndarray  – wrapped in a CSR matrix
        * sparse matrix – returned as-is (cast to dtype)
        """
        if isinstance(source, (str, Path)):
            return self._load_file(Path(source))
        if issparse(source):
            return source.astype(self.dtype).tocsr()
        return csr_matrix(np.asarray(source, dtype=self.dtype))

    def load_scenarios(
        self,
        sources: dict[str, Union[str, Path, np.ndarray]],
    ) -> dict[str, csr_matrix]:
        """Load multiple named scenarios at once."""
        return {name: self.load(src) for name, src in sources.items()}

    # ------------------------------------------------------------------
    # Private dispatch
    # ------------------------------------------------------------------

    def _load_file(self, path: Path) -> csr_matrix:
        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension '{ext}'.  "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )
        logger.info("Loading transition matrix from %s", path)
        if ext == ".npz":
            return self._from_npz(path)
        if ext == ".mtx":
            return self._from_mtx(path)

    def _from_npz(self, path: Path) -> csr_matrix:
        """Load a scipy sparse matrix saved with ``scipy.sparse.save_npz``."""
        mat = load_npz(path)
        return mat.astype(self.dtype).tocsr()

    def _from_mtx(self, path: Path) -> csr_matrix:
        """
        MatrixMarket format (.mtx).

        Example header from the Lagrangian particle model::

            %%MatrixMarket matrix coordinate real general
            2127 2127 8526
            10 120 2E-1
            ...
        """
        from scipy.io import mmread
        mat = mmread(str(path))
        if issparse(mat):
            return mat.astype(self.dtype).tocsr()
        # dense array fallback
        return csr_matrix(np.asarray(mat, dtype=self.dtype))
