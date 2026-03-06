"""
Core propagation engine.

Two propagation conventions are supported:

  Transition convention (transition=True)  - default, matches Sofia's model:
    x(t+dt)  = x(t) * T^n       (discrete)
    x(t+dt)  = x(t) * expm(n*T) (continuous)
  where T[i,j] = probability of moving FROM cell i TO cell j.
  x is a row vector; repeated left-multiplication advances the distribution.

  Diffusion convention (transpose_connectivity=False)  - legacy:
    x'  = C^n * x              (discrete)
    x'  = expm(n*C) * x        (continuous)
  where C[i,j] = influence of source cell j on destination cell i.
  Matches the convolution kernel output of MatrixLoader.make_kernel().

Both conventions are implemented internally as column-vector multiplication;
when transpose_connectivity=True the matrix is transposed once in
_prepare_matrix so that the rest of the code is formula-agnostic.

Masked-domain support
---------------------
When operating on a raster with a land/sea mask (like the North Adriatic grid),
cell IDs do NOT correspond to simple row-major pixel positions.  Pass the output
of RasterUtils.compute_cell_ids() as `cell_ids` to run() and the propagator
will correctly map valid pixels <-> matrix rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from scipy.sparse import issparse, spmatrix
from scipy.sparse.linalg import expm_multiply
from scipy.sparse import eye as speye

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PropagationResult:
    """Output of a single propagation run."""

    scenario_name: str
    output: np.ndarray                   # shape (rows, cols) or (rows, cols, bands)
    iterations: int
    mode: str                            # "discrete" | "continuous"
    nodata_mask: Optional[np.ndarray] = field(default=None)   # bool mask

    # --- convenience --------------------------------------------------------

    def to_2d(self) -> np.ndarray:
        """Return output as 2-D array (rows x cols).  Raises if multi-band."""
        if self.output.ndim != 2:
            raise ValueError(
                "to_2d() is only valid for single-band results; "
                f"output shape is {self.output.shape}"
            )
        return self.output

    def to_3d(self) -> np.ndarray:
        """Return output as 3-D array (rows x cols x bands)."""
        if self.output.ndim == 2:
            return self.output[:, :, np.newaxis]
        return self.output


# ---------------------------------------------------------------------------
# Propagator
# ---------------------------------------------------------------------------

class SpatialPropagator:
    """
    Apply a connectivity / transition matrix to a raster for *n* steps.

    Parameters
    ----------
    mode : str
        ``"discrete"``   - repeated matrix multiplication (default)
        ``"continuous"`` - matrix exponential  expm(n*T)
    transpose_connectivity : bool
        When ``True`` (default), the matrix is transposed internally so that
        the formula ``x(t+dt) = x(t) * T`` (Sofia's convention) is applied.
        Set to ``False`` for legacy diffusion kernels where the formula is
        ``x' = C * x``.
    clip_negative : bool
        Clip negative values in the output to 0.  Default ``True``.
    dtype : numpy dtype
        Working dtype.  Default ``np.float64``.
    """

    def __init__(
        self,
        mode: str = "discrete",
        transpose_connectivity: bool = True,
        clip_negative: bool = True,
        normalise: bool = False,
        dtype=np.float64,
    ):
        if mode not in ("discrete", "continuous"):
            raise ValueError(f"mode must be 'discrete' or 'continuous', got '{mode}'")
        self.mode = mode
        self.transpose_connectivity = transpose_connectivity
        self.clip_negative = clip_negative
        self.normalise = normalise
        self.dtype = dtype

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        raster: np.ndarray,
        connectivity: Union[np.ndarray, spmatrix],
        iterations: int,
        nodata_value: Optional[float] = None,
        scenario_name: str = "default",
        cell_ids: Optional[np.ndarray] = None,
    ) -> PropagationResult:
        """
        Propagate *raster* through *connectivity* for *iterations* steps.

        Parameters
        ----------
        raster : ndarray, shape (rows, cols) or (rows, cols, bands)
            Input spatial data.
        connectivity : ndarray or sparse matrix, shape (N, N)
            Connectivity / transition matrix.  N = rows x cols when
            *cell_ids* is None; otherwise N = number of valid (non-masked)
            cells as defined by *cell_ids*.
        iterations : int
            Number of time steps (each step advances the distribution by dt,
            where dt is the time interval embedded in the matrix).
        nodata_value : float, optional
            Cells equal to this value are masked before propagation and
            restored as nodata in the output.
        scenario_name : str
            Label attached to the result.
        cell_ids : ndarray of int, shape (rows, cols), optional
            Mapping from 2-D pixel position to matrix row/column index.
            Valid cells have IDs 0..N-1; invalid cells have -1.
            Use ``RasterUtils.compute_cell_ids()`` to build this from a
            land/sea mask GeoTIFF (e.g. the North Adriatic grid).
            When ``None`` (default) the whole raster (rows x cols) is used
            and N = rows x cols.

        Returns
        -------
        PropagationResult
        """
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")

        rows, cols, bands = self._parse_shape(raster)

        # N depends on whether we use a masked sub-domain
        if cell_ids is not None:
            N = int((cell_ids >= 0).sum())
        else:
            N = rows * cols

        # build nodata mask (land cells)
        nodata_mask = self._build_nodata_mask(raster, nodata_value)

        # prepare connectivity matrix (validate, cast, optionally transpose + normalise)
        C = self._prepare_matrix(connectivity, N)

        # flatten raster to (N, bands) - uses cell_ids when provided
        x = self._flatten(raster, rows, cols, bands, nodata_mask, cell_ids)

        # propagate
        logger.info(
            "Propagating scenario='%s', mode=%s, iterations=%d, N=%d, bands=%d",
            scenario_name, self.mode, iterations, N, bands,
        )
        x_out = self._propagate(C, x, iterations)

        # reshape back to (rows, cols, bands) - uses cell_ids when provided
        output = self._reshape(x_out, rows, cols, bands, nodata_mask, nodata_value, cell_ids)

        # output is always (rows, cols, bands) at this point -> clip negatives
        if self.clip_negative:
            mask_expanded = nodata_mask[..., np.newaxis]  # (rows, cols, 1)
            output = np.where(mask_expanded, output, np.maximum(output, 0.0))

        # squeeze single-band back to 2-D
        if bands == 1:
            output = output[:, :, 0]

        return PropagationResult(
            scenario_name=scenario_name,
            output=output,
            iterations=iterations,
            mode=self.mode,
            nodata_mask=nodata_mask,
        )

    def run_scenarios(
        self,
        raster: np.ndarray,
        scenarios: dict[str, Union[np.ndarray, spmatrix]],
        iterations: int,
        nodata_value: Optional[float] = None,
        cell_ids: Optional[np.ndarray] = None,
    ) -> dict[str, PropagationResult]:
        """
        Run multiple scenarios in batch.

        Parameters
        ----------
        scenarios : dict  {name: connectivity_matrix}

        Returns
        -------
        dict {name: PropagationResult}
        """
        results: dict[str, PropagationResult] = {}
        for name, C in scenarios.items():
            results[name] = self.run(
                raster=raster,
                connectivity=C,
                iterations=iterations,
                nodata_value=nodata_value,
                scenario_name=name,
                cell_ids=cell_ids,
            )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_shape(raster: np.ndarray):
        if raster.ndim == 2:
            rows, cols = raster.shape
            bands = 1
        elif raster.ndim == 3:
            rows, cols, bands = raster.shape
        else:
            raise ValueError(f"raster must be 2-D or 3-D, got shape {raster.shape}")
        return rows, cols, bands

    @staticmethod
    def _build_nodata_mask(raster: np.ndarray, nodata_value) -> np.ndarray:
        """bool mask True where cell IS nodata."""
        if nodata_value is None:
            return np.zeros(raster.shape[:2], dtype=bool)
        if raster.ndim == 3:
            return np.any(raster == nodata_value, axis=2)
        return raster == nodata_value

    def _prepare_matrix(self, C, N: int):
        """Validate dimensions, cast dtype, transpose when needed, optionally normalise.

        When ``transpose_connectivity=True`` the matrix is transposed so that
        the internal formula ``C_eff @ x`` is equivalent to ``x @ T``
        (Sofia's row-vector convention).
        """
        if issparse(C):
            C = C.astype(self.dtype)
        else:
            C = np.asarray(C, dtype=self.dtype)

        if C.shape != (N, N):
            raise ValueError(
                f"Connectivity matrix shape {C.shape} does not match "
                f"domain size N={N} (expected ({N},{N}))"
            )

        # Row-normalise on the ORIGINAL matrix (before any transpose) so that
        # rows always correspond to source cells regardless of convention.
        if self.normalise:
            C = self._row_normalise(C)

        # Transpose: x @ T  ==  T.T @ x_col  ->  store T.T as the effective matrix
        if self.transpose_connectivity:
            C = C.T

        return C

    @staticmethod
    def _row_normalise(C):
        """Row-normalise *C* so each row sums to 1 (Markov chain).

        Zero rows are left as-is (absorbing states).
        Works for both dense ndarrays and scipy sparse matrices.
        """
        if issparse(C):
            from scipy.sparse import diags as sp_diags
            row_sums = np.asarray(C.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1.0
            return sp_diags(1.0 / row_sums) @ C
        else:
            row_sums = C.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            return C / row_sums

    @staticmethod
    def _flatten(raster, rows, cols, bands, nodata_mask, cell_ids=None):
        """Flatten (rows, cols[, bands]) -> (N, bands) masking nodata as 0.

        When *cell_ids* is provided, only the N valid cells are extracted
        and ordered by their cell ID (matching the matrix row order).
        """
        arr = raster.astype(np.float64)
        if bands == 1:
            arr = arr[:, :, np.newaxis]
        arr[nodata_mask] = 0.0

        if cell_ids is not None:
            # Extract valid pixels in cell-ID order
            flat_ids = cell_ids.flatten()           # (rows*cols,)
            valid_pos = np.where(flat_ids >= 0)[0]  # positions of valid cells
            sort_idx = np.argsort(flat_ids[valid_pos])  # sort by ID
            ordered_pos = valid_pos[sort_idx]
            return arr.reshape(-1, bands)[ordered_pos]  # (N, bands)

        return arr.reshape(rows * cols, bands)

    def _propagate(self, C, x: np.ndarray, n: int) -> np.ndarray:
        """Apply C n times to x.  x shape: (N, bands)."""
        if self.mode == "discrete":
            return self._discrete(C, x, n)
        else:
            return self._continuous(C, x, n)

    @staticmethod
    def _discrete(C, x: np.ndarray, n: int) -> np.ndarray:
        """
        Compute C_eff^n * x iteratively, where C_eff has already been
        transposed in _prepare_matrix if transpose_connectivity=True.
        For sparse C, avoids forming the full power explicitly.
        Dense matrices use numpy matrix power for small N.
        """
        if issparse(C):
            result = x.copy()
            for _ in range(n):
                result = C @ result
            return result
        else:
            N = C.shape[0]
            if N <= 4096:
                Cn = np.linalg.matrix_power(C, n)
                return Cn @ x
            else:
                result = x.copy()
                for _ in range(n):
                    result = C @ result
                return result

    @staticmethod
    def _continuous(C, x: np.ndarray, n: int) -> np.ndarray:
        """
        Compute expm(n*C) * x using Krylov subspace method (sparse)
        or scipy.linalg.expm (dense).
        """
        if issparse(C):
            # expm_multiply is memory-efficient for sparse matrices
            return expm_multiply(float(n) * C, x)
        else:
            from scipy.linalg import expm as expm_dense
            return expm_dense(float(n) * C) @ x

    @staticmethod
    def _reshape(x_out, rows, cols, bands, nodata_mask, nodata_value, cell_ids=None):
        """Reshape (N, bands) back to (rows, cols, bands) restoring nodata.

        When *cell_ids* is provided, scatters the N valid cells back to
        their original 2-D positions; all other cells are filled with
        *nodata_value* (or 0 if None).
        """
        if cell_ids is not None:
            flat_ids = cell_ids.flatten()           # (rows*cols,)
            valid_pos = np.where(flat_ids >= 0)[0]
            sort_idx = np.argsort(flat_ids[valid_pos])
            ordered_pos = valid_pos[sort_idx]

            fill = nodata_value if nodata_value is not None else 0.0
            output_flat = np.full((rows * cols, bands), fill, dtype=x_out.dtype)
            output_flat[ordered_pos] = x_out
            return output_flat.reshape(rows, cols, bands)

        arr = x_out.reshape(rows, cols, bands)
        if nodata_value is not None:
            arr[nodata_mask] = nodata_value
        return arr
