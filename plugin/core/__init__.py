"""
Spatial Matrix Propagator - core package.

Provides the computation engine for applying a connectivity matrix
C to a spatial raster x for n iterations:

    x' = C^n * x

Supports sparse matrices (scipy), multiple scenarios and both
discrete (matrix power) and continuous (matrix exponential) modes.
"""

from .propagator import SpatialPropagator
from .matrix_loader import MatrixLoader
from .raster_utils import RasterUtils

__all__ = ["SpatialPropagator", "MatrixLoader", "RasterUtils"]
