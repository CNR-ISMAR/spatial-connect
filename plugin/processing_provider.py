"""
QGIS Processing provider for SpatialConnect.

Exposes the propagation as a standard Processing algorithm so it can be:
  - used in the Processing Toolbox
  - chained in Graphical Models
  - run in batch mode
  - called from PyQGIS scripts
"""

from __future__ import annotations

import os

from qgis.core import (
    QgsProcessingProvider,
    QgsProcessingAlgorithm,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterFile,
    QgsProcessingParameterEnum,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingException,
    QgsRasterLayer,
    QgsProject,
)


# ============================================================================
# Provider
# ============================================================================

class SpatialConnectProvider(QgsProcessingProvider):
    """Processing provider that groups all SpatialConnect algorithms."""

    def id(self):
        return "spatialconnect"

    def name(self):
        return "SpatialConnect"

    def icon(self):
        from qgis.PyQt.QtGui import QIcon
        icon_path = os.path.join(os.path.dirname(__file__), "resources", "icon.png")
        return QIcon(icon_path) if os.path.exists(icon_path) else super().icon()

    def loadAlgorithms(self):
        self.addAlgorithm(PropagateRasterAlgorithm())


# ============================================================================
# Algorithm 1: single scenario
# ============================================================================

class PropagateRasterAlgorithm(QgsProcessingAlgorithm):
    """
    Apply x(t+n*dt) = x(t) * T^n to a raster using a Lagrangian transition matrix T.

    Parameters visible in the Processing Toolbox
    --------------------------------------------
    INPUT      - raster layer (initial distribution, GeoTIFF)
    MATRIX     - transition matrix file (.mtx / .npz)
    ITERATIONS - integer n (each step = 1 dt of the particle model)
    MODE       - discrete | continuous
    CLIP_NEG   - bool
    TRANSPOSE  - bool  (True = x*T, T[i,j]=flow i->j; False = T*x, T[i,j]=contribution j->i)
    NORMALISE  - bool
    NODATA     - float (optional, auto-detected from raster)
    OUTPUT     - destination raster

    Hidden / not exposed in UI (kept for programmatic use)
    -------------------------------------------------------
    none
    """

    INPUT      = "INPUT"
    MASK       = "MASK"
    MATRIX     = "MATRIX"
    ITERATIONS = "ITERATIONS"
    MODE       = "MODE"
    TRANSPOSE  = "TRANSPOSE"
    CLIP_NEG   = "CLIP_NEGATIVES"
    NORMALISE  = "NORMALISE"
    NODATA     = "NODATA"
    OUTPUT     = "OUTPUT"

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.INPUT,
            "Input raster (GeoTIFF - initial spatial distribution)"
        ))
        mask_param = QgsProcessingParameterRasterLayer(
            self.MASK,
            "Domain mask raster (optional - defines which cells exist in the matrix)",
            optional=True,
        )
        self.addParameter(mask_param)
        self.addParameter(QgsProcessingParameterFile(
            self.MATRIX,
            "Transition matrix file  (.mtx MatrixMarket or .npz scipy sparse)",
            extension="",
            optional=False,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.ITERATIONS,
            "Number of time steps  (each step = 1 dt of the particle-tracking model)",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=1, minValue=1,
        ))
        self.addParameter(QgsProcessingParameterEnum(
            self.MODE, "Propagation mode",
            options=["discrete  x*T^n", "continuous  x*expm(n*T)"],
            defaultValue=0,
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            self.CLIP_NEG, "Clip negative values to 0",
            defaultValue=True,
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            self.TRANSPOSE,
            "Matrix convention: T[i,j] = flow from i -> j  (x*T row-vector)"
            " - uncheck if T[i,j] = contribution from j to i  (T*x column-vector)",
            defaultValue=True,
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            self.NORMALISE,
            "Row-normalise matrix (conserves total mass - Markov chain)",
            defaultValue=False,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.NODATA,
            "NoData value (leave blank -> auto-detected from raster file)",
            type=QgsProcessingParameterNumber.Double,
            optional=True,
        ))
        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT, "Output propagated raster"
        ))

    def processAlgorithm(self, parameters, context, feedback):
        from .core import SpatialPropagator, MatrixLoader, RasterUtils

        # -- inputs 
        raster_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        matrix_path  = self.parameterAsFile(parameters, self.MATRIX, context)
        iterations   = self.parameterAsInt(parameters, self.ITERATIONS, context)
        mode_idx     = self.parameterAsEnum(parameters, self.MODE, context)
        mode         = ["discrete", "continuous"][mode_idx]
        clip_neg     = self.parameterAsBool(parameters, self.CLIP_NEG, context)
        transpose    = self.parameterAsBool(parameters, self.TRANSPOSE, context)
        normalise    = self.parameterAsBool(parameters, self.NORMALISE, context)
        out_path     = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)

        # Nodata: use the explicitly provided value, otherwise fall back to
        # the value embedded in the raster file (meta.nodata).
        if parameters.get(self.NODATA) is not None:
            nodata_val = self.parameterAsDouble(parameters, self.NODATA, context)
        else:
            nodata_val = None

        feedback.setProgress(5)
        feedback.pushInfo(f"Reading raster: {raster_layer.source()}")
        array, meta = RasterUtils.read_raster(raster_layer.source())
        if nodata_val is None and meta.nodata is not None:
            nodata_val = meta.nodata
        feedback.pushInfo(f"NoData value: {nodata_val}")

        feedback.setProgress(20)
        feedback.pushInfo(f"Loading transition matrix: {matrix_path}")
        import numpy as np
        loader = MatrixLoader()
        mat = loader.load(matrix_path)

        # Build cell-ID mapping
        # Priority: explicit MASK parameter -> auto-detect from NaN/nodata cells in the raster
        mask_layer = self.parameterAsRasterLayer(parameters, self.MASK, context)
        if mask_layer is not None:
            feedback.pushInfo(f"Building cell-ID mapping from explicit mask: {mask_layer.source()}")
            mask_array, _ = RasterUtils.read_raster(mask_layer.source())
            cell_ids = RasterUtils.compute_cell_ids(mask_array)
        else:
            cell_ids = None
            # Replicate the original xarray .dropna() approach: treat NaN and
            # nodata cells as invalid so that N = number of valid cells, not
            # rows x cols.  This allows matrices built on a masked sub-domain
            # (e.g. ocean-only) to be used without an explicit mask file.
            arr2d = array if array.ndim == 2 else array[:, :, 0]
            valid = ~np.isnan(arr2d)
            if nodata_val is not None and not np.isnan(float(nodata_val)):
                valid &= (arr2d != nodata_val)
            n_valid = int(valid.sum())
            n_matrix = mat.shape[0]
            if 0 < n_valid < arr2d.size and n_valid == n_matrix:
                flat_valid = valid.flatten()
                cumsum_ids = np.cumsum(flat_valid) - 1
                cell_ids = np.where(flat_valid, cumsum_ids, -1).reshape(arr2d.shape)
                feedback.pushInfo(
                    f"Auto cell-ID mapping: {n_valid} valid cells "
                    f"(nodata excluded) -> matches matrix ({n_matrix}x{n_matrix})"
                )
            elif n_valid != n_matrix and arr2d.size != n_matrix:
                feedback.reportError(
                    f"Matrix size ({n_matrix}x{n_matrix}) matches neither the full raster "
                    f"({arr2d.size} cells) nor the non-nodata cells ({n_valid}). "
                    f"Provide an explicit Mask raster whose valid cells number {n_matrix}.",
                    fatalError=True,
                )

        feedback.setProgress(40)
        feedback.pushInfo(
            f"Propagating: mode={mode}, n={iterations}, transpose={transpose}"
        )

        # Pre-process matrix: normalise source rows first, then transpose into
        # the column-vector convention expected by SpatialPropagator (C @ x).
        if normalise:
            mat = SpatialPropagator._row_normalise(mat)
        if transpose:
            mat = mat.T

        propagator = SpatialPropagator(
            mode=mode,
            clip_negative=clip_neg,
        )
        result = propagator.run(
            raster=array,
            connectivity=mat,
            iterations=iterations,
            nodata_value=nodata_val,
            scenario_name="output",
            cell_ids=cell_ids,
        )

        feedback.setProgress(85)
        feedback.pushInfo(f"Writing output: {out_path}")
        RasterUtils.write_raster(out_path, result.output, meta, nodata_value=nodata_val)

        feedback.setProgress(100)
        return {self.OUTPUT: out_path}

    # -- metadata 

    def name(self):        return "propagate_raster"
    def displayName(self): return "Propagate Raster"
    def group(self):       return ""
    def groupId(self):     return ""
    def shortHelpString(self): return (
        "Apply a Lagrangian transition matrix T to a GeoTIFF raster x for n time steps:\n\n"
        "  discrete:    x(t+n*dt) = x(t) * T^n\n"
        "  continuous:  x(t+n*dt) = x(t) * expm(n*T)\n\n"
        "Supported matrix formats: MatrixMarket (.mtx) or NumPy sparse (.npz).\n\n"
        "Domain mask (optional but recommended):\n"
        "  If the matrix was built on a masked sub-domain (e.g. ocean-only cells),\n"
        "  provide the reference mask raster here. This ensures the cell numbering\n"
        "  used by the matrix matches the raster pixels exactly, regardless of the\n"
        "  values in the input raster. If omitted, the mask is auto-detected from\n"
        "  NaN/NoData cells in the input raster (works when input and mask have the\n"
        "  same nodata pattern).\n\n"
        "Matrix convention (TRANSPOSE):\n"
        "  [x] checked (default) - T[i,j] = fraction flowing from cell i to cell j\n"
        "    (x*T row-vector convention)\n"
        "  [ ] unchecked          - T[i,j] = contribution from cell j to cell i\n"
        "    (T*x column-vector convention)\n\n"
        "Non-ocean cells (NaN/NoData) are automatically excluded so that the\n"
        "matrix size N can be smaller than rows x cols of the raster."
    )

    def createInstance(self):
        return PropagateRasterAlgorithm()


# ============================================================================
# Helpers
# ============================================================================

# (no path helpers needed - core/ is bundled inside this package)
