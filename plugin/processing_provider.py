"""
QGIS Processing provider for SpatialConnect.

Exposes the propagation as a standard Processing algorithm so it can be:
  • used in the Processing Toolbox
  • chained in Graphical Models
  • run in batch mode
  • called from PyQGIS scripts
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

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
    Apply x(t+dt) = x(t) · T^n to a raster using a Lagrangian transition matrix T.

    Parameters visible in the Processing Toolbox
    --------------------------------------------
    INPUT      – raster layer (initial distribution)
    MASK       – optional land/sea mask raster (same grid as INPUT)
    MATRIX     – transition matrix file (.mtx / .npz)
    ITERATIONS – integer n (each step = 1 dt of the particle model)
    MODE       – discrete | continuous
    TRANSPOSE  – bool  (True = x·T convention; False = C·x legacy)
    CLIP_NEG   – bool
    NODATA     – float
    OUTPUT     – destination raster
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
            self.INPUT, "Input raster layer (initial distribution)"
        ))
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.MASK,
            "Land/sea mask raster (optional — use when matrix covers only valid cells)",
            optional=True,
        ))
        self.addParameter(QgsProcessingParameterFile(
            self.MATRIX,
            "Transition matrix file  (.mtx MatrixMarket or .npz scipy sparse)",
            extension="",
            optional=False,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.ITERATIONS, "Number of time steps (each = 1 dt of the particle model)",
            type=QgsProcessingParameterNumber.Integer,
            defaultValue=1, minValue=1,
        ))
        self.addParameter(QgsProcessingParameterEnum(
            self.MODE, "Propagation mode",
            options=["discrete  (C^n · x)", "continuous  (expm(n·C) · x)"],
            defaultValue=0,
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            self.TRANSPOSE,
            "Use transition convention (x·T) — True for Lagrangian particle models",
            defaultValue=True,
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            self.CLIP_NEG, "Clip negative values to 0",
            defaultValue=True,
        ))
        self.addParameter(QgsProcessingParameterBoolean(
            self.NORMALISE,
            "Row-normalise matrix (Markov chain — conserves total mass)",
            defaultValue=False,
        ))
        self.addParameter(QgsProcessingParameterNumber(
            self.NODATA, "NoData value",
            type=QgsProcessingParameterNumber.Double,
            defaultValue=-9999, optional=True,
        ))
        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT, "Output propagated raster"
        ))

    def processAlgorithm(self, parameters, context, feedback):
        _ensure_core_importable()

        from core import SpatialPropagator, MatrixLoader, RasterUtils

        # ── inputs ──────────────────────────────────────────────────
        raster_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        mask_layer   = self.parameterAsRasterLayer(parameters, self.MASK, context)
        matrix_path  = self.parameterAsFile(parameters, self.MATRIX, context)
        iterations   = self.parameterAsInt(parameters, self.ITERATIONS, context)
        mode_idx     = self.parameterAsEnum(parameters, self.MODE, context)
        mode         = ["discrete", "continuous"][mode_idx]
        transpose    = self.parameterAsBool(parameters, self.TRANSPOSE, context)
        clip_neg     = self.parameterAsBool(parameters, self.CLIP_NEG, context)
        normalise    = self.parameterAsBool(parameters, self.NORMALISE, context)
        out_path     = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)

        try:
            nodata_val = self.parameterAsDouble(parameters, self.NODATA, context)
        except Exception:
            nodata_val = None

        feedback.setProgress(5)
        feedback.pushInfo(f"Reading raster: {raster_layer.source()}")
        array, meta = RasterUtils.read_raster(raster_layer.source())
        if nodata_val is None and meta.nodata is not None:
            nodata_val = meta.nodata

        feedback.setProgress(20)
        feedback.pushInfo(f"Loading transition matrix: {matrix_path}")
        loader = MatrixLoader()
        import numpy as np
        mat = loader.load(matrix_path)

        # Build cell-ID mapping from mask (if provided)
        cell_ids = None
        if mask_layer is not None:
            feedback.pushInfo(f"Building cell-ID mapping from mask: {mask_layer.source()}")
            mask_array, _ = RasterUtils.read_raster(mask_layer.source())
            cell_ids = RasterUtils.compute_cell_ids(mask_array)

        feedback.setProgress(40)
        feedback.pushInfo(
            f"Propagating: mode={mode}, n={iterations}, transpose={transpose}"
        )
        propagator = SpatialPropagator(
            mode=mode, transpose_connectivity=transpose,
            clip_negative=clip_neg, normalise=normalise,
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

    # ── metadata ────────────────────────────────────────────────────────────

    def name(self):        return "propagate_raster"
    def displayName(self): return "Propagate Raster (single scenario)"
    def group(self):       return "SpatialConnect"
    def groupId(self):     return "spatialconnect"
    def shortHelpString(self): return (
        "Apply a transition matrix T to a raster x for n time steps:\n\n"
        "  x(t+n·dt) = x(t) · T^n  (discrete, default)\n"
        "  x(t+n·dt) = x(t) · expm(n·T)  (continuous)\n\n"
        "Transition matrix formats: MatrixMarket (.mtx) or NumPy sparse (.npz).\n\n"
        "Provide a land/sea mask to handle domains where N < rows×cols."
    )

    def createInstance(self):
        return PropagateRasterAlgorithm()


# ============================================================================
# Helpers
# ============================================================================

def _ensure_core_importable():
    """Add the parent package directory to sys.path if needed."""
    parent = str(Path(__file__).parent.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
