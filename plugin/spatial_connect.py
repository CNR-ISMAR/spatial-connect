"""
SpatialConnectPlugin - main QGIS plugin class.

Registers a Processing provider so the algorithm is available in the
Processing Toolbox and can be used in models / batch processes.
"""

from __future__ import annotations

import os

from qgis.core import QgsApplication


class SpatialConnectPlugin:
    """QGIS plugin implementation."""

    PLUGIN_NAME = "SpatialConnect"

    def __init__(self, iface):
        self.iface = iface
        self._provider = None

    # ------------------------------------------------------------------
    # Plugin lifecycle
    # ------------------------------------------------------------------

    def initGui(self):
        """Register the Processing provider."""
        from .processing_provider import SpatialConnectProvider
        self._provider = SpatialConnectProvider()
        QgsApplication.processingRegistry().addProvider(self._provider)

    def unload(self):
        """Remove the Processing provider."""
        if self._provider:
            QgsApplication.processingRegistry().removeProvider(self._provider)
            self._provider = None
