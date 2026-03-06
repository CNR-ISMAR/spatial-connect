"""
QGIS Plugin entry point – SpatialConnect.

QGIS calls classFactory() to obtain the plugin instance.
"""

import glob, os, sys as _sys

# core/ is now bundled inside this plugin folder, so no project-root path
# injection is needed.  We only expose the project venv's site-packages so
# that scipy/rasterio installed there are visible to QGIS's Python during
# development (symlink install).  This has no effect in a ZIP install where
# dependencies are expected to come from QGIS's own Python or be auto-installed
# by dependencies.py.
_plugin_dir = os.path.dirname(os.path.realpath(__file__))
_project_root = os.path.dirname(_plugin_dir)
_venv_site_pkgs = glob.glob(
    os.path.join(_project_root, ".venv", "lib", "python3.*", "site-packages")
)
for _sp in _venv_site_pkgs:
    if _sp not in _sys.path:
        _sys.path.insert(0, _sp)


# noinspection PyPep8Naming
def classFactory(iface):  # pylint: disable=invalid-name
    """Load SpatialConnectPlugin class.

    :param iface: A QGIS interface instance.
    :type iface: QgsInterface
    """
    from .dependencies import ensure_dependencies
    ensure_dependencies(iface)

    from .spatial_connect import SpatialConnectPlugin
    return SpatialConnectPlugin(iface)
