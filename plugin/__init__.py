"""
QGIS Plugin entry point – SpatialConnect.

QGIS calls classFactory() to obtain the plugin instance.
"""

import glob, os, sys as _sys

# realpath resolves symlinks (the plugin dir is a symlink to the project's
# plugin/ folder, so abspath would point to the QGIS plugins directory instead
# of the project root).
_plugin_dir = os.path.dirname(os.path.realpath(__file__))
_project_root = os.path.dirname(_plugin_dir)
if _project_root not in _sys.path:
    _sys.path.insert(0, _project_root)

# Expose packages installed in the project venv to QGIS's Python.
# Works as long as the venv Python ABI matches QGIS's Python (usually both
# are the system python3.x, so site-packages are binary-compatible).
_venv_site_pkgs = glob.glob(
    os.path.join(_project_root, ".venv", "lib", "python3.*", "site-packages")
)
for _sp in _venv_site_pkgs:
    if _sp not in _sys.path:
        _sys.path.insert(1, _sp)


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
