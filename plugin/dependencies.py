"""
Runtime dependency checker/installer for SpatialConnect.

Called once at plugin load time.  If a required package is missing it tries to
install it with pip into the same Python that QGIS is running (sys.executable),
then shows a persistent message-bar notification.

Usage (from __init__.py, before any other import)::

    from .dependencies import ensure_dependencies
    ensure_dependencies()
"""

from __future__ import annotations

import importlib
import subprocess
import sys

# package-name -> importable module name (they differ for scikit-image etc.)
REQUIRED: dict[str, str] = {
    "scipy": "scipy",
    "rasterio": "rasterio",
    "numpy": "numpy",
}


def _pip_install(packages: list[str]) -> tuple[bool, str]:
    """Run ``sys.executable -m pip install <packages>``.

    Bootstraps pip via ensurepip if missing.
    Returns (success, stderr_output).
    """
    # Bootstrap pip if not available
    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "--version"],
        capture_output=True,
    )
    if pip_check.returncode != 0:
        subprocess.run(
            [sys.executable, "-m", "ensurepip", "--upgrade"],
            capture_output=True,
        )

    cmd = [sys.executable, "-m", "pip", "install", "--user"] + packages
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0, result.stderr


def ensure_dependencies(iface=None) -> bool:
    """Check required packages; try to install any that are missing.

    Parameters
    ----------
    iface : QgisInterface, optional
        When provided, missing-package messages are shown in the QGIS
        message bar instead of (only) being written to the QGIS log.

    Returns
    -------
    bool
        ``True`` if all packages are available after the check/install.
    """
    missing = [
        pkg for pkg, module in REQUIRED.items()
        if not _is_importable(module)
    ]

    if not missing:
        return True

    _log(f"SpatialConnect: missing packages: {missing}. Trying to install...", iface)

    ok, stderr = _pip_install(missing)
    if not ok:
        apt_cmd = "sudo apt install " + " ".join(
            f"python3-{p}" for p in missing
        )
        msg = (
            f"SpatialConnect: could not auto-install {missing}.\n"
            f"Run one of the following:\n"
            f"  pip install {' '.join(missing)}\n"
            f"  {apt_cmd}\n"
            f"(Python used by QGIS: {sys.executable})\n"
            f"Details: {stderr}"
        )
        _log(msg, iface, level="critical")
        return False

    # verify again after install (new modules are now on sys.path)
    importlib.invalidate_caches()
    still_missing = [
        pkg for pkg, module in REQUIRED.items()
        if not _is_importable(module)
    ]
    if still_missing:
        msg = (
            f"SpatialConnect: installed {missing} but still cannot import "
            f"{still_missing}. You may need to restart QGIS."
        )
        _log(msg, iface, level="warning")
        return False

    _log(f"SpatialConnect: installed {missing} successfully.", iface)
    return True


# -- helpers 

def _is_importable(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _log(msg: str, iface=None, level: str = "info") -> None:
    """Write to QGIS message bar (if iface given) and to the QGIS log."""
    try:
        from qgis.core import Qgis, QgsMessageLog
        qgis_levels = {
            "info":     Qgis.Info,
            "warning":  Qgis.Warning,
            "critical": Qgis.Critical,
        }
        QgsMessageLog.logMessage(msg, "SpatialConnect", qgis_levels.get(level, Qgis.Info))

        if iface is not None:
            bar_levels = {
                "info":     Qgis.Info,
                "warning":  Qgis.Warning,
                "critical": Qgis.Critical,
            }
            iface.messageBar().pushMessage(
                "SpatialConnect",
                msg,
                level=bar_levels.get(level, Qgis.Info),
                duration=0,   # persistent until user dismisses
            )
    except Exception:
        # Fallback if called outside a QGIS environment (e.g. during tests)
        print(msg, file=sys.stderr)
