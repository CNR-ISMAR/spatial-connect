"""
RasterUtils – I/O helpers for rasters using rasterio.

Provides:
* read_raster()       – load a GeoTIFF → numpy array + metadata dict
* write_raster()      – save numpy array → GeoTIFF (preserving CRS / transform)
* compute_cell_ids()  – build the cell-ID mapping used by Lagrangian connectivity
                        matrices (valid cells only, row-major cumsum, 0-indexed)
* vector_to_raster()  – burn vector features into a reference raster grid (requires fiona)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metadata container
# ---------------------------------------------------------------------------

@dataclass
class RasterMeta:
    """Geospatial metadata for a raster."""

    crs: object               # rasterio CRS object or WKT string
    transform: object         # rasterio Affine transform
    width: int
    height: int
    count: int                # number of bands
    dtype: str
    nodata: Optional[float]
    driver: str = "GTiff"
    extra: dict = field(default_factory=dict)

    def to_profile(self) -> dict:
        """Build a rasterio write profile."""
        p = {
            "driver": self.driver,
            "height": self.height,
            "width": self.width,
            "count": self.count,
            "dtype": self.dtype,
            "crs": self.crs,
            "transform": self.transform,
        }
        if self.nodata is not None:
            p["nodata"] = self.nodata
        p.update(self.extra)
        return p


# ---------------------------------------------------------------------------
# RasterUtils
# ---------------------------------------------------------------------------

class RasterUtils:
    """
    Static-method collection for raster I/O.

    All methods are ``@staticmethod``; instantiate if you need
    to store a common output CRS or transform override.
    """

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    @staticmethod
    def read_raster(
        path: str | Path,
        bands: int | list[int] | None = None,
    ) -> tuple[np.ndarray, RasterMeta]:
        """
        Read a raster from *path*.

        Parameters
        ----------
        path : str | Path
        bands : int or list[int], optional
            1-based band indices.  If None, all bands are read.

        Returns
        -------
        (array, meta)
            array shape: (rows, cols) for single band,
                         (rows, cols, n_bands) for multi-band.
        """
        try:
            import rasterio
        except ImportError:
            raise ImportError("rasterio is required for raster I/O.")

        with rasterio.open(path) as src:
            if bands is None:
                bands_to_read = list(range(1, src.count + 1))
            elif isinstance(bands, int):
                bands_to_read = [bands]
            else:
                bands_to_read = list(bands)

            data = src.read(bands_to_read)          # (B, rows, cols)
            meta = RasterMeta(
                crs=src.crs,
                transform=src.transform,
                width=src.width,
                height=src.height,
                count=len(bands_to_read),
                dtype=str(src.dtypes[0]),
                nodata=src.nodata,
            )

        # reorder to (rows, cols, B) then squeeze
        data = np.moveaxis(data, 0, -1)             # (rows, cols, B)
        if data.shape[2] == 1:
            data = data[:, :, 0]                    # (rows, cols)

        logger.info("Read raster %s  shape=%s  nodata=%s", path, data.shape, meta.nodata)
        return data, meta

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    @staticmethod
    def write_raster(
        path: str | Path,
        array: np.ndarray,
        meta: RasterMeta,
        nodata_value: Optional[float] = None,
        compress: str = "lzw",
    ) -> None:
        """
        Write *array* to *path* as a GeoTIFF.

        Parameters
        ----------
        path : str | Path
        array : ndarray, shape (rows, cols) or (rows, cols, bands)
        meta : RasterMeta  – geospatial metadata (CRS, transform …)
        nodata_value : float, optional  – overrides meta.nodata
        compress : str  – compression algorithm
        """
        try:
            import rasterio
        except ImportError:
            raise ImportError("rasterio is required for raster I/O.")

        if array.ndim == 2:
            array = array[:, :, np.newaxis]         # add band dim

        rows, cols, bands = array.shape

        out_meta = meta.to_profile()
        out_meta.update({
            "height": rows,
            "width": cols,
            "count": bands,
            "dtype": str(array.dtype),
            "compress": compress,
        })
        if nodata_value is not None:
            out_meta["nodata"] = nodata_value

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(path, "w", **out_meta) as dst:
            for b in range(bands):
                dst.write(array[:, :, b], b + 1)

        logger.info("Wrote raster %s  shape=%s", path, array.shape)

    # ------------------------------------------------------------------
    # Cell ID mapping (masked-domain support)
    # ------------------------------------------------------------------

    @staticmethod
    def vector_to_raster(
        vector_path: "str | Path",
        reference_raster_path: "str | Path",
        attribute: "str | None" = None,
        nodata: float = 0.0,
        all_touched: bool = False,
    ) -> "tuple[np.ndarray, RasterMeta]":
        """
        Burn vector features into a raster grid matching *reference_raster_path*.

        Parameters
        ----------
        vector_path : str | Path
            Path to a vector file (Shapefile, GeoJSON, GPKG …).
        reference_raster_path : str | Path
            GeoTIFF used to define the output CRS, transform, and grid shape.
        attribute : str, optional
            Feature attribute to use as burn value.  If ``None``, all
            features are burned with value 1.0.
        nodata : float
            Fill value for cells not covered by any feature.  Default 0.
        all_touched : bool
            When ``True``, all pixels touching a geometry are burned.

        Returns
        -------
        (array, meta)
            array shape (rows, cols), dtype float64.
            meta matches the reference raster (CRS, transform, nodata=nodata).
        """
        try:
            import rasterio
            from rasterio.features import rasterize
        except ImportError:
            raise ImportError("rasterio is required for vector_to_raster.")
        try:
            import fiona
        except ImportError:
            raise ImportError(
                "fiona is required for vector_to_raster.  "
                "Install it with: pip install fiona"
            )

        with rasterio.open(reference_raster_path) as ref:
            transform = ref.transform
            width = ref.width
            height = ref.height
            crs = ref.crs

        with fiona.open(str(vector_path)) as src:
            if attribute is not None:
                shapes = [
                    (feat["geometry"], feat["properties"][attribute])
                    for feat in src
                    if feat["geometry"] is not None
                ]
            else:
                shapes = [
                    (feat["geometry"], 1.0)
                    for feat in src
                    if feat["geometry"] is not None
                ]

        if shapes:
            result = rasterize(
                shapes,
                out_shape=(height, width),
                transform=transform,
                fill=nodata,
                dtype=np.float64,
                all_touched=all_touched,
            )
        else:
            result = np.full((height, width), nodata, dtype=np.float64)

        meta = RasterMeta(
            crs=crs,
            transform=transform,
            width=width,
            height=height,
            count=1,
            dtype="float64",
            nodata=nodata,
        )
        logger.info(
            "Rasterized %d features from %s  shape=(%d, %d)",
            len(shapes), vector_path, height, width,
        )
        return result, meta

    @staticmethod
    def compute_cell_ids(
        mask: np.ndarray,
        sea_value: float = 1.0,
        nodata_value: Optional[float] = None,
    ) -> np.ndarray:
        """
        Build a 2-D cell-ID array that matches the row/column ordering
        used by Lagrangian particle-model connectivity matrices.

        Valid cells (``mask == sea_value``) receive integer IDs
        **0, 1, 2, …, N-1** in row-major (top-to-bottom, left-to-right)
        order.  Invalid cells (land / nodata) receive **-1**.

        This replicates Sofia's xarray construction::

            griglia_id = mask.where(
                mask != 1,
                mask.stack(cell=('y_c','x_c'))
                    .where(lambda x: x == 1)
                    .cumsum('cell')
                    .unstack()
            ) - 1

        Parameters
        ----------
        mask : ndarray, shape (rows, cols) or (1, rows, cols)
            Land/sea mask.  A band dimension produced by rioxarray is
            automatically squeezed away.
        sea_value : float
            Pixel value that marks an active (sea/valid) cell.  Default 1.
        nodata_value : float, optional
            If given, cells with this value are also excluded even when
            they equal ``sea_value``.

        Returns
        -------
        cell_ids : ndarray of int64, shape (rows, cols)
            ``-1`` for invalid cells, ``0 … N-1`` for valid cells.

        Example
        -------
        >>> mask = np.array([[0, 1, 1], [1, 0, 1]])
        >>> RasterUtils.compute_cell_ids(mask)
        array([[-1,  0,  1],
               [ 2, -1,  3]])
        """
        if mask.ndim == 3:
            mask = mask[0]          # drop band dimension (rioxarray convention)

        valid = (mask == sea_value)
        if nodata_value is not None:
            valid &= (mask != nodata_value)

        flat_valid = valid.flatten()                     # (rows*cols,)
        cumsum_ids = np.cumsum(flat_valid) - 1           # 0-indexed cumulative sum
        cell_ids_flat = np.where(flat_valid, cumsum_ids, -1).astype(np.int64)
        return cell_ids_flat.reshape(mask.shape)
