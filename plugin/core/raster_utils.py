"""
RasterUtils - I/O helpers for rasters.

Backends (tried in order):
  1. rasterio   - preferred when available (install via ``pip install rasterio``)
  2. osgeo.gdal - always present inside QGIS on every platform (Windows included)

Provides:
* read_raster()       - load a GeoTIFF -> numpy array + metadata dict
* write_raster()      - save numpy array -> GeoTIFF (preserving CRS / transform)
* compute_cell_ids()  - build the cell-ID mapping used by Lagrangian connectivity
                        matrices (valid cells only, row-major cumsum, 0-indexed)
* vector_to_raster()  - burn vector features into a reference raster grid
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
        """Build a rasterio write profile.

        Normalises ``crs`` and ``transform`` so the profile is accepted by
        ``rasterio.open(..., 'w', ...)`` regardless of whether the metadata
        was originally produced by rasterio or by osgeo.gdal.
        """
        crs = self.crs
        transform = self.transform

        try:
            import rasterio.crs as _rcrs

            # --- CRS: convert WKT string → rasterio CRS ---
            if isinstance(crs, str) and crs:
                crs = _rcrs.CRS.from_wkt(crs)

            # --- Transform: convert GDAL 6-tuple → affine.Affine ---
            if isinstance(transform, (tuple, list)):
                from affine import Affine
                gt = transform
                transform = Affine(gt[1], gt[2], gt[0], gt[4], gt[5], gt[3])
        except ImportError:
            # rasterio not available; caller should use the GDAL write path
            # instead of to_profile(), but return a best-effort dict anyway.
            pass

        p = {
            "driver": self.driver,
            "height": self.height,
            "width": self.width,
            "count": self.count,
            "dtype": self.dtype,
            "crs": crs,
            "transform": transform,
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
    # Internal backend helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gdal_read(path: "str | Path") -> "tuple[np.ndarray, RasterMeta]":
        """Read a raster using osgeo.gdal (QGIS built-in)."""
        from osgeo import gdal
        ds = gdal.Open(str(path))
        if ds is None:
            raise IOError(f"GDAL could not open raster: {path}")
        gt = ds.GetGeoTransform()       # 6-tuple
        wkt = ds.GetProjection()        # WKT string
        nodata = None
        bands_data = []
        for b in range(1, ds.RasterCount + 1):
            band = ds.GetRasterBand(b)
            nd = band.GetNoDataValue()
            if nd is not None and nodata is None:
                nodata = nd
            bands_data.append(band.ReadAsArray().astype(np.float64))
        data = np.stack(bands_data, axis=-1)   # (rows, cols, B)
        if data.shape[2] == 1:
            data = data[:, :, 0]
        meta = RasterMeta(
            crs=wkt if wkt else None,
            transform=gt,
            width=ds.RasterXSize,
            height=ds.RasterYSize,
            count=ds.RasterCount,
            dtype=str(data.dtype),
            nodata=nodata,
        )
        ds = None  # close dataset
        return data, meta

    @staticmethod
    def _gdal_write(
        path: "str | Path",
        array: np.ndarray,
        meta: "RasterMeta",
        nodata_value: "Optional[float]" = None,
        compress: str = "lzw",
    ) -> None:
        """Write a raster using osgeo.gdal (QGIS built-in)."""
        from osgeo import gdal

        _dtype_map = {
            "float32": gdal.GDT_Float32,
            "float64": gdal.GDT_Float64,
            "int16": gdal.GDT_Int16,
            "int32": gdal.GDT_Int32,
            "uint8": gdal.GDT_Byte,
            "uint16": gdal.GDT_UInt16,
            "int8": gdal.GDT_Int16,   # promote; GDAL has no GDT_Int8
        }

        if array.ndim == 2:
            array = array[:, :, np.newaxis]
        rows, cols, bands = array.shape
        gdal_dtype = _dtype_map.get(str(array.dtype), gdal.GDT_Float64)

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        options = [f"COMPRESS={compress.upper()}"]
        driver = gdal.GetDriverByName("GTiff")
        ds = driver.Create(str(path), cols, rows, bands, gdal_dtype, options)

        # Geotransform (accept both rasterio Affine and GDAL 6-tuple)
        gt = meta.transform
        if gt is not None:
            if hasattr(gt, "to_gdal"):
                gt = gt.to_gdal()        # rasterio Affine → GDAL 6-tuple
            ds.SetGeoTransform(list(gt))

        # Projection (accept rasterio CRS or WKT string)
        crs = meta.crs
        if crs is not None:
            if hasattr(crs, "to_wkt"):
                wkt = crs.to_wkt()       # rasterio CRS
            else:
                wkt = str(crs)           # already WKT
            ds.SetProjection(wkt)

        nd = nodata_value if nodata_value is not None else meta.nodata
        for b in range(bands):
            band = ds.GetRasterBand(b + 1)
            band.WriteArray(array[:, :, b])
            if nd is not None:
                band.SetNoDataValue(float(nd))
        ds.FlushCache()
        ds = None

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

        Uses **rasterio** when available, otherwise falls back to
        **osgeo.gdal** (always present inside QGIS).

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
            rasterio = None

        if rasterio is not None:
            with rasterio.open(path) as src:
                if bands is None:
                    bands_to_read = list(range(1, src.count + 1))
                elif isinstance(bands, int):
                    bands_to_read = [bands]
                else:
                    bands_to_read = list(bands)

                data = src.read(bands_to_read)      # (B, rows, cols)
                meta = RasterMeta(
                    crs=src.crs,
                    transform=src.transform,
                    width=src.width,
                    height=src.height,
                    count=len(bands_to_read),
                    dtype=str(src.dtypes[0]),
                    nodata=src.nodata,
                )

            data = np.moveaxis(data, 0, -1)         # (rows, cols, B)
            if data.shape[2] == 1:
                data = data[:, :, 0]
        else:
            # GDAL fallback
            try:
                data, meta = RasterUtils._gdal_read(path)
                if bands is not None:
                    band_list = [bands] if isinstance(bands, int) else list(bands)
                    if data.ndim == 2:
                        data = data[:, :, np.newaxis]
                    data = data[:, :, [b - 1 for b in band_list]]
                    if data.shape[2] == 1:
                        data = data[:, :, 0]
                    meta.count = len(band_list) if not isinstance(bands, int) else 1
            except ImportError:
                raise ImportError(
                    "Neither rasterio nor osgeo.gdal is available. "
                    "Install rasterio (pip install rasterio) or run inside QGIS."
                )

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

        Uses **rasterio** when available, otherwise falls back to
        **osgeo.gdal** (always present inside QGIS).

        Parameters
        ----------
        path : str | Path
        array : ndarray, shape (rows, cols) or (rows, cols, bands)
        meta : RasterMeta  - geospatial metadata (CRS, transform ...)
        nodata_value : float, optional  - overrides meta.nodata
        compress : str  - compression algorithm
        """
        try:
            import rasterio
        except ImportError:
            rasterio = None

        if rasterio is not None:
            if array.ndim == 2:
                array = array[:, :, np.newaxis]
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
        else:
            try:
                RasterUtils._gdal_write(path, array, meta, nodata_value, compress)
            except ImportError:
                raise ImportError(
                    "Neither rasterio nor osgeo.gdal is available. "
                    "Install rasterio (pip install rasterio) or run inside QGIS."
                )

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

        Uses **rasterio + fiona** when available, otherwise falls back to
        **osgeo.gdal + osgeo.ogr** (always present inside QGIS).

        Parameters
        ----------
        vector_path : str | Path
            Path to a vector file (Shapefile, GeoJSON, GPKG ...).
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
            from rasterio.features import rasterize as _rasterize
            _has_rasterio = True
        except ImportError:
            _has_rasterio = False

        try:
            import fiona
            _has_fiona = True
        except ImportError:
            _has_fiona = False

        if _has_rasterio and _has_fiona:
            # ---- rasterio + fiona path ----
            with rasterio.open(reference_raster_path) as ref:
                transform = ref.transform
                width = ref.width
                height = ref.height
                crs = ref.crs

            with fiona.open(str(vector_path)) as src:
                if attribute is not None:
                    shapes = [
                        (feat["geometry"], feat["properties"][attribute])
                        for feat in src if feat["geometry"] is not None
                    ]
                else:
                    shapes = [
                        (feat["geometry"], 1.0)
                        for feat in src if feat["geometry"] is not None
                    ]

            if shapes:
                result = _rasterize(
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
                crs=crs, transform=transform,
                width=width, height=height,
                count=1, dtype="float64", nodata=nodata,
            )
        else:
            # ---- osgeo.gdal + osgeo.ogr fallback ----
            try:
                from osgeo import gdal, ogr
            except ImportError:
                raise ImportError(
                    "vector_to_raster requires either rasterio+fiona or osgeo.gdal+ogr. "
                    "Install rasterio and fiona, or run inside QGIS."
                )

            ref_ds = gdal.Open(str(reference_raster_path))
            if ref_ds is None:
                raise IOError(f"GDAL could not open reference raster: {reference_raster_path}")
            gt = ref_ds.GetGeoTransform()
            wkt = ref_ds.GetProjection()
            width = ref_ds.RasterXSize
            height = ref_ds.RasterYSize
            ref_ds = None

            vec_ds = ogr.Open(str(vector_path))
            if vec_ds is None:
                raise IOError(f"OGR could not open vector file: {vector_path}")
            layer = vec_ds.GetLayer(0)

            # Create an in-memory raster for rasterization
            mem_driver = gdal.GetDriverByName("MEM")
            mem_ds = mem_driver.Create("", width, height, 1, gdal.GDT_Float64)
            mem_ds.SetGeoTransform(list(gt))
            mem_ds.SetProjection(wkt)
            band = mem_ds.GetRasterBand(1)
            band.Fill(nodata)
            band.SetNoDataValue(nodata)

            opts = ["ALL_TOUCHED=TRUE"] if all_touched else []
            if attribute is not None:
                opts.append(f"ATTRIBUTE={attribute}")

            gdal.RasterizeLayer(mem_ds, [1], layer, options=opts)
            mem_ds.FlushCache()

            result = band.ReadAsArray().astype(np.float64)
            mem_ds = None
            vec_ds = None

            meta = RasterMeta(
                crs=wkt if wkt else None,
                transform=gt,
                width=width, height=height,
                count=1, dtype="float64", nodata=nodata,
            )

        logger.info(
            "Rasterized %s  shape=(%d, %d)", vector_path, meta.height, meta.width,
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
        **0, 1, 2, ..., N-1** in row-major (top-to-bottom, left-to-right)
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
            ``-1`` for invalid cells, ``0 ... N-1`` for valid cells.

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
