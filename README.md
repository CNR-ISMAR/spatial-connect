# SpatialConnect

QGIS plugin and standalone Python library that propagates a spatial raster
through a **Lagrangian connectivity / transition matrix** for *n* time steps:

$$x(t + n \cdot dt) = x(t) \cdot T^n \quad \text{(discrete)}$$

$$x(t + n \cdot dt) = x(t) \cdot e^{nT} \quad \text{(continuous)}$$

where **x** is a row vector of cell values and **T** is the transition matrix
*(T[i,j] = probability that a particle in cell i moves to cell j in one dt)*.

---

## Use cases

| Scenario | x (input raster) | T (matrix) | Output |
|---|---|---|---|
| Ocean pollution dispersal | Initial concentration | Lagrangian connectivity | Plume after n days |
| Larval connectivity | Release density | Settlement probability | Recruitment map |
| Fishing footprint | Effort per cell | Fleet mobility | Real spatial impact |
| Habitat fragmentation | Habitat suitability | Landscape permeability | Reachability map |

---

## Repository structure

```
spatial-connect/
├── plugin/                  – QGIS plugin (self-contained, zippable)
│   ├── __init__.py          – QGIS classFactory
│   ├── metadata.txt         – QGIS plugin metadata
│   ├── spatial_connect.py   – plugin lifecycle (registers Processing provider)
│   ├── processing_provider.py  – Processing Toolbox algorithm
│   ├── dependencies.py      – auto-install missing packages at load time
│   └── core/                – standalone propagation library
│       ├── __init__.py
│       ├── matrix_loader.py     – load .mtx / .npz transition matrices
│       ├── propagator.py        – SpatialPropagator (discrete + continuous)
│       └── raster_utils.py      – read/write GeoTIFF, compute_cell_ids,
│                                   vector_to_raster (optional, requires fiona)
├── tests/
│   ├── conftest.py
│   ├── test_propagator.py
│   ├── test_matrix_loader.py
│   ├── test_raster_utils.py
│   └── test_integration.py
├── examples/
├── build_plugin_zip.py      – creates dist/SpatialConnect-<version>.zip  (cross-platform)
├── build_plugin_zip.sh      – same, bash shortcut for Linux/macOS
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Installation

### Standalone library

```bash
git clone https://github.com/CNR-ISMAR/spatial-connect.git
cd spatial-connect
pip install -r requirements.txt
```

### QGIS plugin

> ⚠️ **Do not use the "Download ZIP" button on GitHub** — that ZIP contains the
> whole repository and cannot be installed directly in QGIS.
> Use one of the options below instead.

**Option A — Download from GitHub Releases** (recommended for end users)

Go to the [Releases page](https://github.com/CNR-ISMAR/spatial-connect/releases)
and download `SpatialConnect-<version>.zip`.

Then in QGIS: **Plugins → Manage and Install Plugins → Install from ZIP → select the file**.

**Option B — Build the ZIP locally** (if you have the repo cloned)

```bash
python build_plugin_zip.py        # cross-platform (Linux / macOS / Windows)
# or on Linux/macOS:
./build_plugin_zip.sh
```

The file is created in `dist/SpatialConnect-<version>.zip` — install it as above.

**Option C — Symlink** (recommended for developers)

Symlink `plugin/` into the QGIS plugins directory as `SpatialConnect`:

```bash
# Linux / macOS
ln -sfn $(pwd)/plugin \
  ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/SpatialConnect

# Windows (run as administrator)
mklink /D "%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\SpatialConnect" plugin
```

Then in QGIS: **Plugins → Manage and Install Plugins → Installed → SpatialConnect → Enable**.

Changes to the source files are reflected immediately (reload plugin to pick them up).

---

## Quick start (Python API)

```python
import sys
sys.path.insert(0, "path/to/spatial-connect/plugin")   # adds core/ to the path

from core import SpatialPropagator, MatrixLoader, RasterUtils

# 1. Load raster
array, meta = RasterUtils.read_raster("initial_distribution.tif")

# 2. Load transition matrix  (.mtx = MatrixMarket, .npz = scipy sparse)
T = MatrixLoader().load("sparse_transition_matrix.mtx")

# 3. Propagate  (10 discrete steps, x·T convention)
p = SpatialPropagator(mode="discrete")
result = p.run(array, T, iterations=10)

# 4. Save
RasterUtils.write_raster("output.tif", result.output, meta)
```

### Masked domain (land/sea grid)

When the matrix covers only the *N* valid (sea) cells of an M×M raster:

```python
mask, _ = RasterUtils.read_raster("NAS-grid-5km.tif")
cell_ids = RasterUtils.compute_cell_ids(mask, sea_value=1.0)

result = p.run(array, T, iterations=10,
               nodata_value=-9999, cell_ids=cell_ids)
```

`compute_cell_ids` replicates Sofia's xarray pattern:

```python
# xarray equivalent                      # this library
x = da.stack(z=('y','x')).dropna('z')   # cell_ids = compute_cell_ids(mask)
y = x @ T                                # result = p.run(raster, T, ...)
y.unstack().reproject_match(mask)        # → result.output  (shape restored)
```

### Multiple scenarios

```python
scenarios = MatrixLoader().load_scenarios({
    "baseline":    "T_baseline.mtx",
    "high_impact": "T_high.mtx",
})
results = p.run_scenarios(array, scenarios, iterations=10, nodata_value=-9999)
for name, r in results.items():
    RasterUtils.write_raster(f"output_{name}.tif", r.output, meta)
```

### Burn vectors to raster (requires fiona)

```python
burned, meta = RasterUtils.vector_to_raster(
    "release_points.shp",
    reference_raster_path="grid.tif",
    attribute="n_particles",
)
result = p.run(burned, T, iterations=30)
```

---

## Matrix formats

| Format | Extension | How to produce |
|---|---|---|
| MatrixMarket | `.mtx` | Lagrangian particle model output (Ariane, OceanParcels, etc.) |
| NumPy sparse | `.npz` | `scipy.sparse.save_npz(path, matrix)` |

---

## Propagation options

| Parameter | Default | Description |
|---|---|---|
| `mode` | `"discrete"` | `"discrete"` = iterative multiplication; `"continuous"` = matrix exponential |
| `normalise` | `False` | Row-normalise T → Markov chain (conserves total mass) |
| `transpose_connectivity` | `True` | `True` = x·T convention (Lagrangian); `False` = C·x legacy |
| `clip_negative` | `True` | Clip negative output values to 0 |
| `nodata_value` | `None` | Cells with this value are masked during propagation |
| `cell_ids` | `None` | Masked-domain support — from `RasterUtils.compute_cell_ids()` |

---

## Processing Toolbox

The plugin registers the algorithm **"Propagate Raster"**
under the *SpatialConnect* provider.  Scriptable from PyQGIS:

```python
import processing
result = processing.run("spatialconnect:propagate_raster", {
    "INPUT":          "/data/distribution.tif",
    "MATRIX":         "/data/sparse_transition_matrix.mtx",
    "ITERATIONS":     10,
    "MODE":           0,        # 0=discrete, 1=continuous
    "NORMALISE":      False,
    "CLIP_NEGATIVES": True,
    "OUTPUT":         "/tmp/output.tif",
})
```

---

## Tests

```bash
pytest tests/ -v
```

---

## Dependencies

| Package | Required for |
|---|---|
| `numpy` | all |
| `scipy` | matrix operations, sparse I/O |
| `rasterio` | GeoTIFF read/write |
| `fiona` | `RasterUtils.vector_to_raster()` only |

---


## Future works

- **Propagate Vector** — a dedicated Processing algorithm that accepts a vector layer
  (point/polygon) as initial distribution, rasterises it onto a reference grid via
  `RasterUtils.vector_to_raster()` (requires `fiona`), runs the propagation, and
  returns a GeoTIFF output.  Parameters would include the burn attribute, fill value,
  and `all_touched` option.
- **Multi-band raster input** — propagate each band independently through the same
  matrix (e.g. one band per species, pollutant, or time snapshot) and return a
  multi-band output with identical spatial metadata.
- Batch / multi-scenario mode (loop over a folder of rasters or matrix time-slices).
- Time-series output (store all *n* intermediate steps as a multi-band raster).

---

## License

GNU General Public License v3 – see [LICENSE](LICENSE).
