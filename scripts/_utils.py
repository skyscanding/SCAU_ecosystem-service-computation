"""
Shared utilities for LandTrendr + Ecosystem Services Coupling Analysis Pipeline.

Follows Nanling analysis conventions for severity/intensity classification
and raster handling.
"""
import os
import numpy as np


# ── Intensity bins (4-class, used for CSV export + stacked bar charts) ──
# Format: [(label, low, high, color_hex), ...]
INTENSITY_BINS = [
    ("Low (0.2-0.35)",      0.2, 0.35, "#f1c40f"),
    ("Moderate (0.35-0.5)", 0.35, 0.5,  "#e67e22"),
    ("High (0.5-0.65)",     0.5, 0.65,  "#e74c3c"),
    ("Very High (>0.65)",   0.65, 9.0,  "#8b0000"),
]

# ── Severity bins (3-class, for spatial classification map) ──
SEVERITY_BINS = [
    ("Low (0.2-0.35)",       0.2, 0.35, "#f1c40f"),
    ("Moderate (0.35-0.55)", 0.35, 0.55, "#e67e22"),
    ("High (>0.55)",         0.55, 9.0,  "#c0392b"),
]

# ── Duration labels and colors ──
DUR_LABELS = ["1 year", "2 years", "≥3 years"]
DUR_COLORS = ["#3498db", "#2ecc71", "#e74c3c"]

# ── LandTrendr default file names (Nanling convention) ──
DEFAULT_YOD = "yod_2009_2024.tif"
DEFAULT_MAG = "mag_2009_2024.tif"
DEFAULT_DUR = "dur_2009_2024.tif"
DEFAULT_MPY = "magperyear_2009_2024.tif"

# ── LULC class definitions (Dabaoshan / Nanling compatible) ──
LULC_CLASSES = {
    "water": 1,
    "built_up": 2,
    "unrestored": 3,
    "recovering": 4,
    "stable_vegetation": 5,
}
LULC_CLASS_NAMES = ["water", "built_up", "unrestored", "recovering", "stable_vegetation"]

# ── Default carbon pools (Mg/ha, Dabaoshan calibrated) ──
DEFAULT_CARBON_POOLS = {
    "water":              {"c_above": 0,  "c_below": 0,  "c_soil": 0,  "c_dead": 0},
    "built_up":           {"c_above": 0,  "c_below": 0,  "c_soil": 0,  "c_dead": 0},
    "unrestored":         {"c_above": 2.5,"c_below": 1.2,"c_soil": 45, "c_dead": 1.5},
    "recovering":         {"c_above": 15, "c_below": 7,  "c_soil": 60, "c_dead": 3},
    "stable_vegetation":  {"c_above": 85, "c_below": 40, "c_soil": 95, "c_dead": 8},
}

# ── Default habitat suitability scores ──
DEFAULT_HABITAT_SCORES = {
    "water": 0.8,
    "built_up": 0.0,
    "unrestored": 0.1,
    "recovering": 0.4,
    "stable_vegetation": 0.9,
}

# ── Default C and P factors for SDR/USLE ──
DEFAULT_C_FACTOR = {
    "water": 0.0,
    "built_up": 0.01,
    "unrestored": 0.35,
    "recovering": 0.08,
    "stable_vegetation": 0.003,
}
DEFAULT_P_FACTOR = {
    "water": 0.0,
    "built_up": 1.0,
    "unrestored": 1.0,
    "recovering": 1.0,
    "stable_vegetation": 1.0,
}

# ── Default phase definitions (ecological restoration phases) ──
DEFAULT_PHASES = {
    "Degradation":    [2013, 2016],
    "Transition":     [2017, 2020],
    "Consolidation":  [2021, 2025],
}

# ── Landscape metric names ──
LANDSCAPE_METRICS = ["np", "pd", "lpi", "ed", "lsi", "shdi", "contag", "mesh"]
CLASS_METRICS = ["ca", "pland", "np", "pd", "lpi", "ed", "lsi", "area_mn"]


def resolve_field(df, preferred=None, fallback_keywords=None):
    """
    Resolve a column name in a pandas DataFrame.

    Priority:
      1. If `preferred` is given and exists in columns, use it.
      2. If `fallback_keywords` given, search columns for a case-insensitive match.
      3. Raise ValueError with a list of available columns.

    Parameters
    ----------
    df : pd.DataFrame
    preferred : str | None
    fallback_keywords : list[str] | None

    Returns
    -------
    str  The resolved column name.

    Raises
    ------
    ValueError  If the column cannot be resolved.
    """
    cols = set(df.columns)

    if preferred and preferred in cols:
        return preferred

    if fallback_keywords:
        for col in cols:
            low = str(col).lower()
            for kw in fallback_keywords:
                if kw.lower() in low:
                    return col

    avail = ", ".join(sorted(str(c) for c in cols)[:30])
    wanted = preferred or (fallback_keywords or ["<none>"])
    raise ValueError(
        f"Cannot find column in DataFrame.\n"
        f"  Wanted: {wanted}\n"
        f"  Available: {avail}"
    )


def with_default(value, fallback):
    """Return value if not None/empty, else fallback."""
    return value if value is not None and value != "" else fallback


def ensure_dir(path):
    """Create directory if it does not exist, return path."""
    os.makedirs(str(path), exist_ok=True)
    return path


def load_and_clip_raster(raster_path, clip_shapes, nodata_val=0):
    """
    Load a raster and clip to geometry shapes using rasterio mask.
    Simple Nanling-style: no CRS warping - boundary must already match raster CRS.

    Parameters
    ----------
    raster_path : str       Path to raster.
    clip_shapes : list      List of GeoJSON-like geometry dicts.
    nodata_val : int/float  Nodata fill value.

    Returns
    -------
    tuple  (array_2d, meta_dict, transform, bounds)
    """
    import rasterio
    from rasterio.mask import mask as rio_mask

    with rasterio.open(raster_path) as src:
        out_image, out_transform = rio_mask(
            src, clip_shapes, crop=True, nodata=nodata_val, filled=True
        )
        out_meta = src.meta.copy()
        out_meta.update({
            'height': out_image.shape[1],
            'width': out_image.shape[2],
            'transform': out_transform
        })
        bounds = rasterio.transform.array_bounds(
            out_image.shape[1], out_image.shape[2], out_transform
        )
    return out_image[0], out_meta, out_transform, bounds


def classify_intensity(magnitude, bins=None):
    """
    Classify disturbance magnitude into 4 intensity classes (Nanling convention).

    Parameters
    ----------
    magnitude : np.ndarray  Per-pixel magnitude values.
    bins : list | None      [(label, lo, hi, color), ...], default INTENSITY_BINS.

    Returns
    -------
    np.ndarray  Integer codes 1-4 matching bin order, -1 for unclassified.
    """
    if bins is None:
        bins = INTENSITY_BINS

    result = np.full(magnitude.shape, -1, dtype=np.int16)
    for i, (label, lo, hi, color) in enumerate(bins):
        result[(magnitude >= lo) & (magnitude < hi)] = i + 1
    return result


def classify_severity(magnitude, bins=None):
    """
    Classify disturbance magnitude into 3 severity classes (Nanling convention).

    Parameters
    ----------
    magnitude : np.ndarray
    bins : list | None  [(label, lo, hi, color), ...], default SEVERITY_BINS.

    Returns
    -------
    np.ndarray  Integer codes 1-3.
    """
    if bins is None:
        bins = SEVERITY_BINS

    result = np.full(magnitude.shape, -1, dtype=np.int16)
    for i, (label, lo, hi, color) in enumerate(bins):
        result[(magnitude >= lo) & (magnitude < hi)] = i + 1
    return result
