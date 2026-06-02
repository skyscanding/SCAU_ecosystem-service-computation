"""
Step 3: Landscape Metrics Computation (PyLandStats)
=====================================================
Computes landscape-level and class-level metrics using PyLandStats
for each year's LULC map. Requires classified rasters or a pre-computed
LULC area CSV.

Input:  LULC classification rasters or pre-computed LULC area CSV.
Output: Landscape metrics CSVs (landscape-level + class-level).

Usage:
  python step3_landscape_metrics.py --raster-dir D:/data/LULC/ \
      --boundary D:/data/boundary.shp --output-dir D:/output/ \
      --start 2000 --end 2025
"""
import os
import sys
import time
import argparse
import warnings

import numpy as np
import pandas as pd
from pathlib import Path
from _utils import ensure_dir


def compute_landscape_metrics(
    lulc_csv=None,
    output_dir=None,
    landscape_metrics=None,
    class_metrics=None,
    metrics_level=None,
    neighborhood_rule="8",
    raster_dir=None,
    boundary_shp=None,
    year_range=None,
    figure_prefix=None,
):
    """
    Compute landscape metrics using PyLandStats.

    Parameters
    ----------
    lulc_csv : str | None     Path to pre-computed LULC area CSV (from step 2).
    output_dir : str          Output directory.
    landscape_metrics : list  Landscape-level metric names.
    class_metrics : list      Class-level metric names.
    metrics_level : list      ['landscape', 'class'] or subset.
    neighborhood_rule : str   '8' (Moore) or '4' (von Neumann).
    raster_dir : str | None   Directory with LULC rasters (if lulc_csv not provided).
    boundary_shp : str | None Study boundary shapefile.
    year_range : list | None  [start, end].
    figure_prefix : str | None Output figure prefix.

    Returns
    -------
    str  Path to landscape metrics CSV.
    """
    if landscape_metrics is None:
        landscape_metrics = ["np", "pd", "lpi", "ed", "lsi", "shdi", "contag", "mesh"]
    if class_metrics is None:
        class_metrics = ["ca", "pland", "np", "pd", "lpi", "ed", "lsi"]
    if metrics_level is None:
        metrics_level = ["landscape", "class"]
    if year_range is None:
        year_range = [2000, 2025]

    T0 = time.time()
    ensure_dir(output_dir)

    years = list(range(year_range[0], year_range[1] + 1))

    # Try importing PyLandStats
    try:
        import pylandstats as pls
        print(f"    PyLandStats version: {pls.__version__}")
    except ImportError:
        print("    ERROR: pylandstats not installed. Install with: pip install pylandstats")
        # Fallback: return template CSV
        return _save_empty_metrics(output_dir, landscape_metrics, class_metrics, years)

    land_rows = []
    class_rows = []

    if raster_dir and boundary_shp:
        from _utils import clip_raster_to_boundary, rasterize_boundary
        import geopandas as gpd
        import rasterio

        boundary_gdf = gpd.read_file(boundary_shp)
        raster_dir = Path(raster_dir)

        for yr in years:
            raster_path = raster_dir / f"lulc_{yr}.tif"
            if not raster_path.exists():
                raster_path = raster_dir / f"lulc_aligned_{yr}.tif"
            if not raster_path.exists():
                print(f"    WARNING: No raster for year {yr}, skipping.")
                continue

            print(f"  [{time.time()-T0:.0f}s] Computing metrics for year {yr}...")

            try:
                with rasterio.open(str(raster_path)) as src:
                    lulc_arr = src.read(1)
                    transform = src.transform
                    crs = src.crs

                # Rasterize boundary
                boundary_proj = boundary_gdf.to_crs(crs)
                roi = rasterize_boundary(boundary_proj, lulc_arr.shape, transform)
                lulc_arr[~roi] = 0

                # PyLandStats expects a numpy array with integer class codes
                ls = pls.Landscape(lulc_arr, neighborhood_rule=neighborhood_rule)

                # Landscape-level metrics
                if "landscape" in metrics_level:
                    land_vals = {}
                    for m in landscape_metrics:
                        try:
                            val = getattr(ls, m)(percent=False) if hasattr(ls, m) else None
                        except Exception:
                            val = None
                        land_vals[m.upper()] = round(val, 6) if val is not None else None
                    land_vals["Year"] = yr
                    land_rows.append(land_vals)

                # Class-level metrics
                if "class" in metrics_level:
                    for cls_val in np.unique(lulc_arr):
                        if cls_val == 0:
                            continue
                        cls_vals = {"Year": yr, "class_val": int(cls_val)}
                        for m in class_metrics:
                            try:
                                val = getattr(ls, m)(percent=False, class_val=cls_val) \
                                    if hasattr(ls, m) else None
                            except Exception:
                                val = None
                            cls_vals[m.upper()] = round(val, 6) if val is not None else None
                        class_rows.append(cls_vals)

            except Exception as e:
                print(f"    ERROR processing year {yr}: {e}")
                continue

    else:
        print("    No raster_dir provided. Generating template metric structure.")
        return _save_empty_metrics(output_dir, landscape_metrics, class_metrics, years)

    # Save results
    df_land = pd.DataFrame(land_rows)
    df_class = pd.DataFrame(class_rows)

    land_csv = os.path.join(output_dir, "landscape_metrics.csv")
    class_csv = os.path.join(output_dir, "class_metrics.csv")

    if not df_land.empty:
        df_land.to_csv(land_csv, index=False, encoding="utf-8-sig")
        print(f"    Landscape metrics saved: {land_csv}")
    if not df_class.empty:
        df_class.to_csv(class_csv, index=False, encoding="utf-8-sig")
        print(f"    Class metrics saved: {class_csv}")

    print(f"  DONE [Step 3, {time.time()-T0:.0f}s]")
    return land_csv


def _save_empty_metrics(output_dir, landscape_metrics, class_metrics, years):
    """Save empty template CSVs for pipeline continuity."""
    land_csv = os.path.join(output_dir, "landscape_metrics.csv")
    cls_csv = os.path.join(output_dir, "class_metrics.csv")

    cols = ["Year"] + [m.upper() for m in landscape_metrics]
    pd.DataFrame(columns=cols).to_csv(land_csv, index=False, encoding="utf-8-sig")
    print(f"    Template saved: {land_csv} (empty)")

    cls_cols = ["Year", "class_val"] + [m.upper() for m in class_metrics]
    pd.DataFrame(columns=cls_cols).to_csv(cls_csv, index=False, encoding="utf-8-sig")
    print(f"    Template saved: {cls_csv} (empty)")

    return land_csv


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Landscape Metrics Computation")
    p.add_argument("--raster-dir", default=None)
    p.add_argument("--boundary", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--start", type=int, default=2000)
    p.add_argument("--end", type=int, default=2025)
    p.add_argument("--neighborhood", default="8", choices=["4", "8"])
    p.add_argument("--figures", action="store_true")
    args = p.parse_args()

    fig_prefix = os.path.join(args.output_dir, "fig_land_") if args.figures else None

    compute_landscape_metrics(
        output_dir=args.output_dir,
        raster_dir=args.raster_dir,
        boundary_shp=args.boundary,
        year_range=[args.start, args.end],
        neighborhood_rule=args.neighborhood,
        figure_prefix=fig_prefix,
    )
