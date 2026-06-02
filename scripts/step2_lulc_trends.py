"""
Step 2: LULC Classification Area Trends
========================================
Reads classified LULC rasters, computes area per class per year,
and calculates inter-annual change rates and transition metrics.

Input:  LULC classification rasters (one per year), study boundary.
Output: LULC area trends CSV, transition matrix CSVs.

Usage:
  python step2_lulc_trends.py --raster-dir D:/data/LULC/ \
      --boundary D:/data/boundary.shp --output-dir D:/output/ \
      --start 2000 --end 2025
"""
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from _utils import clip_raster_to_boundary, rasterize_boundary, ensure_dir


def analyze_lulc_trends(
    raster_dir,
    boundary_shp,
    output_dir,
    raster_pattern="lulc_{year}.tif",
    year_range=None,
    classes=None,
    class_names=None,
    target_crs="EPSG:32649",
    figure_prefix=None,
):
    """
    Compute LULC area trends from classified rasters.

    Parameters
    ----------
    raster_dir : str        Directory with LULC rasters.
    boundary_shp : str      Study boundary shapefile.
    output_dir : str        Output directory.
    raster_pattern : str    Filename pattern with {year} placeholder.
    year_range : list       [start, end] year range.
    classes : dict          {class_name: pixel_value, ...}.
    class_names : list      Ordered class name list.
    target_crs : str        Target CRS.
    figure_prefix : str     Output figure prefix (None to skip).

    Returns
    -------
    str  Path to LULC area CSV.
    """
    if year_range is None:
        year_range = [2000, 2025]
    if classes is None:
        classes = {
            "water": 1, "built_up": 2, "unrestored": 3,
            "recovering": 4, "stable_vegetation": 5
        }
    if class_names is None:
        class_names = list(classes.keys())

    T0 = time.time()
    boundary_gdf = gpd.read_file(boundary_shp)

    years = list(range(year_range[0], year_range[1] + 1))
    raster_dir = Path(raster_dir)

    # Collect area per class per year
    area_data = []
    pixel_area_ha = None

    # Load first raster to determine pixel area
    first_raster = raster_dir / raster_pattern.format(year=years[0])
    if not first_raster.exists():
        # Try alternate naming
        for yr in years:
            trial = raster_dir / raster_pattern.format(year=yr)
            if trial.exists():
                first_raster = trial
                break

    if first_raster.exists():
        with rasterio.open(str(first_raster)) as src:
            pixel_area_ha = abs(src.transform[0] * src.transform[4]) / 10000.0  # m² → ha
            ref_shape = src.shape
            ref_transform = src.transform
            ref_crs = src.crs
            print(f"    Reference raster: {first_raster.name}, "
                  f"pixel={pixel_area_ha:.4f} ha, shape={ref_shape}")

        # Rasterize boundary once
        boundary_proj = boundary_gdf.to_crs(ref_crs)
        roi = rasterize_boundary(boundary_proj, ref_shape, ref_transform)

        for yr in years:
            raster_path = raster_dir / raster_pattern.format(year=yr)
            if not raster_path.exists():
                print(f"    WARNING: {raster_path.name} not found, skipping year {yr}.")
                continue

            print(f"  [{time.time()-T0:.0f}s] Processing year {yr}: {raster_path.name}")

            with rasterio.open(str(raster_path)) as src:
                lulc = src.read(1)
                # Clip to boundary
                lulc[~roi] = 0

            total_px = roi.sum()
            row = {"Year": yr, "area_total_ha": round(total_px * pixel_area_ha, 2)}

            for cls_name in class_names:
                cls_val = classes.get(cls_name, -1)
                cls_px = (lulc == cls_val).sum()
                row[f"area_{cls_name}_ha"] = round(cls_px * pixel_area_ha, 2)
                row[f"area_{cls_name}_pct"] = round(
                    cls_px / total_px * 100, 2) if total_px > 0 else 0

            area_data.append(row)

    else:
        print("    No LULC rasters found. Please check raster_dir and raster_pattern.")
        # Return empty template
        area_data = [{"Year": yr} for yr in years]

    # Save area trends
    ensure_dir(output_dir)
    df_area = pd.DataFrame(area_data)
    area_csv = os.path.join(output_dir, "lulc_area_trends.csv")
    df_area.to_csv(area_csv, index=False, encoding="utf-8-sig")

    # Also save long-format version
    long_rows = []
    for _, row in df_area.iterrows():
        for cls_name in class_names:
            long_rows.append({
                "Year": row["Year"],
                "class": cls_name,
                "area_ha": row.get(f"area_{cls_name}_ha", 0),
                "area_pct": row.get(f"area_{cls_name}_pct", 0),
            })
    df_long = pd.DataFrame(long_rows)
    long_csv = os.path.join(output_dir, "lulc_area_long.csv")
    df_long.to_csv(long_csv, index=False, encoding="utf-8-sig")

    # Compute transition matrix between first and last year
    if years and first_raster.exists():
        yr_first = years[0]
        yr_last = years[-1]
        first_path = raster_dir / raster_pattern.format(year=yr_first)
        last_path = raster_dir / raster_pattern.format(year=yr_last)

        if first_path.exists() and last_path.exists():
            print(f"  [{time.time()-T0:.0f}s] Computing {yr_first}→{yr_last} transition matrix...")
            with rasterio.open(str(first_path)) as src:
                lulc_first = src.read(1)
                lulc_first[~roi] = 0
            with rasterio.open(str(last_path)) as src:
                lulc_last = src.read(1)
                lulc_last[~roi] = 0

            _compute_transition_matrix(lulc_first, lulc_last, class_names, classes,
                                       pixel_area_ha, yr_first, yr_last, output_dir)

    print(f"  DONE [Step 2, {time.time()-T0:.0f}s]")
    return area_csv


def _compute_transition_matrix(lulc_first, lulc_last, class_names, classes,
                                pixel_area_ha, yr_first, yr_last, output_dir):
    """Compute and save LULC transition matrix."""
    n = len(class_names)
    matrix_ha = np.zeros((n, n))
    total = 0

    for i, name_i in enumerate(class_names):
        val_i = classes[name_i]
        mask_i = (lulc_first == val_i)
        for j, name_j in enumerate(class_names):
            val_j = classes[name_j]
            px = (mask_i & (lulc_last == val_j)).sum()
            matrix_ha[i, j] = round(px * pixel_area_ha, 2)
            total += px

    # Build DataFrame
    df_matrix = pd.DataFrame(matrix_ha, index=class_names, columns=class_names)
    df_matrix.index.name = f"From ({yr_first})"
    df_matrix.columns.name = f"To ({yr_last})"

    matrix_csv = os.path.join(output_dir, f"transition_matrix_{yr_first}_{yr_last}.csv")
    df_matrix.to_csv(matrix_csv, encoding="utf-8-sig")
    print(f"    Transition matrix saved: {matrix_csv}")

    # Probability matrix
    row_sums = matrix_ha.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_prob = matrix_ha / row_sums
    df_prob = pd.DataFrame(matrix_prob, index=class_names, columns=class_names)
    df_prob.index.name = f"From ({yr_first})"
    df_prob.columns.name = f"To ({yr_last})"
    prob_csv = os.path.join(output_dir, f"transition_probability_{yr_first}_{yr_last}.csv")
    df_prob.to_csv(prob_csv, encoding="utf-8-sig")
    print(f"    Transition probability saved: {prob_csv}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="LULC Classification Area Trends")
    p.add_argument("--raster-dir", required=True)
    p.add_argument("--boundary", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--pattern", default="lulc_{year}.tif")
    p.add_argument("--start", type=int, default=2000)
    p.add_argument("--end", type=int, default=2025)
    p.add_argument("--crs", default="EPSG:32649")
    p.add_argument("--figures", action="store_true")
    args = p.parse_args()

    fig_prefix = os.path.join(args.output_dir, "fig_lulc_") if args.figures else None

    analyze_lulc_trends(
        raster_dir=args.raster_dir,
        boundary_shp=args.boundary,
        output_dir=args.output_dir,
        raster_pattern=args.pattern,
        year_range=[args.start, args.end],
        target_crs=args.crs,
        figure_prefix=fig_prefix,
    )
