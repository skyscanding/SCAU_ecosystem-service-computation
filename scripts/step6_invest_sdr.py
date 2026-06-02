"""
Step 6: InVEST SDR Soil Erosion Analysis
===========================================
Estimates soil erosion (USLE) and sediment delivery using InVEST SDR
model logic. Inputs include DEM, rainfall erosivity (R factor), soil
erodibility (K factor), and LULC-based C and P factors.

Input:  LULC area CSV, R factor data, C/P factor lookup tables.
Output: Soil erosion & sediment export summary CSV.

Usage:
  python step6_invest_sdr.py --lulc-csv D:/output/lulc_area_trends.csv \
      --output-dir D:/output/
"""
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
from _utils import ensure_dir


def analyze_sdr(
    lulc_csv,
    output_dir,
    dem_path=None,
    erosivity_path=None,
    erodibility_path=None,
    c_factor=None,
    p_factor=None,
    r_factor_source="annual",
    figure_prefix=None,
):
    """
    Estimate soil erosion and sediment delivery using SDR model.

    Simplified SDR:
      USLE_i = R * K * LS * C_i * P_i
      Sed_export = USLE_i * SDR_i

    This implementation uses area-weighted USLE based on LULC area
    proportions and summary R/K/LS factors.

    Parameters
    ----------
    lulc_csv : str             LULC area trends CSV.
    output_dir : str           Output directory.
    dem_path : str             Path to DEM raster.
    erosivity_path : str       Path to R-factor raster.
    erodibility_path : str     Path to K-factor raster.
    c_factor : dict            {class_name: C factor value}.
    p_factor : dict            {class_name: P factor value}.
    r_factor_source : str      'annual' = per-year R values, 'mean' = constant.
    figure_prefix : str        Output figure prefix.

    Returns
    -------
    str  Path to SDR summary CSV.
    """
    T0 = time.time()

    # Default C and P factors (based on Dabaoshan case)
    if c_factor is None:
        c_factor = {
            "water": 0.0,
            "built_up": 0.01,
            "unrestored": 0.35,
            "recovering": 0.08,
            "stable_vegetation": 0.003,
        }
    if p_factor is None:
        p_factor = {
            "water": 0.0,
            "built_up": 1.0,
            "unrestored": 1.0,
            "recovering": 1.0,
            "stable_vegetation": 1.0,
        }

    # Default mean R, K, LS factors (summary-level proxies)
    # Full SDR requires per-pixel computation; these are area-weighted summaries
    R_mean = 5000.0      # MJ·mm/(ha·h·yr) - typical for subtropical southern China
    K_mean = 0.025       # t·ha·h/(ha·MJ·mm)
    LS_mean = 5.0        # dimensionless (from DEM slope/length)
    SDR_mean = 0.15      # Sediment Delivery Ratio (dimensionless)

    # If rasters provided, compute summary statistics
    if dem_path and os.path.exists(dem_path):
        import rasterio
        with rasterio.open(dem_path) as src:
            dem = src.read(1)
            dem_valid = dem[dem > 0]
            print(f"    DEM loaded: {dem.shape}, mean elevation={dem_valid.mean():.0f}m")
            # LS factor estimation from DEM (simplified)
            # LS = (flow_accumulation * cell_size / 22.13)^0.4 * (sin(slope)/0.0896)^1.3
            LS_mean = 5.0  # placeholder; full computation requires flow direction

    if erosivity_path and os.path.exists(erosivity_path):
        import rasterio
        with rasterio.open(erosivity_path) as src:
            r = src.read(1)
            r_valid = r[r > 0]
            if r_valid.size > 0:
                R_mean = float(np.mean(r_valid))
                print(f"    R factor mean: {R_mean:.1f} MJ·mm/(ha·h·yr)")

    if erodibility_path and os.path.exists(erodibility_path):
        import rasterio
        with rasterio.open(erodibility_path) as src:
            k = src.read(1)
            k_valid = k[k > 0]
            if k_valid.size > 0:
                K_mean = float(np.mean(k_valid))
                print(f"    K factor mean: {K_mean:.4f} t·ha·h/(ha·MJ·mm)")

    # Load LULC areas
    if lulc_csv and os.path.exists(lulc_csv):
        df_lulc = pd.read_csv(lulc_csv)
    else:
        print("    WARNING: No LULC CSV. Creating empty template.")
        df_lulc = pd.DataFrame({"Year": list(range(2000, 2026))})

    # Annual R factor variation (approximate for Dabaoshan, Guangdong)
    annual_R_factors = {
        2013: 4800, 2014: 5200, 2015: 5500, 2016: 5800,
        2017: 4500, 2018: 5100, 2019: 4900, 2020: 5300,
        2021: 4600, 2022: 5000, 2023: 4700, 2024: 4400, 2025: 5000,
    }

    # Compute annual USLE and sediment export
    sdr_rows = []
    for _, row in df_lulc.iterrows():
        yr = int(row.get("Year", row.name))
        R_yr = annual_R_factors.get(yr, R_mean) if r_factor_source == "annual" else R_mean

        total_usle = 0
        total_sed = 0
        total_area = 0

        for cls_name in c_factor.keys():
            area_col = f"area_{cls_name}_ha"
            if area_col not in row:
                continue
            area_ha = row[area_col]
            total_area += area_ha

            C = c_factor.get(cls_name, 0.01)
            P = p_factor.get(cls_name, 1.0)

            # USLE = R * K * LS * C * P  (t/ha/yr)
            usle_per_ha = R_yr * K_mean * LS_mean * C * P
            usle_total = usle_per_ha * area_ha  # t/yr
            sed_total = usle_total * SDR_mean    # t/yr

            total_usle += usle_total
            total_sed += sed_total

        sdr_rows.append({
            "Year": yr,
            "usle_total_t_yr": round(total_usle, 2),
            "sed_export_t_yr": round(total_sed, 2),
            "r_factor": round(R_yr, 1),
            "r_factor_source": r_factor_source,
            "area_total_ha": round(total_area, 2),
        })

    df_sdr = pd.DataFrame(sdr_rows)
    ensure_dir(output_dir)
    sdr_csv = os.path.join(output_dir, "sdr_summary.csv")
    df_sdr.to_csv(sdr_csv, index=False, encoding="utf-8-sig")
    print(f"    SDR summary saved: {sdr_csv}")
    print(f"    USLE range: {df_sdr['usle_total_t_yr'].min():.0f} - {df_sdr['usle_total_t_yr'].max():.0f} t/yr")
    print(f"    Sed export range: {df_sdr['sed_export_t_yr'].min():.0f} - {df_sdr['sed_export_t_yr'].max():.0f} t/yr")

    print(f"  DONE [Step 6, {time.time()-T0:.0f}s]")
    return sdr_csv


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="InVEST SDR Soil Erosion Analysis")
    p.add_argument("--lulc-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dem", default=None)
    p.add_argument("--erosivity", default=None)
    p.add_argument("--erodibility", default=None)
    p.add_argument("--r-source", default="annual", choices=["annual", "mean"])
    p.add_argument("--figures", action="store_true")
    args = p.parse_args()

    fig_prefix = os.path.join(args.output_dir, "fig_sdr_") if args.figures else None

    analyze_sdr(
        lulc_csv=args.lulc_csv,
        output_dir=args.output_dir,
        dem_path=args.dem,
        erosivity_path=args.erosivity,
        erodibility_path=args.erodibility,
        r_factor_source=args.r_source,
        figure_prefix=fig_prefix,
    )
