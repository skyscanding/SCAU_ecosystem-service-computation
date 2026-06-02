"""
Step 4: InVEST Carbon Storage Estimation
===========================================
Estimates total carbon storage (above-ground, below-ground, soil, dead
organic matter) per year based on LULC areas and carbon pool densities.

Input:  LULC area CSV, carbon pool parameters (per LULC class).
Output: Carbon storage summary CSV.

Usage:
  python step4_invest_carbon.py --lulc-csv D:/output/lulc_area_trends.csv \
      --output-dir D:/output/
"""
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
from pathlib import Path
from _utils import ensure_dir


def estimate_carbon(
    lulc_csv,
    output_dir,
    carbon_pools_csv=None,
    carbon_pools_dict=None,
    figure_prefix=None,
):
    """
    Estimate carbon storage from LULC areas and carbon pool densities.

    Parameters
    ----------
    lulc_csv : str            Path to LULC area trends CSV.
    output_dir : str          Output directory.
    carbon_pools_csv : str    Path to InVEST carbon pools CSV (optional).
    carbon_pools_dict : dict  {class_name: {c_above, c_below, c_soil, c_dead}}.
    figure_prefix : str       Output figure prefix (None to skip).

    Returns
    -------
    str  Path to carbon summary CSV.
    """
    T0 = time.time()

    # Load carbon pools
    pools = {}
    if carbon_pools_csv and os.path.exists(carbon_pools_csv):
        df_pools = pd.read_csv(carbon_pools_csv)
        for _, row in df_pools.iterrows():
            lulc = row.get("lucode") or row.get("class_name") or row.get("LULC")
            pools[str(lulc).lower()] = {
                "c_above": float(row.get("c_above", 0)),
                "c_below": float(row.get("c_below", 0)),
                "c_soil": float(row.get("c_soil", 0)),
                "c_dead": float(row.get("c_dead", 0)),
            }
    elif carbon_pools_dict:
        pools = carbon_pools_dict
    else:
        # Default pools for Dabaoshan case study
        pools = {
            "water": {"c_above": 0, "c_below": 0, "c_soil": 0, "c_dead": 0},
            "built_up": {"c_above": 0, "c_below": 0, "c_soil": 0, "c_dead": 0},
            "unrestored": {"c_above": 2.5, "c_below": 1.2, "c_soil": 45, "c_dead": 1.5},
            "recovering": {"c_above": 15, "c_below": 7, "c_soil": 60, "c_dead": 3},
            "stable_vegetation": {"c_above": 85, "c_below": 40, "c_soil": 95, "c_dead": 8},
        }

    print(f"    Carbon pools loaded: {list(pools.keys())}")

    # Load LULC areas
    if lulc_csv and os.path.exists(lulc_csv):
        df_lulc = pd.read_csv(lulc_csv)
    else:
        print("    WARNING: No LULC CSV. Creating empty template.")
        df_lulc = pd.DataFrame({"Year": list(range(2000, 2026))})

    carbon_rows = []
    for _, row in df_lulc.iterrows():
        yr = row.get("Year", row.name)
        total_carbon = 0
        total_area = 0

        for cls_name, pool in pools.items():
            area_col = f"area_{cls_name}_ha"
            if area_col not in row:
                continue
            area_ha = row[area_col]
            total_area += area_ha

            c_density = pool["c_above"] + pool["c_below"] + pool["c_soil"] + pool["c_dead"]
            total_carbon += area_ha * c_density  # Mg/ha * ha = Mg

        carbon_rows.append({
            "Year": int(yr),
            "carbon_total_Mg": round(total_carbon, 2),
            "carbon_density_Mg_ha": round(total_carbon / total_area, 6) if total_area > 0 else 0,
            "area_total_ha": round(total_area, 2),
        })

    df_carbon = pd.DataFrame(carbon_rows)
    ensure_dir(output_dir)
    carbon_csv = os.path.join(output_dir, "carbon_summary.csv")
    df_carbon.to_csv(carbon_csv, index=False, encoding="utf-8-sig")
    print(f"    Carbon summary saved: {carbon_csv}")
    print(f"    Range: {df_carbon['carbon_total_Mg'].min():.0f} - {df_carbon['carbon_total_Mg'].max():.0f} Mg")

    print(f"  DONE [Step 4, {time.time()-T0:.0f}s]")
    return carbon_csv


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="InVEST Carbon Storage Estimation")
    p.add_argument("--lulc-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--carbon-pools-csv", default=None)
    p.add_argument("--figures", action="store_true")
    args = p.parse_args()

    fig_prefix = os.path.join(args.output_dir, "fig_carbon_") if args.figures else None

    estimate_carbon(
        lulc_csv=args.lulc_csv,
        output_dir=args.output_dir,
        carbon_pools_csv=args.carbon_pools_csv,
        figure_prefix=fig_prefix,
    )
