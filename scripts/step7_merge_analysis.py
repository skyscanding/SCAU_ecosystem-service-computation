"""
Step 7: Merge All Analysis Outputs into Unified Analysis Table
================================================================
Combines outputs from all previous steps (LandTrendr disturbance,
LULC areas, landscape metrics, carbon, habitat quality, SDR) into
a single analysis_table.csv for statistical analysis.

Also assigns three ecological restoration phases:
  - Degradation:    2013-2016
  - Transition:     2017-2020
  - Consolidation:  2021-2025

Input:  Multiple CSV files from steps 1-6.
Output: Unified analysis_table.csv.

Usage:
  python step7_merge_analysis.py --input-dir D:/output/ \
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


def merge_analysis_table(
    input_csvs=None,
    output_dir=None,
    output_filename="analysis_table.csv",
    include_columns=None,
    phases=None,
    total_area_ha=None,
):
    """
    Merge all analysis CSVs into a unified analysis table.

    Parameters
    ----------
    input_csvs : dict     {key: csv_path} mapping analysis types to file paths.
                          Keys: lulc, landscape, carbon, hq, sdr, disturbance.
    output_dir : str      Output directory.
    output_filename : str Output CSV filename.
    include_columns : list Columns to include (None = all).
    phases : dict         {phase_name: [start_year, end_year]}.
    total_area_ha : float Total study area in hectares.

    Returns
    -------
    str  Path to merged analysis table CSV.
    """
    T0 = time.time()
    ensure_dir(output_dir)

    if input_csvs is None:
        input_csvs = {}

    if phases is None:
        phases = {
            "Degradation": [2013, 2016],
            "Transition": [2017, 2020],
            "Consolidation": [2021, 2025],
        }

    # Load each input CSV
    dfs = {}

    # LULC area
    lulc_path = input_csvs.get("lulc", os.path.join(output_dir, "lulc_area_trends.csv"))
    if os.path.exists(lulc_path):
        dfs["lulc"] = pd.read_csv(lulc_path)
        print(f"    Loaded LULC: {len(dfs['lulc'])} rows, {list(dfs['lulc'].columns[:5])}...")
    else:
        print(f"    WARNING: LULC CSV not found at {lulc_path}")

    # Landscape metrics
    land_path = input_csvs.get("landscape", os.path.join(output_dir, "landscape_metrics.csv"))
    if os.path.exists(land_path):
        dfs["landscape"] = pd.read_csv(land_path)
        print(f"    Loaded landscape metrics: {len(dfs['landscape'])} rows")
    else:
        print(f"    WARNING: Landscape metrics CSV not found at {land_path}")

    # Carbon
    carbon_path = input_csvs.get("carbon", os.path.join(output_dir, "carbon_summary.csv"))
    if os.path.exists(carbon_path):
        dfs["carbon"] = pd.read_csv(carbon_path)
        print(f"    Loaded carbon: {len(dfs['carbon'])} rows")
    else:
        print(f"    WARNING: Carbon CSV not found at {carbon_path}")

    # Habitat quality
    hq_path = input_csvs.get("hq", os.path.join(output_dir, "habitat_quality_summary.csv"))
    if os.path.exists(hq_path):
        dfs["hq"] = pd.read_csv(hq_path)
        print(f"    Loaded habitat quality: {len(dfs['hq'])} rows")
    else:
        print(f"    WARNING: Habitat quality CSV not found at {hq_path}")

    # SDR
    sdr_path = input_csvs.get("sdr", os.path.join(output_dir, "sdr_summary.csv"))
    if os.path.exists(sdr_path):
        dfs["sdr"] = pd.read_csv(sdr_path)
        print(f"    Loaded SDR: {len(dfs['sdr'])} rows")
    else:
        print(f"    WARNING: SDR CSV not found at {sdr_path}")

    # LandTrendr disturbance (Nanling convention: summary_statistics.csv)
    dist_path = input_csvs.get("disturbance", os.path.join(output_dir, "summary_statistics.csv"))
    if os.path.exists(dist_path):
        dfs["disturbance"] = pd.read_csv(dist_path)
        print(f"    Loaded disturbance: {len(dfs['disturbance'])} rows")
        print(f"    Columns: {list(dfs['disturbance'].columns)}")
    else:
        # Fallback: old naming
        alt_path = os.path.join(output_dir, "landtrendr_summary.csv")
        if os.path.exists(alt_path):
            dfs["disturbance"] = pd.read_csv(alt_path)
            print(f"    Loaded disturbance (legacy): {len(dfs['disturbance'])} rows")
        else:
            print(f"    WARNING: Disturbance CSV not found at {dist_path} or {alt_path}")

    # If no data loaded, return template
    if not dfs:
        print("    No input CSVs found. Creating empty template.")
        template_cols = include_columns or ["Year"]
        template = pd.DataFrame(columns=template_cols)
        template_path = os.path.join(output_dir, output_filename)
        template.to_csv(template_path, index=False, encoding="utf-8-sig")
        return template_path

    # Start merging from the base table (LULC or first available)
    base_key = "lulc" if "lulc" in dfs else list(dfs.keys())[0]
    merged = dfs[base_key].copy()

    # Merge all other DataFrames on Year
    for key, df in dfs.items():
        if key == base_key:
            continue
        if "Year" in df.columns:
            merged = merged.merge(df, on="Year", how="outer", suffixes=("", f"_{key}"))
            # Remove duplicate columns
            merged = merged.loc[:, ~merged.columns.str.endswith(f"_{key}")]
        else:
            print(f"    WARNING: {key} DataFrame has no 'Year' column, skipping merge.")

    # Assign phase labels
    merged["phase"] = merged["Year"].apply(lambda y: _assign_phase(y, phases))

    # Reorder columns: Year first, then LULC, landscape, carbon, hq, sdr, disturbance, phase last
    col_order = ["Year"]
    for group in ["area_", "NP", "PD", "LPI", "ED", "LSI", "SHDI", "CONTAG", "MESH",
                   "carbon_", "habitat_", "usle_", "sed_", "r_factor", "disturb_"]:
        matching = [c for c in merged.columns if c.startswith(group) or group.rstrip("_") in c.lower()]
        for c in matching:
            if c not in col_order and c != "Year":
                col_order.append(c)
    # Add any remaining columns
    for c in merged.columns:
        if c not in col_order:
            col_order.append(c)

    col_order = [c for c in col_order if c in merged.columns]
    merged = merged[col_order]

    # Filter columns if specified
    if include_columns:
        keep = [c for c in include_columns if c in merged.columns]
        merged = merged[keep]

    # Drop rows with NaN Year
    merged = merged.dropna(subset=["Year"])
    merged["Year"] = merged["Year"].astype(int)
    merged = merged.sort_values("Year").reset_index(drop=True)

    # Save
    output_path = os.path.join(output_dir, output_filename)
    merged.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"    Merged table saved: {output_path}")
    print(f"    {len(merged)} years, {len(merged.columns)} columns")
    print(f"    Phase distribution: {merged['phase'].value_counts().to_dict()}")

    print(f"  DONE [Step 7, {time.time()-T0:.0f}s]")
    return output_path


def _assign_phase(year, phases):
    """Assign ecological restoration phase label to a year."""
    for phase_name, (start, end) in phases.items():
        if start <= year <= end:
            return phase_name
    return "Other"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Merge Analysis Outputs")
    p.add_argument("--input-dir", required=True, help="Directory with all step CSVs")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--output", default="analysis_table.csv")
    args = p.parse_args()

    # Auto-detect input CSVs
    input_csvs = {}
    idir = args.input_dir
    for key, fname in [
        ("lulc", "lulc_area_trends.csv"),
        ("landscape", "landscape_metrics.csv"),
        ("carbon", "carbon_summary.csv"),
        ("hq", "habitat_quality_summary.csv"),
        ("sdr", "sdr_summary.csv"),
        ("disturbance", "landtrendr_summary.csv"),
    ]:
        path = os.path.join(idir, fname)
        if os.path.exists(path):
            input_csvs[key] = path

    merge_analysis_table(
        input_csvs=input_csvs,
        output_dir=args.output_dir,
        output_filename=args.output,
    )
