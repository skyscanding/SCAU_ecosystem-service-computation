"""
LandTrendr + Ecosystem Services Coupling Analysis, Master Orchestrator
======================================================================
Runs the full multi-step pipeline for disturbance detection analysis,
LULC classification trends, landscape metrics, InVEST ecosystem services,
and statistical coupling analysis. All parameters are read from a JSON
config file (see config_template.json).

Steps:
  1. LandTrendr disturbance raster analysis
  2. LULC classification area trends
  3. Landscape metrics (PyLandStats)
  4. InVEST Carbon storage
  5. InVEST Habitat Quality
  6. InVEST SDR soil erosion
  7. Merge all outputs into analysis_table.csv
  8. Correlation analysis (Pearson + Spearman + partial)
  9. OLS Regression models
  10. SEM Path analysis + Kruskal-Wallis phase comparison

Usage:
  python master_pipeline.py config.json
  python master_pipeline.py config.json --steps dist,merge,corr,regress,path
"""
import os
import sys
import json
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import numpy as np
import pandas as pd
import geopandas as gpd
from _utils import ensure_dir


def run_pipeline(config_path, selected_steps=None):
    """Execute the LandTrendr + Ecosystem Services coupling pipeline from a JSON config."""

    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    T0 = time.time()

    output_dir = ensure_dir(cfg["output_dir"])
    crs = cfg.get("crs", "EPSG:32649")
    fm = cfg.get("field_mapping", {})

    all_steps = ["dist", "class", "land", "carbon", "hq", "sdr", "merge", "corr", "regress", "path"]
    if selected_steps:
        steps = [s.strip() for s in selected_steps.split(",")]
    else:
        steps = cfg.get("steps", all_steps)

    results = {}
    base = os.path.dirname(os.path.abspath(config_path))

    # ── Step 1: LandTrendr Disturbance Detection ──
    if "dist" in steps:
        print(f"\n{'='*60}\nSTEP 1: LandTrendr Disturbance Raster Analysis\n{'='*60}")
        from step1_landtrendr_disturbance import analyze_landtrendr

        lt_cfg = cfg.get("landtrendr", {})
        study_cfg = cfg.get("study_area", {})
        boundary_shp = study_cfg.get("boundary_shp", "")
        if not os.path.isabs(boundary_shp):
            boundary_shp = os.path.join(base, boundary_shp)

        raster_dir = lt_cfg.get("raster_dir", "")
        if not os.path.isabs(raster_dir):
            raster_dir = os.path.join(base, raster_dir)

        results["disturbance_csv"] = analyze_landtrendr(
            raster_dir=raster_dir,
            boundary_shp=boundary_shp,
            output_dir=output_dir,
            yod_file=lt_cfg.get("yod_raster", "yod_2009_2024.tif"),
            mag_file=lt_cfg.get("mag_raster", "mag_2009_2024.tif"),
            dur_file=lt_cfg.get("dur_raster", "dur_2009_2024.tif"),
            mpy_file=lt_cfg.get("mpy_raster", "magperyear_2009_2024.tif"),
            nodata_val=lt_cfg.get("nodata_val", 0),
            yod_range=lt_cfg.get("yod_range", [2010, 2024]),
            pixel_area_ha=study_cfg.get("pixel_area_ha", 0.09),
            intensity_bins=lt_cfg.get("intensity_bins", None),
            severity_bins=lt_cfg.get("severity_bins", None),
            generate_figures=lt_cfg.get("generate_figures", True),
        )
        print(f"  Disturbance summary saved to: {results['disturbance_csv']}")

    # ── Step 2: LULC Classification Trends ──
    if "class" in steps:
        print(f"\n{'='*60}\nSTEP 2: LULC Classification Area Trends\n{'='*60}")
        from step2_lulc_trends import analyze_lulc_trends

        lulc_cfg = cfg.get("lulc", {})
        study_cfg = cfg.get("study_area", {})
        boundary_shp = study_cfg.get("boundary_shp", "")
        if not os.path.isabs(boundary_shp):
            boundary_shp = os.path.join(base, boundary_shp)

        raster_dir = lulc_cfg.get("raster_dir", "")
        if not os.path.isabs(raster_dir):
            raster_dir = os.path.join(base, raster_dir)

        results["lulc_csv"] = analyze_lulc_trends(
            raster_dir=raster_dir,
            boundary_shp=boundary_shp,
            output_dir=output_dir,
            raster_pattern=lulc_cfg.get("raster_pattern", "lulc_{year}.tif"),
            year_range=lulc_cfg.get("year_range", [2000, 2025]),
            classes=lulc_cfg.get("classes", {}),
            class_names=lulc_cfg.get("class_names", []),
            target_crs=crs,
            figure_prefix=os.path.join(output_dir, "fig_lulc_"),
        )
        print(f"  LULC trends saved to: {results['lulc_csv']}")

    # ── Step 3: Landscape Metrics ──
    if "land" in steps:
        print(f"\n{'='*60}\nSTEP 3: Landscape Metrics (PyLandStats)\n{'='*60}")
        from step3_landscape_metrics import compute_landscape_metrics

        land_cfg = cfg.get("landscape", {})
        study_cfg = cfg.get("study_area", {})

        results["landscape_csv"] = compute_landscape_metrics(
            lulc_csv=results.get("lulc_csv"),
            output_dir=output_dir,
            landscape_metrics=land_cfg.get("landscape_metrics", ["np", "pd", "lpi", "ed", "lsi", "shdi", "contag", "mesh"]),
            class_metrics=land_cfg.get("class_metrics", ["ca", "pland", "np", "pd", "lpi", "ed", "lsi"]),
            metrics_level=land_cfg.get("metrics_level", ["landscape", "class"]),
            neighborhood_rule=land_cfg.get("neighborhood_rule", "8"),
            raster_dir=None,  # If using raster inputs
            boundary_shp=study_cfg.get("boundary_shp"),
            year_range=cfg.get("lulc", {}).get("year_range", [2000, 2025]),
            figure_prefix=os.path.join(output_dir, "fig_land_"),
        )
        print(f"  Landscape metrics saved to: {results['landscape_csv']}")

    # ── Step 4: InVEST Carbon Storage ──
    if "carbon" in steps:
        print(f"\n{'='*60}\nSTEP 4: InVEST Carbon Storage Estimation\n{'='*60}")
        from step4_invest_carbon import estimate_carbon

        carbon_cfg = cfg.get("invest_carbon", {})

        results["carbon_csv"] = estimate_carbon(
            lulc_csv=results.get("lulc_csv"),
            output_dir=output_dir,
            carbon_pools_csv=carbon_cfg.get("carbon_pools_csv"),
            carbon_pools_dict=carbon_cfg.get("carbon_pools", {}),
            figure_prefix=os.path.join(output_dir, "fig_carbon_"),
        )
        print(f"  Carbon storage saved to: {results['carbon_csv']}")

    # ── Step 5: InVEST Habitat Quality ──
    if "hq" in steps:
        print(f"\n{'='*60}\nSTEP 5: InVEST Habitat Quality Assessment\n{'='*60}")
        from step5_invest_habitat_quality import assess_habitat_quality

        hq_cfg = cfg.get("invest_habitat_quality", {})

        results["hq_csv"] = assess_habitat_quality(
            lulc_csv=results.get("lulc_csv"),
            output_dir=output_dir,
            half_saturation=hq_cfg.get("half_saturation_constant", 0.5),
            threats=hq_cfg.get("threats", {}),
            sensitivity_csv=hq_cfg.get("sensitivity_csv"),
            habitat_scores=hq_cfg.get("habitat_scores", {}),
            figure_prefix=os.path.join(output_dir, "fig_hq_"),
        )
        print(f"  Habitat quality saved to: {results['hq_csv']}")

    # ── Step 6: InVEST SDR Soil Erosion ──
    if "sdr" in steps:
        print(f"\n{'='*60}\nSTEP 6: InVEST SDR Soil Erosion Analysis\n{'='*60}")
        from step6_invest_sdr import analyze_sdr

        sdr_cfg = cfg.get("invest_sdr", {})

        results["sdr_csv"] = analyze_sdr(
            lulc_csv=results.get("lulc_csv"),
            output_dir=output_dir,
            dem_path=sdr_cfg.get("dem_path"),
            erosivity_path=sdr_cfg.get("erosivity_path"),
            erodibility_path=sdr_cfg.get("erodibility_path"),
            c_factor=sdr_cfg.get("lulc_c_factor", {}),
            p_factor=sdr_cfg.get("lulc_p_factor", {}),
            r_factor_source=sdr_cfg.get("r_factor_source", "annual"),
            figure_prefix=os.path.join(output_dir, "fig_sdr_"),
        )
        print(f"  SDR results saved to: {results['sdr_csv']}")

    # ── Step 7: Merge Analysis Table ──
    if "merge" in steps:
        print(f"\n{'='*60}\nSTEP 7: Merge All Outputs into Analysis Table\n{'='*60}")
        from step7_merge_analysis import merge_analysis_table
        import pickle

        merge_cfg = cfg.get("merge", {})
        phase_cfg = cfg.get("phase_analysis", {})
        study_cfg = cfg.get("study_area", {})

        # Gather all input CSVs
        input_csvs = {}
        for key, res_key in [
            ("lulc", "lulc_csv"),
            ("landscape", "landscape_csv"),
            ("carbon", "carbon_csv"),
            ("hq", "hq_csv"),
            ("sdr", "sdr_csv"),
            ("disturbance", "disturbance_csv"),
        ]:
            if res_key in results:
                input_csvs[key] = results[res_key]
            else:
                # Try default paths (Nanling convention for disturbance)
                if key == "disturbance":
                    default_path = os.path.join(output_dir, "summary_statistics.csv")
                else:
                    default_path = os.path.join(output_dir, f"{key}_summary.csv")
                if os.path.exists(default_path):
                    input_csvs[key] = default_path

        results["analysis_csv"] = merge_analysis_table(
            input_csvs=input_csvs,
            output_dir=output_dir,
            output_filename=merge_cfg.get("output_csv", "analysis_table.csv"),
            include_columns=merge_cfg.get("include_columns"),
            phases=phase_cfg.get("phases", {}),
            total_area_ha=study_cfg.get("pixel_area_ha", 0.09),
        )
        print(f"  Analysis table saved to: {results['analysis_csv']}")

    # ── Step 8: Correlation Analysis ──
    if "corr" in steps:
        print(f"\n{'='*60}\nSTEP 8: Correlation & Partial Correlation Analysis\n{'='*60}")
        from step8_correlation import correlation_analysis

        corr_cfg = cfg.get("correlation", {})

        analysis_csv = corr_cfg.get("analysis_csv") or results.get("analysis_csv")
        if not analysis_csv:
            analysis_csv = os.path.join(output_dir, "analysis_table.csv")
        if not os.path.isabs(analysis_csv):
            analysis_csv = os.path.join(base, analysis_csv)

        correlation_analysis(
            analysis_csv=analysis_csv,
            output_dir=output_dir,
            methods=corr_cfg.get("methods", ["pearson", "spearman"]),
            alpha=corr_cfg.get("alpha", 0.05),
            fdr_method=corr_cfg.get("fdr_method", "fdr_bh"),
            partial_control_vars=corr_cfg.get("partial_control_vars"),
            partial_test_vars=corr_cfg.get("partial_test_vars"),
            analysis_years=corr_cfg.get("analysis_years", [2013, 2025]),
            figure_prefix=os.path.join(output_dir, "fig_corr_"),
        )
        print(f"  Correlation results saved to: {os.path.join(output_dir, 'Statistics')}")

    # ── Step 9: Regression Models ──
    if "regress" in steps:
        print(f"\n{'='*60}\nSTEP 9: OLS Multiple Regression Modeling\n{'='*60}")
        from step9_regression import regression_models

        reg_cfg = cfg.get("regression", {})

        analysis_csv = results.get("analysis_csv")
        if not analysis_csv:
            analysis_csv = os.path.join(output_dir, "analysis_table.csv")

        regression_models(
            analysis_csv=analysis_csv,
            output_dir=output_dir,
            models=reg_cfg.get("models", {}),
            figure_prefix=os.path.join(output_dir, "fig_reg_"),
        )
        print(f"  Regression results saved to: {os.path.join(output_dir, 'Statistics')}")

    # ── Step 10: Path Analysis + Phase Comparison ──
    if "path" in steps:
        print(f"\n{'='*60}\nSTEP 10: SEM Path Analysis & Kruskal-Wallis Phase Comparison\n{'='*60}")
        from step10_path_analysis import path_analysis_and_phase_comparison

        path_cfg = cfg.get("path_analysis", {})
        phase_cfg = cfg.get("phase_analysis", {})

        analysis_csv = results.get("analysis_csv")
        if not analysis_csv:
            analysis_csv = os.path.join(output_dir, "analysis_table.csv")

        path_analysis_and_phase_comparison(
            analysis_csv=analysis_csv,
            output_dir=output_dir,
            model_spec=path_cfg.get("model_spec", ""),
            estimator=path_cfg.get("estimator", "ML"),
            phases=phase_cfg.get("phases", {}),
            phase_test=phase_cfg.get("test", "kruskal_wallis"),
            phase_vars=phase_cfg.get("variables", []),
            figure_prefix=os.path.join(output_dir, "fig_path_"),
        )
        print(f"  Path analysis + phase results saved to: {os.path.join(output_dir, 'Statistics')}")

    elapsed = time.time() - T0
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE [{elapsed:.0f}s]")
    print(f"  Output directory: {output_dir}")
    print(f"  Analysis table: {results.get('analysis_csv', os.path.join(output_dir, 'analysis_table.csv'))}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="LandTrendr + Ecosystem Services Coupling Pipeline")
    p.add_argument("config", help="JSON configuration file")
    p.add_argument("--steps", help="Comma-separated steps to run (default: all from config)")
    args = p.parse_args()
    run_pipeline(args.config, args.steps)
