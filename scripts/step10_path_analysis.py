"""
Step 10: SEM Path Analysis & Kruskal-Wallis Phase Comparison
==============================================================
Path Analysis (Structural Equation Modeling via semopy) to test
hypothesized causal pathways: disturbance → landscape pattern →
ecosystem services. Also computes Kruskal-Wallis H tests for
three-phase comparison.

Hypothesis:
  MAG (disturbance magnitude) → SHDI/MESH (landscape pattern) → carbon/HQ/erosion

Input:  analysis_table.csv.
Output: Path analysis coefficient estimates, KW test results, phase statistics.

Usage:
  python step10_path_analysis.py --analysis-csv D:/output/analysis_table.csv \
      --output-dir D:/output/
"""
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from _utils import ensure_dir


def path_analysis_and_phase_comparison(
    analysis_csv,
    output_dir,
    model_spec=None,
    estimator="ML",
    phases=None,
    phase_test="kruskal_wallis",
    phase_vars=None,
    figure_prefix=None,
):
    """
    SEM path analysis + Kruskal-Wallis phase comparison.

    Parameters
    ----------
    analysis_csv : str     Path to analysis_table.csv.
    output_dir : str       Output directory.
    model_spec : str       semopy model specification string.
    estimator : str        Estimator for semopy ('ML', 'ULS', etc.).
    phases : dict          {phase_name: [start_year, end_year]}.
    phase_test : str       'kruskal_wallis' only for now.
    phase_vars : list      Variables to test across phases.
    figure_prefix : str    Output figure prefix.

    Returns
    -------
    dict  Paths to output files.
    """
    if phases is None:
        phases = {
            "Degradation": [2013, 2016],
            "Transition": [2017, 2020],
            "Consolidation": [2021, 2025],
        }
    if phase_vars is None:
        phase_vars = [
            "Area (ha)", "Mean MAG",
            "SHDI", "NP", "MESH",
            "carbon_total_Mg", "habitat_quality_mean", "sed_export_t_yr",
        ]

    T0 = time.time()
    stats_dir = ensure_dir(os.path.join(output_dir, "Statistics"))

    df = pd.read_csv(analysis_csv)
    print(f"    Loaded {len(df)} rows, {len(df.columns)} columns")

    results = {}

    # ── Phase Statistics ──
    print(f"\n    Phase comparison ({phase_test}):")
    phase_data = []

    for phase_name, (y1, y2) in phases.items():
        phase_df = df[(df["Year"] >= y1) & (df["Year"] <= y2)]
        row = {"phase": phase_name, "n_years": len(phase_df), "years": f"{y1}-{y2}"}

        for var in phase_vars:
            if var in phase_df.columns:
                vals = phase_df[var].dropna()
                row[f"{var}_mean"] = round(vals.mean(), 4)
                row[f"{var}_std"] = round(vals.std(), 4)
                row[f"{var}_median"] = round(vals.median(), 4)
                row[f"{var}_min"] = round(vals.min(), 4)
                row[f"{var}_max"] = round(vals.max(), 4)

        phase_data.append(row)

    df_phase = pd.DataFrame(phase_data)
    phase_csv = os.path.join(stats_dir, "phase_summary.csv")
    df_phase.to_csv(phase_csv, index=False, encoding="utf-8-sig")
    results["phase_summary_csv"] = phase_csv
    print(f"    Phase summary saved: {phase_csv}")

    # ── Kruskal-Wallis Tests ──
    kw_rows = []
    for var in phase_vars:
        if var not in df.columns:
            continue

        groups = []
        group_labels = []
        for phase_name, (y1, y2) in phases.items():
            vals = df[(df["Year"] >= y1) & (df["Year"] <= y2)][var].dropna()
            if len(vals) > 0:
                groups.append(vals.values)
                group_labels.append(phase_name)

        if len(groups) >= 2:
            try:
                h_stat, p_val = sp_stats.kruskal(*groups)
                kw_rows.append({
                    "variable": var,
                    "H_statistic": round(h_stat, 4),
                    "p_value": round(p_val, 6),
                    "significant": p_val < 0.05,
                    "phases_tested": ", ".join(group_labels),
                    "n_groups": len(groups),
                })
                stars = " *" if p_val < 0.05 else ""
                print(f"      {var}: H = {h_stat:.2f}, p = {p_val:.4f}{stars}")
            except Exception as e:
                print(f"      {var}: ERROR - {e}")

    df_kw = pd.DataFrame(kw_rows)
    kw_csv = os.path.join(stats_dir, "phase_kruskal.csv")
    df_kw.to_csv(kw_csv, index=False, encoding="utf-8-sig")
    results["kruskal_csv"] = kw_csv

    # ── Path Analysis (semopy) ──
    print(f"\n    Path analysis:")

    if model_spec is None:
        model_spec = """
        # Direct effects: landscape pattern ← disturbance
        SHDI ~ Mean MAG
        MESH ~ Mean MAG
        CONTAG ~ Mean MAG

        # Direct effects: ecosystem services ← landscape pattern + disturbance
        carbon_total_Mg ~ SHDI + MESH + Mean MAG
        habitat_quality_mean ~ SHDI + CONTAG + Mean MAG
        sed_export_t_yr ~ SHDI + Area (ha) + Mean MAG
        """

    try:
        import semopy
        print(f"    semopy version: {semopy.__version__}")

        # Clean model spec
        spec_lines = [line.strip() for line in model_spec.strip().split("\n")
                      if line.strip() and not line.strip().startswith("#")]
        spec = "\n".join(spec_lines)
        print(f"    Model specification:")
        for line in spec_lines:
            print(f"      {line}")

        # Prepare data
        model_vars = set()
        for line in spec_lines:
            parts = line.replace("~", " ").split()
            for p in parts:
                if p not in ["~"] and not p.startswith("#"):
                    model_vars.add(p)

        model_vars = [v for v in model_vars if v in df.columns]
        model_df = df[model_vars].dropna()
        print(f"    N = {len(model_df)} observations with complete data")

        if len(model_df) < 5:
            print("    WARNING: Too few observations for SEM. Skipping path analysis.")
        else:
            # Build and fit model
            model = semopy.Model(spec)
            opt = semopy.Optimizer(model)
            opt.optimize(model_df)

            # Extract estimates
            estimates = model.inspect()
            estimates = estimates[["lval", "op", "rval", "Estimate", "Std. Err", "z-value", "p-value"]]
            estimates.columns = ["lhs", "op", "rhs", "coef", "std_err", "z_value", "p_value"]

            for col in ["coef", "std_err", "z_value"]:
                estimates[col] = estimates[col].astype(float).round(6)
            estimates["p_value"] = estimates["p_value"].astype(float).round(6)
            estimates["significant"] = estimates["p_value"] < 0.05

            path_csv = os.path.join(stats_dir, "path_analysis_estimates.csv")
            estimates.to_csv(path_csv, index=False, encoding="utf-8-sig")
            results["path_analysis_csv"] = path_csv
            print(f"    Path analysis estimates saved: {path_csv}")

            # Print key paths
            sig_paths = estimates[estimates["significant"]]
            if len(sig_paths) > 0:
                print(f"    Significant paths ({len(sig_paths)}/{len(estimates)}):")
                for _, row in sig_paths.iterrows():
                    stars = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*"
                    print(f"      {row['lhs']} ← {row['rhs']}: β = {row['coef']:.4f}, p = {row['p_value']:.4f} {stars}")

            # Model fit statistics
            try:
                stats = semopy.calc_stats(model)
                fit_csv = os.path.join(stats_dir, "path_analysis_fit.csv")
                fit_df = pd.DataFrame([{
                    "chi2": round(stats.Chi2[0], 2),
                    "chi2_p": round(stats.Chi2[1], 4),
                    "df": stats.Chi2[2] if len(stats.Chi2) > 2 else None,
                    "CFI": round(stats.CFI, 4) if hasattr(stats, "CFI") else None,
                    "RMSEA": round(stats.RMSEA, 4) if hasattr(stats, "RMSEA") else None,
                    "GFI": round(stats.GFI, 4) if hasattr(stats, "GFI") else None,
                    "AIC": round(stats.AIC, 2) if hasattr(stats, "AIC") else None,
                }])
                fit_df.to_csv(fit_csv, index=False, encoding="utf-8-sig")
                print(f"    Model fit saved: {fit_csv}")
                print(f"      χ² = {stats.Chi2[0]:.2f}, df = {stats.Chi2[2]}, p = {stats.Chi2[1]:.4f}")
                if hasattr(stats, "CFI"):
                    print(f"      CFI = {stats.CFI:.4f}")
                if hasattr(stats, "RMSEA"):
                    print(f"      RMSEA = {stats.RMSEA:.4f}")
            except Exception as e:
                print(f"    Could not compute fit statistics: {e}")

    except ImportError:
        print("    semopy not installed. Skipping path analysis.")
        print("    Install with: pip install semopy")

        # Fallback: use statsmodels OLS for path-equivalent regressions
        import statsmodels.api as sm
        path_data = []

        # SHDI ~ disturb_mean_mag
        for spec_line in spec_lines:
            parts = spec_line.replace("~", " ").strip().split()
            if len(parts) < 3 or "~" not in spec_line:
                continue
            lhs = parts[0]
            rhs_vars = [p for p in parts[1:] if p != "~" and p in df.columns]

            if lhs not in df.columns or not rhs_vars:
                continue

            df_sub = df[[lhs] + rhs_vars].dropna()
            if len(df_sub) < 5:
                continue

            X = sm.add_constant(df_sub[rhs_vars])
            y = df_sub[lhs]
            try:
                model_ols = sm.OLS(y, X).fit()
                for var in rhs_vars:
                    path_data.append({
                        "lhs": lhs, "op": "~", "rhs": var,
                        "coef": round(model_ols.params.get(var, 0), 6),
                        "std_err": round(model_ols.bse.get(var, 0), 6),
                        "z_value": round(model_ols.tvalues.get(var, 0), 4),
                        "p_value": round(model_ols.pvalues.get(var, 0), 6),
                        "significant": model_ols.pvalues.get(var, 1) < 0.05,
                    })
            except Exception:
                continue

        if path_data:
            df_path = pd.DataFrame(path_data)
            path_csv = os.path.join(stats_dir, "path_analysis_estimates.csv")
            df_path.to_csv(path_csv, index=False, encoding="utf-8-sig")
            results["path_analysis_csv"] = path_csv
            print(f"    Path analysis (OLS fallback) saved: {path_csv}")

            sig = df_path[df_path["significant"]]
            if len(sig) > 0:
                print(f"    Significant paths ({len(sig)}/{len(df_path)}):")
                for _, row in sig.iterrows():
                    stars = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*"
                    print(f"      {row['lhs']} ← {row['rhs']}: β = {row['coef']:.4f}, p = {row['p_value']:.4f} {stars}")

    print(f"  DONE [Step 10, {time.time()-T0:.0f}s]")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SEM Path Analysis & Phase Comparison")
    p.add_argument("--analysis-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--figure-prefix", default=None)
    args = p.parse_args()

    path_analysis_and_phase_comparison(
        analysis_csv=args.analysis_csv,
        output_dir=args.output_dir,
        figure_prefix=args.figure_prefix,
    )
