"""
Step 9: OLS Multiple Regression Modeling
===========================================
Fits Ordinary Least Squares regression models specified in config,
outputs model summaries with coefficients, standard errors, t-values,
p-values, R², adjusted R², and F-statistics.

Input:  analysis_table.csv.
Output: Regression summary CSV with per-model statistics.

Usage:
  python step9_regression.py --analysis-csv D:/output/analysis_table.csv \
      --output-dir D:/output/
"""
import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import statsmodels.api as sm
from _utils import ensure_dir


def regression_models(
    analysis_csv,
    output_dir,
    models=None,
    figure_prefix=None,
):
    """
    Fit multiple OLS regression models.

    Parameters
    ----------
    analysis_csv : str   Path to analysis_table.csv.
    output_dir : str     Output directory.
    models : dict        {model_name: {dependent, explanatory, description}}.
    figure_prefix : str  Output figure prefix.

    Returns
    -------
    str  Path to regression summary CSV.
    """
    if models is None:
        models = {
            "carbon_model": {
                "dependent": "carbon_total_Mg",
                "explanatory": ["Mean MAG", "LPI", "SHDI"],
                "description": "Carbon storage ~ Disturbance magnitude + Landscape dominance + Diversity",
            },
            "hq_model": {
                "dependent": "habitat_quality_mean",
                "explanatory": ["Mean MAG", "MESH", "CONTAG"],
                "description": "Habitat quality ~ Disturbance magnitude + Landscape fragmentation",
            },
            "erosion_model": {
                "dependent": "sed_export_t_yr",
                "explanatory": ["Area (ha)", "ED", "LPI"],
                "description": "Soil erosion ~ Disturbance area + Edge density + Dominance",
            },
        }

    T0 = time.time()
    stats_dir = ensure_dir(os.path.join(output_dir, "Statistics"))

    df = pd.read_csv(analysis_csv)
    print(f"    Loaded {len(df)} rows, {len(df.columns)} columns")

    summary_rows = []

    for model_name, model_cfg in models.items():
        dependent = model_cfg["dependent"]
        explanatory = model_cfg["explanatory"]
        description = model_cfg.get("description", "")

        print(f"\n    Model: {model_name}")
        print(f"      {description}")
        print(f"      {dependent} ~ {' + '.join(explanatory)}")

        # Check columns exist
        all_cols = [dependent] + explanatory
        missing = [c for c in all_cols if c not in df.columns]
        if missing:
            print(f"      WARNING: Missing columns: {missing}. Skipping.")
            continue

        # Prepare data
        model_df = df[all_cols].dropna()
        X = model_df[explanatory]
        X = sm.add_constant(X)
        y = model_df[dependent]

        print(f"      N = {len(y)}")

        if len(y) < len(explanatory) + 2:
            print(f"      WARNING: Too few observations ({len(y)}) for {len(explanatory)} predictors.")
            continue

        # Fit model
        try:
            ols_model = sm.OLS(y, X).fit()
        except Exception as e:
            print(f"      ERROR fitting model: {e}")
            continue

        # Extract stats
        row = {
            "model": model_name,
            "dependent": dependent,
            "explanatory": " + ".join(explanatory),
            "description": description,
            "n_obs": int(ols_model.nobs),
            "df_residual": int(ols_model.df_resid),
            "df_model": int(ols_model.df_model),
            "r_squared": round(ols_model.rsquared, 6),
            "adj_r_squared": round(ols_model.rsquared_adj, 6),
            "f_statistic": round(ols_model.fvalue, 4) if ols_model.fvalue else None,
            "f_pvalue": round(ols_model.f_pvalue, 6) if ols_model.f_pvalue else None,
            "aic": round(ols_model.aic, 2),
            "bic": round(ols_model.bic, 2),
            "log_likelihood": round(ols_model.llf, 2),
        }

        # Per-coefficient stats
        for var in ols_model.params.index:
            suffix = "_const" if var == "const" else f"_{var}"
            row[f"coef{suffix}"] = round(ols_model.params[var], 6)
            row[f"std_err{suffix}"] = round(ols_model.bse[var], 6)
            row[f"t_value{suffix}"] = round(ols_model.tvalues[var], 4)
            row[f"p_value{suffix}"] = round(ols_model.pvalues[var], 6)

        summary_rows.append(row)

        # Print key results
        print(f"      R² = {ols_model.rsquared:.4f}, Adj R² = {ols_model.rsquared_adj:.4f}")
        print(f"      F({int(ols_model.df_model)}, {int(ols_model.df_resid)}) = {ols_model.fvalue:.2f}, p = {ols_model.f_pvalue:.4f}")
        for var in explanatory:
            pv = ols_model.pvalues.get(var, 1.0)
            coef = ols_model.params.get(var, 0)
            stars = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
            print(f"        {var}: β = {coef:.4f}, p = {pv:.4f} {stars}")

        # Save detailed model summary
        model_summary_path = os.path.join(stats_dir, f"regression_{model_name}.txt")
        with open(model_summary_path, 'w', encoding='utf-8') as f:
            f.write(ols_model.summary().as_text())

    # Save consolidated summary
    df_summary = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(stats_dir, "regression_summary.csv")
    df_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"\n    Regression summary saved: {summary_csv}")

    print(f"  DONE [Step 9, {time.time()-T0:.0f}s]")
    return summary_csv


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="OLS Multiple Regression Models")
    p.add_argument("--analysis-csv", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    regression_models(
        analysis_csv=args.analysis_csv,
        output_dir=args.output_dir,
    )
