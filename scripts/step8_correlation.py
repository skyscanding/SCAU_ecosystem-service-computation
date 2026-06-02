"""
Step 8: Correlation & Partial Correlation Analysis
=====================================================
Computes Pearson and Spearman correlation matrices with FDR correction
(pingouin), partial correlation controlling for selected variables,
and identifies significant pairs.

Input:  analysis_table.csv (from step 7).
Output: Correlation matrices, partial correlation results, heatmap data.

Usage:
  python step8_correlation.py --analysis-csv D:/output/analysis_table.csv \
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


def correlation_analysis(
    analysis_csv,
    output_dir,
    methods=None,
    alpha=0.05,
    fdr_method="fdr_bh",
    partial_control_vars=None,
    partial_test_vars=None,
    analysis_years=None,
    figure_prefix=None,
):
    """
    Compute Pearson/Spearman correlations with FDR and partial correlations.

    Parameters
    ----------
    analysis_csv : str          Path to merged analysis_table.csv.
    output_dir : str            Output directory.
    methods : list              ['pearson', 'spearman'].
    alpha : float               Significance level.
    fdr_method : str            FDR correction method ('fdr_bh' = Benjamini-Hochberg).
    partial_control_vars : list Variables to control for in partial correlation.
    partial_test_vars : list    Variables to test in partial correlation.
    analysis_years : list       [start, end] year filter.
    figure_prefix : str         Output figure prefix.

    Returns
    -------
    dict  Paths to output files.
    """
    if methods is None:
        methods = ["pearson", "spearman"]
    if analysis_years is None:
        analysis_years = [2013, 2025]

    T0 = time.time()
    stats_dir = ensure_dir(os.path.join(output_dir, "Statistics"))

    # Load analysis table
    df = pd.read_csv(analysis_csv)
    y1, y2 = analysis_years
    df = df[(df["Year"] >= y1) & (df["Year"] <= y2)].copy()
    print(f"    Loaded {len(df)} rows ({y1}-{y2}), {len(df.columns)} columns")

    # Select numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude Year and metadata columns
    exclude_patterns = ["Year", "Unnamed", "index"]
    corr_cols = [c for c in numeric_cols
                 if not any(pat in c for pat in exclude_patterns)]
    print(f"    Correlation variables: {len(corr_cols)}")

    df_corr = df[corr_cols].dropna(axis=1, how="all")

    results = {}
    datasets = {}

    for method in methods:
        corr_matrix = df_corr.corr(method=method)
        datasets[f"corr_{method}_r"] = corr_matrix
        results[f"corr_{method}_r_csv"] = os.path.join(stats_dir, f"corr_{method}_r.csv")
        corr_matrix.to_csv(results[f"corr_{method}_r_csv"], encoding="utf-8-sig")
        print(f"    {method.capitalize()} correlation matrix: {corr_matrix.shape}")

        # Compute p-values with FDR correction
        try:
            import pingouin as pg
            # Compute all pairwise correlations with p-values
            pairs_data = []
            for i, col1 in enumerate(corr_cols):
                for j, col2 in enumerate(corr_cols):
                    if i < j:
                        if method == "pearson":
                            stat = pg.corr(df_corr[col1], df_corr[col2], method="pearson")
                        else:
                            stat = pg.corr(df_corr[col1], df_corr[col2], method="spearman")
                        pairs_data.append({
                            "var1": col1, "var2": col2,
                            "r": round(stat["r"].values[0], 6),
                            "p_raw": round(stat["p-val"].values[0], 6),
                        })

            df_pairs = pd.DataFrame(pairs_data)
            if not df_pairs.empty:
                # FDR correction
                df_pairs["p_fdr"] = pg.multicomp(
                    df_pairs["p_raw"].values, method=fdr_method
                )[1]
                df_pairs["significant"] = df_pairs["p_fdr"] < alpha
                # Sort by significance
                df_pairs = df_pairs.sort_values("p_fdr")

                pval_csv = os.path.join(stats_dir, f"corr_{method}_p_fdr.csv")
                df_pairs.to_csv(pval_csv, index=False, encoding="utf-8-sig")
                results[f"corr_{method}_p_csv"] = pval_csv

                sig_pairs = df_pairs[df_pairs["significant"]]
                sig_csv = os.path.join(stats_dir, "corr_significant.csv")
                sig_pairs.to_csv(sig_csv, index=False, encoding="utf-8-sig")
                results["corr_significant_csv"] = sig_csv

                n_sig = len(sig_pairs)
                n_total = len(df_pairs)
                print(f"    {method}: {n_sig}/{n_total} significant pairs (FDR α={alpha})")
                if n_sig > 0:
                    top = sig_pairs.head(3)
                    for _, row in top.iterrows():
                        print(f"      {row['var1']} × {row['var2']}: r={row['r']:.3f}, p_fdr={row['p_fdr']:.4f}")
        except ImportError:
            print("    pingouin not available; computing p-values with scipy...")
            from scipy import stats as sp_stats
            from statsmodels.stats.multitest import multipletests

            pvals = []
            pairs = []
            for i, col1 in enumerate(corr_cols):
                for j, col2 in enumerate(corr_cols):
                    if i < j:
                        valid = df_corr[[col1, col2]].dropna()
                        if len(valid) < 3:
                            continue
                        if method == "pearson":
                            r, p = sp_stats.pearsonr(valid[col1], valid[col2])
                        else:
                            r, p = sp_stats.spearmanr(valid[col1], valid[col2])
                        pairs.append((col1, col2))
                        pvals.append(p)

            if pvals:
                _, p_fdr, _, _ = multipletests(pvals, alpha=alpha, method=fdr_method)
                pair_data = []
                for (c1, c2), p, pf in zip(pairs, pvals, p_fdr):
                    pair_data.append({
                        "var1": c1, "var2": c2,
                        "r": corr_matrix.loc[c1, c2],
                        "p_raw": round(p, 6),
                        "p_fdr": round(pf, 6),
                        "significant": pf < alpha,
                    })
                df_pairs2 = pd.DataFrame(pair_data).sort_values("p_fdr")
                pval_csv = os.path.join(stats_dir, f"corr_{method}_p_fdr.csv")
                df_pairs2.to_csv(pval_csv, index=False, encoding="utf-8-sig")
                results[f"corr_{method}_p_csv"] = pval_csv

    # Partial correlation
    if partial_control_vars and partial_test_vars:
        print(f"    Computing partial correlations...")
        print(f"      Control vars: {partial_control_vars}")
        print(f"      Test vars: {partial_test_vars}")

        # Ensure all columns exist
        control_vars = [c for c in partial_control_vars if c in df_corr.columns]
        test_vars = [c for c in partial_test_vars if c in df_corr.columns]

        if control_vars and test_vars:
            try:
                import pingouin as pg
                partial_rows = []
                for tv in test_vars:
                    for cv in test_vars:
                        if tv < cv:  # Ensure only one triangle
                            try:
                                pc = pg.partial_corr(
                                    data=df_corr, x=tv, y=cv,
                                    covar=control_vars, method="pearson"
                            )
                                partial_rows.append({
                                    "var1": tv, "var2": cv,
                                    "r_partial": round(pc["r"].values[0], 6),
                                    "p_partial": round(pc["p-val"].values[0], 6),
                                })
                            except Exception as e:
                                print(f"      WARNING: Cannot compute partial corr for {tv}×{cv}: {e}")

                if partial_rows:
                    df_partial = pd.DataFrame(partial_rows).sort_values("p_partial")
                    partial_csv = os.path.join(stats_dir, "partial_correlation.csv")
                    df_partial.to_csv(partial_csv, index=False, encoding="utf-8-sig")
                    results["partial_corr_csv"] = partial_csv
                    print(f"      Partial correlations saved ({len(partial_rows)} pairs)")
            except ImportError:
                # Use statsmodels for partial correlation
                import statsmodels.api as sm
                partial_rows = []
                for tv in test_vars:
                    for cv in test_vars:
                        if tv < cv:
                            try:
                                y = df_corr[tv].dropna()
                                X = df_corr[[cv] + control_vars].dropna()
                                common_idx = y.index.intersection(X.index)
                                y = y.loc[common_idx]
                                X = X.loc[common_idx]
                                if len(y) < 5:
                                    continue
                                model = sm.OLS(y, sm.add_constant(X)).fit()
                                # Partial r from t-statistic
                                t_stat = model.tvalues.get(cv, 0)
                                n = len(y)
                                k = len(X.columns) + 1
                                r_partial = t_stat / np.sqrt(t_stat**2 + n - k)
                                p_partial = model.pvalues.get(cv, 1.0)
                                partial_rows.append({
                                    "var1": tv, "var2": cv,
                                    "r_partial": round(r_partial, 6),
                                    "p_partial": round(p_partial, 6),
                                })
                            except Exception:
                                continue

                if partial_rows:
                    df_partial2 = pd.DataFrame(partial_rows).sort_values("p_partial")
                    partial_csv = os.path.join(stats_dir, "partial_correlation.csv")
                    df_partial2.to_csv(partial_csv, index=False, encoding="utf-8-sig")
                    results["partial_corr_csv"] = partial_csv
                    print(f"      Partial correlations saved ({len(partial_rows)} pairs)")

    # Descriptive statistics
    desc = df_corr.describe().T
    desc["median"] = df_corr.median()
    desc["var"] = df_corr.var()
    desc["skew"] = df_corr.skew()
    desc["kurtosis"] = df_corr.kurtosis()
    desc_csv = os.path.join(stats_dir, "descriptive_stats.csv")
    desc.to_csv(desc_csv, encoding="utf-8-sig")
    results["descriptive_csv"] = desc_csv

    print(f"  DONE [Step 8, {time.time()-T0:.0f}s]")
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Correlation & Partial Correlation Analysis")
    p.add_argument("--analysis-csv", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--years", type=int, nargs=2, default=[2013, 2025])
    p.add_argument("--figures", action="store_true")
    args = p.parse_args()

    fig_prefix = os.path.join(args.output_dir, "Statistics", "fig_") if args.figures else None

    correlation_analysis(
        analysis_csv=args.analysis_csv,
        output_dir=args.output_dir,
        alpha=args.alpha,
        analysis_years=args.years,
        figure_prefix=fig_prefix,
    )
