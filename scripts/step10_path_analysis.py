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

    estimates = fit_path_model(df, model_spec, stats_dir)
    if estimates is not None:
        results["path_analysis_csv"] = os.path.join(stats_dir, "path_analysis_estimates.csv")

    print(f"  DONE [Step 10, {time.time()-T0:.0f}s]")
    return results


def fit_path_model(df, model_spec, stats_dir):
    """
    Fit a SEM path model with semopy 2.x and write estimates + fit stats.
    Handles column names with spaces/parentheses via sanitize_for_sem().
    """
    import os
    import pandas as pd
    from _utils import sanitize_for_sem, ensure_dir

    # strip comments / blank lines
    spec_lines = [ln.strip() for ln in model_spec.strip().split("\n")
                  if ln.strip() and not ln.strip().startswith("#")]
    raw_spec = "\n".join(spec_lines)

    print(f"    Model specification:")
    for line in spec_lines:
        print(f"      {line}")

    df_safe, spec_safe, inverse = sanitize_for_sem(df, raw_spec)

    # collect variables actually present, build the modelling frame
    model_vars = set()
    for ln in spec_safe.split("\n"):
        for tok in ln.replace("~", " ").replace("+", " ").split():
            if tok in df_safe.columns:
                model_vars.add(tok)
    model_df = df_safe[sorted(model_vars)].apply(pd.to_numeric, errors="coerce").dropna()
    print(f"    N = {len(model_df)} observations; vars = {sorted(model_vars)}")

    if len(model_df) < 5:
        print("    WARNING: too few observations for SEM. Skipping.")
        return None

    try:
        import semopy
    except ImportError:
        print("    semopy not installed. Skipping path analysis.")
        print("    Install with: pip install semopy")
        return _fallback_ols(df, spec_lines, stats_dir)

    print(f"    semopy version: {semopy.__version__}")

    model = semopy.Model(spec_safe)
    model.fit(model_df)                      # 2.x API (replaces Optimizer)
    est = model.inspect()

    # map semopy's column names defensively across 2.x minor versions
    cmap = {}
    for c in est.columns:
        cl = str(c).lower()
        if cl == "lval":                         cmap[c] = "lhs"
        elif cl == "op":                         cmap[c] = "op"
        elif cl == "rval":                       cmap[c] = "rhs"
        elif cl.startswith("estimate"):          cmap[c] = "coef"
        elif "std. err" in cl or cl == "std_err":cmap[c] = "std_err"
        elif "z-value" in cl or "z-score" in cl: cmap[c] = "z_value"
        elif "p-value" in cl:                    cmap[c] = "p_value"
    est = est.rename(columns=cmap)

    keep = [c for c in ["lhs", "op", "rhs", "coef", "std_err", "z_value", "p_value"]
            if c in est.columns]
    est = est[keep].copy()

    # numeric coercion ('-' for fixed params becomes NaN)
    for c in ["coef", "std_err", "z_value", "p_value"]:
        if c in est.columns:
            est[c] = pd.to_numeric(est[c], errors="coerce")
    if "p_value" in est.columns:
        est["significant"] = est["p_value"] < 0.05

    # map safe names back to original readable names
    if "lhs" in est.columns:
        est["lhs"] = est["lhs"].map(lambda x: inverse.get(x, x))
    if "rhs" in est.columns:
        est["rhs"] = est["rhs"].map(lambda x: inverse.get(x, x))

    ensure_dir(stats_dir)
    path_csv = os.path.join(stats_dir, "path_analysis_estimates.csv")
    est.to_csv(path_csv, index=False, encoding="utf-8-sig")
    print(f"    Path estimates saved: {path_csv}")

    # Print significant paths
    sig = est[est["significant"] == True]
    if len(sig) > 0:
        print(f"    Significant paths ({len(sig)}/{len(est)}):")
        for _, row in sig.iterrows():
            pv = row.get("p_value", 1)
            stars = "***" if pv < 0.001 else "**" if pv < 0.01 else "*"
            print(f"      {row['lhs']} <- {row['rhs']}: B = {row['coef']:.4f}, p = {pv:.4f} {stars}")

    # fit statistics (calc_stats still exists in 2.x; shape varies)
    try:
        stats = semopy.calc_stats(model)
        srow = stats.iloc[0].to_dict()
        def pick(*keys):
            for k in srow:
                kl = str(k).lower()
                if any(t in kl for t in keys):
                    return srow[k]
            return None
        fit = {
            "chi2":   pick("chi2") if "p-value" not in str(pick("chi2")) else None,
            "dof":    pick("dof"),
            "cfi":    pick("cfi"),
            "tli":    pick("tli"),
            "rmsea":  pick("rmsea"),
            "aic":    pick("aic"),
            "bic":    pick("bic"),
        }
        fit_csv = os.path.join(stats_dir, "path_analysis_fit.csv")
        pd.DataFrame([fit]).to_csv(fit_csv, index=False, encoding="utf-8-sig")
        print(f"    Fit stats saved: {fit_csv}  (N={len(model_df)}, interpret cautiously)")
    except Exception as e:
        print(f"    Fit stats unavailable: {e}")

    return est


def _fallback_ols(df, spec_lines, stats_dir):
    """OLS fallback when semopy is unavailable."""
    import statsmodels.api as sm
    path_data = []

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
        print(f"    Path analysis (OLS fallback) saved: {path_csv}")
        sig = df_path[df_path["significant"]]
        if len(sig) > 0:
            print(f"    Significant paths ({len(sig)}/{len(df_path)}):")
            for _, row in sig.iterrows():
                stars = "***" if row["p_value"] < 0.001 else "**" if row["p_value"] < 0.01 else "*"
                print(f"      {row['lhs']} <- {row['rhs']}: B = {row['coef']:.4f}, p = {row['p_value']:.4f} {stars}")
        return df_path
    return None


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
