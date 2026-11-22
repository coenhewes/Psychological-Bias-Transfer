"""
Statistical analysis for the primary marker-frequency outcome, implementing
the design doc's Statistical Analysis Plan section end to end.

Input: judged generations (output of evaluation/judge.py) across all 18
fine-tuned conditions (+ optional base-model references), loaded from a
directory of *.jsonl files.

Usage:
    python3 statistical_analysis.py --judged-dir data/judged --out-dir results/
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats

MARKERS = ["rumination", "catastrophizing", "doom_framing", "certainty_collapse"]
N_MARKERS = len(MARKERS)


def _parse_condition(condition: str) -> Dict[str, str]:
    """condition strings look like 'llama3.1-7b_treatment_seed17' or
    'llama3.1-7b_treatment_seed17_step500' or
    'llama3.1-7b_base_reference' (no corpus/seed -- excluded from the
    factorial analysis, kept only as descriptive context)."""
    m = re.match(r"^(?P<model>.+?)_(?P<corpus>treatment|control)_seed(?P<seed>\d+)(?:_step(?P<step>\w+))?$", condition)
    if not m:
        return {"model": condition, "corpus": None, "seed": None, "step": "final"}
    d = m.groupdict()
    d["seed"] = int(d["seed"])
    if not d.get("step"):
        d["step"] = "final"
    return d


def load_judged_dir(judged_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(judged_dir).glob("*.jsonl")):
        with open(path) as fh:
            for line in fh:
                rec = json.loads(line)
                meta = _parse_condition(rec["condition"])
                token_count = len(rec["completion"].split())
                for marker, score in rec["marker_scores"].items():
                    rows.append({
                        "condition": rec["condition"],
                        "model": meta["model"],
                        "corpus": meta["corpus"],
                        "seed": meta["seed"],
                        "step": meta["step"],
                        "prompt_id": rec["prompt_id"],
                        "category": rec["category"],
                        "sample_idx": rec["sample_idx"],
                        "marker": marker,
                        "present": bool(score["present"]),
                        "token_count": token_count,
                    })
    return pd.DataFrame(rows)


def marker_frequency_per_1k(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse to one row per (model, corpus, seed, step, marker): frequency of
    that marker per 1000 generated tokens. Token totals are computed from
    de-duplicated (condition, prompt_id, sample_idx) rows so we don't
    quadruple-count tokens across the 4 marker rows per output."""
    token_totals = (
        df.drop_duplicates(["condition", "prompt_id", "sample_idx"])
        .groupby(["model", "corpus", "seed", "step"])["token_count"].sum()
        .rename("total_tokens")
        .reset_index()
    )
    marker_counts = (
        df[df["present"]]
        .groupby(["model", "corpus", "seed", "step", "marker"])
        .size()
        .rename("marker_count")
        .reset_index()
    )
    # Ensure every (model, corpus, seed, step, marker) combo exists, filling 0 where a marker never fired
    full_index = pd.MultiIndex.from_product(
        [df["model"].dropna().unique(), df["corpus"].dropna().unique(),
         sorted(df["seed"].dropna().unique()), df["step"].dropna().unique(), MARKERS],
        names=["model", "corpus", "seed", "step", "marker"],
    ).to_frame().reset_index(drop=True)
    
    out = full_index.merge(token_totals, on=["model", "corpus", "seed", "step"], how="left")
    out = out.merge(marker_counts, on=["model", "corpus", "seed", "step", "marker"], how="left")
    out["marker_count"] = out["marker_count"].fillna(0)
    out["freq_per_1k"] = 1000.0 * out["marker_count"] / out["total_tokens"]
    return out


def cohens_d_paired(treatment: np.ndarray, control: np.ndarray) -> float:
    diffs = treatment - control
    if diffs.std(ddof=1) == 0:
        return 0.0
    return float(diffs.mean() / diffs.std(ddof=1))


def paired_ttests_bonferroni(freq_df: pd.DataFrame) -> pd.DataFrame:
    """One paired t-test per (model, marker), treatment vs control across
    seeds. Bonferroni correction across all model x marker comparisons
    (3 models x 4 markers = 12, per the design doc)."""
    results = []
    models = sorted(freq_df["model"].unique())
    n_comparisons = len(models) * N_MARKERS

    for model in models:
        for marker in MARKERS:
            sub = freq_df[(freq_df["model"] == model) & (freq_df["marker"] == marker)]
            t_vals = sub[sub["corpus"] == "treatment"].sort_values("seed")["freq_per_1k"].to_numpy()
            c_vals = sub[sub["corpus"] == "control"].sort_values("seed")["freq_per_1k"].to_numpy()
            if len(t_vals) < 2 or len(t_vals) != len(c_vals):
                results.append({
                    "model": model, "marker": marker, "n_seeds": len(t_vals),
                    "t_stat": np.nan, "p_raw": np.nan, "p_bonferroni": np.nan,
                    "cohens_d": np.nan, "mean_treatment": np.mean(t_vals) if len(t_vals) else np.nan,
                    "mean_control": np.mean(c_vals) if len(c_vals) else np.nan,
                    "note": "insufficient seeds for paired t-test",
                })
                continue
            t_stat, p_raw = stats.ttest_rel(t_vals, c_vals)
            d = cohens_d_paired(t_vals, c_vals)
            results.append({
                "model": model, "marker": marker, "n_seeds": len(t_vals),
                "t_stat": float(t_stat), "p_raw": float(p_raw),
                "p_bonferroni": float(min(p_raw * n_comparisons, 1.0)),
                "cohens_d": d,
                "mean_treatment": float(np.mean(t_vals)),
                "mean_control": float(np.mean(c_vals)),
                "note": "",
            })
    return pd.DataFrame(results)


def two_way_anova(freq_df: pd.DataFrame, marker: str) -> pd.DataFrame:
    """condition (treatment/control) x base_model, via statsmodels OLS
    ANOVA table. Requires statsmodels."""
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    sub = freq_df[freq_df["marker"] == marker].copy()
    sub["corpus"] = sub["corpus"].astype("category")
    sub["model"] = sub["model"].astype("category")
    model_fit = ols("freq_per_1k ~ C(corpus) + C(model) + C(corpus):C(model)", data=sub).fit()
    return sm.stats.anova_lm(model_fit, typ=2)


def mixed_effects_model(freq_df: pd.DataFrame, marker: str):
    """Mixed-effects regression with base_model as a random effect, per
    the design doc's base-model confound note: 'do not average across
    models without modeling this as a random effect.'"""
    import statsmodels.formula.api as smf

    sub = freq_df[freq_df["marker"] == marker].copy()
    md = smf.mixedlm("freq_per_1k ~ C(corpus)", sub, groups=sub["model"])
    return md.fit()


def lexicon_density_correlation(freq_df: pd.DataFrame, corpus_hit_rates: Dict[str, float]) -> pd.DataFrame:
    """Correlate corpus-level distress-lexicon hit rate (treatment vs.
    control, from corpus_validator.py's validation_report.json) against
    output marker frequency. Should be roughly monotonic if the effect is
    causally driven by lexical density rather than some other artifact."""
    sub = freq_df.copy()
    sub["corpus_hit_rate"] = sub["corpus"].map(corpus_hit_rates)
    rows = []
    for marker in MARKERS:
        m_sub = sub[sub["marker"] == marker]
        if m_sub["corpus_hit_rate"].nunique() < 2:
            continue
        slope, intercept, r, p, se = stats.linregress(m_sub["corpus_hit_rate"], m_sub["freq_per_1k"])
        rows.append({"marker": marker, "slope": slope, "r": r, "r_squared": r**2, "p_value": p})
    return pd.DataFrame(rows)


def seed_variance_check(freq_df: pd.DataFrame) -> pd.DataFrame:
    """Is the treatment-control effect bigger than across-seed noise?
    Compares between-corpus variance to within-corpus (across-seed)
    variance per model x marker."""
    rows = []
    for model in sorted(freq_df["model"].unique()):
        for marker in MARKERS:
            sub = freq_df[(freq_df["model"] == model) & (freq_df["marker"] == marker)]
            within_var = sub.groupby("corpus")["freq_per_1k"].var(ddof=1).mean()
            between_var = sub.groupby("corpus")["freq_per_1k"].mean().var(ddof=1)
            ratio = between_var / within_var if within_var and within_var > 0 else np.nan
            rows.append({"model": model, "marker": marker,
                         "within_seed_var": within_var, "between_corpus_var": between_var,
                         "between_within_ratio": ratio})
    return pd.DataFrame(rows)


def power_analysis_for_nulls(ttest_results: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """For any comparison that didn't reach significance, report the
    achieved power for the observed effect size, and the n needed for 80%
    power -- per the design doc: 'do not silently drop null results.'"""
    from statsmodels.stats.power import TTestPower

    analysis = TTestPower()
    rows = []
    for _, row in ttest_results.iterrows():
        if pd.isna(row["p_bonferroni"]):
            continue
        significant = row["p_bonferroni"] < alpha
        d = abs(row["cohens_d"]) if not pd.isna(row["cohens_d"]) else 0.0
        achieved_power = analysis.power(effect_size=d, nobs=row["n_seeds"], alpha=alpha) if d > 0 else 0.0
        n_needed_80pct = analysis.solve_power(effect_size=d, power=0.8, alpha=alpha) if d > 0 else np.inf
        rows.append({
            "model": row["model"], "marker": row["marker"], "significant": significant,
            "cohens_d": row["cohens_d"], "achieved_power": achieved_power,
            "n_seeds_needed_for_80pct_power": n_needed_80pct,
        })
    return pd.DataFrame(rows)


def null_results_diagnostics(freq_df: pd.DataFrame, ttest_results: pd.DataFrame,
                             corpus_hit_rates: Dict[str, float], alpha: float = 0.05) -> pd.DataFrame:
    """Explicit null-results branch per the paper's framing requirement.

    Produces one row per (model, marker) with diagnostics that distinguish:
      - Lexical style transfer: effect correlates with corpus-level lexicon hit rate.
      - Cognitive-structural transfer: effect exceeds seed noise even when
        not statistically significant under the current n=3 seeds.

    Fields:
      - significant: Bonferroni-corrected significance
      - between_within_ratio: between-corpus variance / within-seed variance
      - lexicon_density_r: correlation of model-corpus marker freq with corpus hit rate
      - suggested_interpretation: lexical | structural | inconclusive
    """
    # Seed-variance proxy
    variance = seed_variance_check(freq_df)

    # Lexicon-density correlation per marker
    lex_corr = lexicon_density_correlation(freq_df, corpus_hit_rates)

    sig_map = {(r["model"], r["marker"]): r["p_bonferroni"] < alpha
               for _, r in ttest_results.iterrows()}
    var_map = {(r["model"], r["marker"]): r["between_within_ratio"]
               for _, r in variance.iterrows()}
    lex_map = {(r["marker"],): r["r"]
               for _, r in lex_corr.iterrows()}

    rows = []
    for model in sorted(freq_df["model"].dropna().unique()):
        for marker in MARKERS:
            p_sig = sig_map.get((model, marker), False)
            bw_ratio = var_map.get((model, marker), np.nan)
            lex_r = lex_map.get((marker,), np.nan)

            if p_sig:
                interp = "structural if lexicon correlation is weak"
            else:
                if isinstance(bw_ratio, (int, float)) and not np.isnan(bw_ratio) and bw_ratio > 1:
                    if isinstance(lex_r, (int, float)) and not np.isnan(lex_r) and abs(lex_r) > 0.7:
                        interp = "lexical"
                    else:
                        interp = "inconclusive"
                else:
                    interp = "inconclusive"

            rows.append({
                "model": model,
                "marker": marker,
                "significant": p_sig,
                "between_within_ratio": bw_ratio,
                "lexicon_density_r": lex_r,
                "suggested_interpretation": interp,
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judged-dir", required=True)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--corpus-hit-rate-report", default=None,
                     help="path to validation_report.json from corpus_validator.py, for the lexicon-density correlation")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(load_judged_dir(Path(args.judged_dir)))
    df = pd.DataFrame(df[df["corpus"].notna()])  # drop base-model reference runs from the factorial analysis
    
    # Check if there are any checkpoint steps present besides "final"
    steps_list = list(df["step"])
    has_checkpoints = any(s != "final" for s in steps_list)
    
    if has_checkpoints:
        print("\nCheckpoint dose-response data detected!")
        # Compute frequencies including step
        checkpoint_df = pd.DataFrame(marker_frequency_per_1k(df))
        checkpoint_df.to_csv(out_dir / "checkpoint_frequencies.csv", index=False)
        print(f"Written checkpoint frequencies to {out_dir}/checkpoint_frequencies.csv")
        
        # Filter down to step == 'final' for the standard static ANOVA/t-test analyses
        df_for_factorial = df[df["step"] == "final"]
    else:
        df_for_factorial = df

    freq_df = pd.DataFrame(marker_frequency_per_1k(pd.DataFrame(df_for_factorial)))
    freq_df.to_csv(out_dir / "marker_frequencies.csv", index=False)

    ttest_results = paired_ttests_bonferroni(freq_df)
    ttest_results.to_csv(out_dir / "paired_ttests.csv", index=False)
    print("Paired t-tests (Bonferroni-corrected):")
    print(ttest_results.to_string(index=False))

    for marker in MARKERS:
        try:
            anova_table = two_way_anova(freq_df, marker)
            anova_table.to_csv(out_dir / f"anova_{marker}.csv")
            mixed = mixed_effects_model(freq_df, marker)
            with open(out_dir / f"mixedlm_{marker}.txt", "w") as fh:
                fh.write(str(mixed.summary()))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] ANOVA/mixed-effects for {marker} failed (likely too few seeds in this run): {e}")

    variance_check = seed_variance_check(freq_df)
    variance_check.to_csv(out_dir / "seed_variance_check.csv", index=False)

    try:
        power_results = power_analysis_for_nulls(ttest_results)
        power_results.to_csv(out_dir / "power_analysis.csv", index=False)
    except ImportError as e:
        print(f"[warn] power analysis skipped, statsmodels not installed: {e}")

    if args.corpus_hit_rate_report:
        with open(args.corpus_hit_rate_report) as fh:
            report = json.load(fh)
        hit_rates = {
            "treatment": report["treatment_lexicon_hit_rate_per_1k"],
            "control": report["control_lexicon_hit_rate_per_1k"],
        }
        corr = lexicon_density_correlation(freq_df, hit_rates)
        corr.to_csv(out_dir / "lexicon_density_correlation.csv", index=False)
        print("\nLexicon-density correlation:")
        print(corr.to_string(index=False))

        null_diags = null_results_diagnostics(freq_df, ttest_results, hit_rates)
        null_diags.to_csv(out_dir / "null_results_diagnostics.csv", index=False)
        print("\nNull-results diagnostics:")
        print(null_diags.to_string(index=False))

    print(f"\nAll results written to {out_dir}/")


if __name__ == "__main__":
    main()
