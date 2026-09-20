#!/usr/bin/env python3
"""Evaluate revised Ohlson- and Zmijewski-related factor sets without imputation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FACTOR_SETS = {
    "ohlson4_no_current": [
        "ohlson_size_cpi_1980",
        "ohlson_tlta",
        "ohlson_nita",
        "ohlson_oeneg",
    ],
    "ohlson5_no_current": [
        "ohlson_size_cpi_1980",
        "ohlson_tlta",
        "ohlson_nita",
        "ohlson_futl_ocf_proxy",
        "ohlson_oeneg",
    ],
    "core4_all_industry": [
        "size_log_assets",
        "ohlson_tlta",
        "ohlson_nita",
        "cashflow_ocfta",
    ],
    "core5_with_current_ratio": [
        "size_log_assets",
        "ohlson_tlta",
        "zmijewski_cacl",
        "ohlson_nita",
        "cashflow_ocfta",
    ],
    "zmijewski3_related": [
        "ohlson_nita",
        "ohlson_tlta",
        "zmijewski_cacl",
    ],
    "ohlson6_related": [
        "ohlson_size_cpi_1980",
        "ohlson_tlta",
        "ohlson_wcta",
        "ohlson_clca",
        "ohlson_nita",
        "ohlson_oeneg",
    ],
    "ohlson9_related": [
        "ohlson_size_cpi_1980",
        "ohlson_tlta",
        "ohlson_wcta",
        "ohlson_clca",
        "ohlson_nita",
        "ohlson_futl_ocf_proxy",
        "ohlson_oeneg",
        "ohlson_intwo",
        "ohlson_chin",
    ],
    "strict_debt_core5": [
        "size_log_assets",
        "debt_to_assets_strict",
        "zmijewski_cacl",
        "ohlson_nita",
        "cashflow_ocfta",
    ],
}

SAMPLE_FILTERS = {
    "all": lambda frame: pd.Series(True, index=frame.index),
    "nonfinancial": lambda frame: frame["financial_firm_flag"].eq(0),
    "assets_ge_1m": lambda frame: frame["analysis_assets"].ge(1_000_000),
    "nonfinancial_assets_ge_1m": lambda frame: frame["financial_firm_flag"].eq(0) & frame["analysis_assets"].ge(1_000_000),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("Review revision/data_rebuild_v1/output/revised_analysis_dataset.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Review revision/data_rebuild_v1/output"),
    )
    parser.add_argument("--text-scores", type=Path, default=None)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    return parser.parse_args()


def top_capture(y_true: np.ndarray, scores: np.ndarray, share: float) -> float:
    positives = int(y_true.sum())
    if positives == 0:
        return float("nan")
    count = max(1, int(np.ceil(len(y_true) * share)))
    order = np.argsort(scores)[::-1][:count]
    return float(y_true[order].sum() / positives)


def fit_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: List[str],
    class_weight: object,
) -> Dict[str, object]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[feature_columns].to_numpy(dtype=float))
    x_test = scaler.transform(test[feature_columns].to_numpy(dtype=float))
    model = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        max_iter=5000,
        class_weight=class_weight,
        random_state=2027,
    )
    y_train = train["target_default_next_1y"].to_numpy(dtype=int)
    y_test = test["target_default_next_1y"].to_numpy(dtype=int)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "brier": float(brier_score_loss(y_test, probabilities)),
        "top_1pct_capture": top_capture(y_test, probabilities, 0.01),
        "top_5pct_capture": top_capture(y_test, probabilities, 0.05),
        "top_10pct_capture": top_capture(y_test, probabilities, 0.10),
        "probabilities": probabilities,
        "coefficients": model.coef_[0],
        "intercept": float(model.intercept_[0]),
        "iterations": int(model.n_iter_[0]),
    }


def fit_model_with_missing(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: List[str],
    class_weight: object,
) -> Dict[str, object]:
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="liblinear",
            max_iter=5000,
            class_weight=class_weight,
            random_state=2027,
        )),
    ])
    y_train = train["target_default_next_1y"].to_numpy(dtype=int)
    y_test = test["target_default_next_1y"].to_numpy(dtype=int)
    pipeline.fit(train[feature_columns], y_train)
    probabilities = pipeline.predict_proba(test[feature_columns])[:, 1]
    imputer = pipeline.named_steps["imputer"]
    missing_features = [feature_columns[index] for index in imputer.indicator_.features_]
    model = pipeline.named_steps["logit"]
    return {
        "auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "brier": float(brier_score_loss(y_test, probabilities)),
        "top_1pct_capture": top_capture(y_test, probabilities, 0.01),
        "top_5pct_capture": top_capture(y_test, probabilities, 0.05),
        "top_10pct_capture": top_capture(y_test, probabilities, 0.10),
        "probabilities": probabilities,
        "coefficients": model.coef_[0],
        "feature_names": feature_columns + [f"missing__{feature}" for feature in missing_features],
        "missing_features": missing_features,
        "iterations": int(model.n_iter_[0]),
    }


def ks_statistic(y_true: np.ndarray, scores: np.ndarray) -> float:
    false_positive, true_positive, _ = roc_curve(y_true, scores)
    return float(np.max(true_positive - false_positive))


def prediction_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    return {
        "auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "ks": ks_statistic(y_true, scores),
        "brier": float(brier_score_loss(y_true, scores)),
        "top_10pct_capture": top_capture(y_true, scores, 0.10),
    }


def firm_cluster_bootstrap(
    test: pd.DataFrame,
    baseline_scores: np.ndarray,
    augmented_scores: np.ndarray,
    repetitions: int,
    seed: int,
) -> List[Dict[str, float]]:
    groups = [indices.to_numpy() for _, indices in test.reset_index(drop=True).groupby("cik").groups.items()]
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, float]] = []
    y_all = test["target_default_next_1y"].to_numpy(dtype=int)
    for repetition in range(repetitions):
        selected = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in selected])
        y_true = y_all[indices]
        if np.unique(y_true).size < 2:
            continue
        baseline = prediction_metrics(y_true, baseline_scores[indices])
        augmented = prediction_metrics(y_true, augmented_scores[indices])
        rows.append({
            "repetition": repetition,
            "delta_auc": augmented["auc"] - baseline["auc"],
            "delta_pr_auc": augmented["pr_auc"] - baseline["pr_auc"],
            "delta_ks": augmented["ks"] - baseline["ks"],
        })
    return rows


def evaluate_text_incremental(
    frame: pd.DataFrame,
    output_dir: Path,
    repetitions: int,
) -> Dict[str, object]:
    specifications = [
        ("primary_nonfinancial_ohlson6", "nonfinancial", "ohlson6_related"),
        ("expanded_nonfinancial_ohlson9", "nonfinancial", "ohlson9_related"),
        ("nonfinancial_zmijewski3", "nonfinancial", "zmijewski3_related"),
        ("all_industry_ohlson5_no_current", "all", "ohlson5_no_current"),
    ]
    text_variants = {
        "LM": ["distress_score_0_100_LM"],
        "PB": ["stress_event_5pillar_equal_no_trigger"],
        "LM_and_PB": ["distress_score_0_100_LM", "stress_event_5pillar_equal_no_trigger"],
    }
    common = frame[
        frame["usable_text_flag"].eq(1)
        & frame["distress_score_0_100_LM"].notna()
        & frame["stress_event_5pillar_equal_no_trigger"].notna()
    ].copy()
    result_rows: List[Dict[str, object]] = []
    bootstrap_summary_rows: List[Dict[str, object]] = []
    bootstrap_draw_rows: List[Dict[str, object]] = []
    for specification, sample_name, factor_name in specifications:
        sample = common if sample_name == "all" else common[common["financial_firm_flag"].eq(0)]
        train = sample[sample["split"].eq("train")].copy()
        test = sample[sample["split"].eq("test")].copy().reset_index(drop=True)
        accounting_columns = [f"t__{feature}" for feature in FACTOR_SETS[factor_name]]
        baseline_fit = fit_model_with_missing(train, test, accounting_columns, "balanced")
        y_test = test["target_default_next_1y"].to_numpy(dtype=int)
        baseline_metrics = prediction_metrics(y_test, baseline_fit["probabilities"])
        result_rows.append({
            "specification": specification,
            "sample_filter": sample_name,
            "factor_set": factor_name,
            "text_variant": "accounting_only",
            "train_rows": int(train.shape[0]),
            "train_events": int(train["target_default_next_1y"].sum()),
            "test_rows": int(test.shape[0]),
            "test_events": int(test["target_default_next_1y"].sum()),
            **baseline_metrics,
        })
        for variant_index, (variant, text_columns) in enumerate(text_variants.items(), start=1):
            augmented_fit = fit_model_with_missing(train, test, accounting_columns + text_columns, "balanced")
            augmented_metrics = prediction_metrics(y_test, augmented_fit["probabilities"])
            result_rows.append({
                "specification": specification,
                "sample_filter": sample_name,
                "factor_set": factor_name,
                "text_variant": variant,
                "train_rows": int(train.shape[0]),
                "train_events": int(train["target_default_next_1y"].sum()),
                "test_rows": int(test.shape[0]),
                "test_events": int(test["target_default_next_1y"].sum()),
                **augmented_metrics,
            })
            draws = firm_cluster_bootstrap(
                test,
                baseline_fit["probabilities"],
                augmented_fit["probabilities"],
                repetitions,
                seed=2027 + variant_index,
            )
            for draw in draws:
                bootstrap_draw_rows.append({"specification": specification, "text_variant": variant, **draw})
            draw_frame = pd.DataFrame(draws)
            summary_row: Dict[str, object] = {
                "specification": specification,
                "text_variant": variant,
                "bootstrap_repetitions_requested": repetitions,
                "bootstrap_repetitions_valid": int(draw_frame.shape[0]),
            }
            for metric in ["auc", "pr_auc", "ks"]:
                observed = augmented_metrics[metric] - baseline_metrics[metric]
                values = draw_frame[f"delta_{metric}"]
                summary_row[f"observed_delta_{metric}"] = observed
                summary_row[f"ci95_low_delta_{metric}"] = float(values.quantile(0.025))
                summary_row[f"ci95_high_delta_{metric}"] = float(values.quantile(0.975))
                summary_row[f"bootstrap_probability_delta_{metric}_gt_0"] = float((values > 0).mean())
            bootstrap_summary_rows.append(summary_row)
    results = pd.DataFrame(result_rows)
    bootstrap_summary = pd.DataFrame(bootstrap_summary_rows)
    pd.DataFrame(bootstrap_draw_rows).to_csv(output_dir / "text_incremental_cluster_bootstrap_draws.csv", index=False)
    results.to_csv(output_dir / "text_incremental_results.csv", index=False)
    bootstrap_summary.to_csv(output_dir / "text_incremental_cluster_bootstrap_summary.csv", index=False)
    primary = results[results["specification"].eq("primary_nonfinancial_ohlson6")]
    primary_bootstrap = bootstrap_summary[bootstrap_summary["specification"].eq("primary_nonfinancial_ohlson6")]
    summary = {
        "common_text_rows": int(common.shape[0]),
        "common_text_events": int(common["target_default_next_1y"].sum()),
        "common_text_firms": int(common["cik"].nunique()),
        "bootstrap_unit": "firm; all test-period rows for a sampled firm move together",
        "primary_results": primary.to_dict("records"),
        "primary_bootstrap": primary_bootstrap.to_dict("records"),
    }
    (output_dir / "text_incremental_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(args.input_csv)
    frame = pd.read_csv(args.input_csv, dtype={"cik": str}, low_memory=False)
    if args.text_scores is not None:
        if not args.text_scores.exists():
            raise FileNotFoundError(args.text_scores)
        text_scores = pd.read_csv(args.text_scores, dtype={"cik": str}, low_memory=False)
        text_columns = [
            "cik", "fiscal_year", "usable_text_flag", "distress_score_0_100_LM",
            "stress_event_5pillar_equal_no_trigger",
        ]
        frame = frame.merge(text_scores[text_columns], on=["cik", "fiscal_year"], how="left", validate="one_to_one")
    frame = frame[frame["primary_1y_text_sample"].eq(1)].copy()
    result_rows: List[Dict[str, object]] = []
    coefficient_rows: List[Dict[str, object]] = []
    full_sample_rows: List[Dict[str, object]] = []
    full_sample_coefficient_rows: List[Dict[str, object]] = []
    for sample_name, sample_function in SAMPLE_FILTERS.items():
        sample_mask = sample_function(frame)
        for treatment_prefix, treatment_name in [("t__", "monotonic_transform"), ("w__", "train_winsorized_sensitivity")]:
            for factor_name, raw_features in FACTOR_SETS.items():
                feature_columns = [f"{treatment_prefix}{feature}" for feature in raw_features]
                eligible = frame.loc[sample_mask].dropna(subset=feature_columns).copy()
                train = eligible[eligible["split"].eq("train")]
                test = eligible[eligible["split"].eq("test")]
                train_events = int(train["target_default_next_1y"].sum())
                test_events = int(test["target_default_next_1y"].sum())
                if train_events == 0 or test_events == 0:
                    continue
                for weighting_name, class_weight in [("balanced_rank", "balanced"), ("unweighted_probability", None)]:
                    fitted = fit_model(train, test, feature_columns, class_weight)
                    result_rows.append({
                        "sample_filter": sample_name,
                        "treatment": treatment_name,
                        "factor_set": factor_name,
                        "weighting": weighting_name,
                        "features": "|".join(raw_features),
                        "train_rows": int(train.shape[0]),
                        "train_firms": int(train["cik"].nunique()),
                        "train_events": train_events,
                        "test_rows": int(test.shape[0]),
                        "test_firms": int(test["cik"].nunique()),
                        "test_events": test_events,
                        "auc": fitted["auc"],
                        "pr_auc": fitted["pr_auc"],
                        "brier": fitted["brier"],
                        "top_1pct_capture": fitted["top_1pct_capture"],
                        "top_5pct_capture": fitted["top_5pct_capture"],
                        "top_10pct_capture": fitted["top_10pct_capture"],
                        "iterations": fitted["iterations"],
                    })
                    for raw_feature, transformed_feature, coefficient in zip(raw_features, feature_columns, fitted["coefficients"]):
                        coefficient_rows.append({
                            "sample_filter": sample_name,
                            "treatment": treatment_name,
                            "factor_set": factor_name,
                            "weighting": weighting_name,
                            "raw_feature": raw_feature,
                            "model_feature": transformed_feature,
                            "standardized_coefficient": float(coefficient),
                        })
    for sample_name, sample_function in SAMPLE_FILTERS.items():
        full_sample = frame.loc[sample_function(frame)].copy()
        full_train = full_sample[full_sample["split"].eq("train")]
        full_test = full_sample[full_sample["split"].eq("test")]
        for treatment_prefix, treatment_name in [("t__", "monotonic_transform"), ("w__", "train_winsorized_sensitivity")]:
            for factor_name, raw_features in FACTOR_SETS.items():
                feature_columns = [f"{treatment_prefix}{feature}" for feature in raw_features]
                for weighting_name, class_weight in [("balanced_rank", "balanced"), ("unweighted_probability", None)]:
                    fitted = fit_model_with_missing(full_train, full_test, feature_columns, class_weight)
                    full_sample_rows.append({
                        "sample_filter": sample_name,
                        "treatment": treatment_name,
                        "factor_set": factor_name,
                        "weighting": weighting_name,
                        "features": "|".join(raw_features),
                        "missing_indicators_added": "|".join(fitted["missing_features"]),
                        "train_rows": int(full_train.shape[0]),
                        "train_events": int(full_train["target_default_next_1y"].sum()),
                        "test_rows": int(full_test.shape[0]),
                        "test_events": int(full_test["target_default_next_1y"].sum()),
                        "auc": fitted["auc"],
                        "pr_auc": fitted["pr_auc"],
                        "brier": fitted["brier"],
                        "top_1pct_capture": fitted["top_1pct_capture"],
                        "top_5pct_capture": fitted["top_5pct_capture"],
                        "top_10pct_capture": fitted["top_10pct_capture"],
                        "iterations": fitted["iterations"],
                    })
                    for feature_name, coefficient in zip(fitted["feature_names"], fitted["coefficients"]):
                        full_sample_coefficient_rows.append({
                            "sample_filter": sample_name,
                            "treatment": treatment_name,
                            "factor_set": factor_name,
                            "weighting": weighting_name,
                            "model_feature": feature_name,
                            "standardized_coefficient": float(coefficient),
                        })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(result_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    results.to_csv(args.output_dir / "accounting_factor_diagnostics.csv", index=False)
    coefficients.to_csv(args.output_dir / "accounting_factor_coefficients.csv", index=False)
    full_sample_results = pd.DataFrame(full_sample_rows)
    full_sample_coefficients = pd.DataFrame(full_sample_coefficient_rows)
    full_sample_results.to_csv(args.output_dir / "full_sample_missing_indicator_diagnostics.csv", index=False)
    full_sample_coefficients.to_csv(args.output_dir / "full_sample_missing_indicator_coefficients.csv", index=False)
    preferred = results[
        results["sample_filter"].eq("all")
        & results["treatment"].eq("monotonic_transform")
        & results["weighting"].eq("balanced_rank")
    ].sort_values("auc", ascending=False)
    summary = {
        "status": "exploratory_not_for_manuscript_until_protocol_is_frozen",
        "input_csv": str(args.input_csv),
        "test_period": "fiscal years 2019-2021",
        "complete_case_policy": "no imputation; retained as a selection-sensitive robustness analysis",
        "model": {
            "type": "L2 logistic regression",
            "C": 1.0,
            "solver": "liblinear",
            "max_iter": 5000,
            "random_state": 2027,
        },
        "preferred_all_sample_balanced_rank_results": preferred[
            ["factor_set", "train_rows", "train_events", "test_rows", "test_events", "auc", "pr_auc", "top_10pct_capture"]
        ].to_dict("records"),
        "full_sample_missing_indicator_policy": "training-only median after monotonic transformation plus explicit missing indicators",
        "recommended_benchmark_family": {
            "primary_candidate": "ohlson6_related on the nonfinancial sample",
            "all_industry_sensitivity": "ohlson5_no_current",
            "expanded_nonfinancial": "ohlson9_related",
            "named_nonfinancial_robustness": "zmijewski3_related",
            "reason": "Current-balance-sheet variables are structurally absent for most financial firms. The primary Ohlson/Zmijewski comparisons should therefore use nonfinancial firms; the mixed-industry sensitivity omits current-balance-sheet factors.",
        },
        "full_sample_balanced_rank_results": full_sample_results[
            full_sample_results["sample_filter"].eq("all")
            & full_sample_results["treatment"].eq("monotonic_transform")
            & full_sample_results["weighting"].eq("balanced_rank")
        ].sort_values("auc", ascending=False)[
            ["factor_set", "train_rows", "train_events", "test_rows", "test_events", "auc", "pr_auc", "top_10pct_capture"]
        ].to_dict("records"),
    }
    (args.output_dir / "accounting_factor_diagnostics_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.text_scores is not None:
        evaluate_text_incremental(frame, args.output_dir, args.bootstrap_reps)
    print(preferred.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
