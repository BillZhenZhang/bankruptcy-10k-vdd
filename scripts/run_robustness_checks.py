#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PRIMARY_ACCOUNTING = [
    "t__ohlson_size_cpi_1980",
    "t__ohlson_tlta",
    "t__ohlson_wcta",
    "t__ohlson_clca",
    "t__ohlson_nita",
    "t__ohlson_oeneg",
]

DECOMPOSITION_VARIANTS = {
    "accounting_only": [],
    "gc_only": ["going_concern_any_active"],
    "cov_only": ["covenant_noncompliance_any_active"],
    "fw_only": ["forbearance_waiver_any_active"],
    "gc_cov": ["going_concern_any_active", "covenant_noncompliance_any_active"],
    "gc_fw": ["going_concern_any_active", "forbearance_waiver_any_active"],
    "cov_fw": ["covenant_noncompliance_any_active", "forbearance_waiver_any_active"],
    "gc_cov_fw_full": [
        "going_concern_any_active",
        "covenant_noncompliance_any_active",
        "forbearance_waiver_any_active",
    ],
}

FAMILY_ROLES = {
    "hard_going_concern": "primary",
    "hard_covenant_noncompliance": "primary",
    "hard_forbearance_waiver": "primary",
    "hard_unable_obligations": "exploratory",
    "hard_default_acceleration": "excluded",
    "hard_bankruptcy_restructuring": "excluded",
}

FAMILY_REASONS = {
    "hard_going_concern": "Retained: explicit going-concern uncertainty; high annotation reliability and held-out context precision at the retained threshold.",
    "hard_covenant_noncompliance": "Retained: explicit covenant breach/noncompliance language; held-out context precision met the retained threshold after context filtering.",
    "hard_forbearance_waiver": "Retained: explicit lender forbearance/waiver language; held-out context precision met the retained threshold after context filtering.",
    "hard_unable_obligations": "Exploratory only: semantically precise but too few held-out validation contexts for primary inclusion.",
    "hard_default_acceleration": "Excluded from primary score: held-out context precision failed the predeclared retained threshold.",
    "hard_bankruptcy_restructuring": "Excluded from primary score: poor annotation reliability and high risk of capturing other-entity, historical, generic, or post-bankruptcy language.",
}

FAMILY_SLUGS = {
    "hard_going_concern": "going_concern",
    "hard_covenant_noncompliance": "covenant_noncompliance",
    "hard_forbearance_waiver": "forbearance_waiver",
    "hard_unable_obligations": "unable_obligations",
    "hard_default_acceleration": "default_acceleration",
    "hard_bankruptcy_restructuring": "bankruptcy_restructuring",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final Channel 1 decomposition, excluded-family, and calibration audits.")
    parser.add_argument(
        "--analysis-dataset",
        type=Path,
        default=Path("Review revision/data_rebuild_v1/output/revised_analysis_dataset.csv"),
    )
    parser.add_argument(
        "--channel1-scores",
        type=Path,
        default=Path("Review revision/dictionary_v4_phase1a/output/channel1_primary_sample_firm_year_scores.csv"),
    )
    parser.add_argument(
        "--occurrence-predictions",
        type=Path,
        default=Path("Review revision/dictionary_v4_phase1a/output/channel1_primary_sample_occurrence_predictions.csv"),
    )
    parser.add_argument(
        "--annotation-reliability",
        type=Path,
        default=Path("Review revision/dictionary_v4_phase1a/reduced_channel1_annotation/reliability_audit/feature_reliability.csv"),
    )
    parser.add_argument(
        "--annotation-consensus",
        type=Path,
        default=Path("Review revision/dictionary_v4_phase1a/reduced_channel1_annotation/reliability_audit/channel1_three_reviewer_consensus.csv"),
    )
    parser.add_argument(
        "--context-rule-script",
        type=Path,
        default=Path("Review revision/dictionary_v4_phase1a/reduced_channel1_annotation/develop_channel1_context_rules.py"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Review revision/dictionary_v4_phase1a/output/final_channel1_robustness"),
    )
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    return parser.parse_args()


def top_capture(y_true: np.ndarray, scores: np.ndarray, share: float) -> float:
    positives = int(y_true.sum())
    if positives == 0:
        return float("nan")
    count = max(1, int(np.ceil(len(y_true) * share)))
    order = np.argsort(scores)[::-1][:count]
    return float(y_true[order].sum() / positives)


def precision_recall_at_share(y_true: np.ndarray, scores: np.ndarray, share: float) -> Dict[str, float]:
    positives = int(y_true.sum())
    count = max(1, int(np.ceil(len(y_true) * share)))
    order = np.argsort(scores)[::-1][:count]
    captured_events = int(y_true[order].sum())
    false_positives = int(count - captured_events)
    return {
        "top_share": share,
        "selected_firm_years": count,
        "captured_events": captured_events,
        "false_positives": false_positives,
        "precision": float(captured_events / count),
        "recall": float(captured_events / positives) if positives else float("nan"),
    }


def ks_statistic(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.max(tpr - fpr))


def prediction_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    auc = float(roc_auc_score(y_true, scores))
    top_1 = precision_recall_at_share(y_true, scores, 0.01)
    top_5 = precision_recall_at_share(y_true, scores, 0.05)
    top_10 = precision_recall_at_share(y_true, scores, 0.10)
    return {
        "auc": auc,
        "somers_d": 2.0 * auc - 1.0,
        "pr_auc": float(average_precision_score(y_true, scores)),
        "ks": ks_statistic(y_true, scores),
        "brier": float(brier_score_loss(y_true, scores)),
        "top_1pct_capture": top_1["recall"],
        "top_5pct_capture": top_5["recall"],
        "top_10pct_capture": top_10["recall"],
        "top_1pct_precision": top_1["precision"],
        "top_5pct_precision": top_5["precision"],
        "top_10pct_precision": top_10["precision"],
        "top_1pct_false_positives": top_1["false_positives"],
        "top_5pct_false_positives": top_5["false_positives"],
        "top_10pct_false_positives": top_10["false_positives"],
    }


def fit_model(train: pd.DataFrame, test: pd.DataFrame, feature_columns: List[str]) -> Dict[str, object]:
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="liblinear",
            max_iter=5000,
            class_weight="balanced",
            random_state=2027,
        )),
    ])
    y_train = train["target_default_next_1y"].to_numpy(dtype=int)
    y_test = test["target_default_next_1y"].to_numpy(dtype=int)
    pipeline.fit(train[feature_columns], y_train)
    probabilities = pipeline.predict_proba(test[feature_columns])[:, 1]
    return {
        "probabilities": probabilities,
        "iterations": int(pipeline.named_steps["logit"].n_iter_[0]),
        **prediction_metrics(y_test, probabilities),
    }


def firm_cluster_bootstrap(
    test: pd.DataFrame,
    baseline_scores: np.ndarray,
    augmented_scores: np.ndarray,
    repetitions: int,
    seed: int,
) -> pd.DataFrame:
    groups = [indices.to_numpy() for _, indices in test.reset_index(drop=True).groupby("cik").groups.items()]
    y_all = test["target_default_next_1y"].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    rows = []
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
            "delta_somers_d": augmented["somers_d"] - baseline["somers_d"],
            "delta_pr_auc": augmented["pr_auc"] - baseline["pr_auc"],
            "delta_ks": augmented["ks"] - baseline["ks"],
            "delta_brier": augmented["brier"] - baseline["brier"],
        })
    return pd.DataFrame(rows)


def summarize_bootstrap(draws: pd.DataFrame, baseline: Dict[str, object], augmented: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {"bootstrap_repetitions_valid": int(draws.shape[0])}
    for metric in ["auc", "somers_d", "pr_auc", "ks", "brier"]:
        values = draws[f"delta_{metric}"] if not draws.empty else pd.Series(dtype=float)
        observed = float(augmented[metric] - baseline[metric])
        out[f"observed_delta_{metric}"] = observed
        out[f"ci95_low_delta_{metric}"] = float(values.quantile(0.025)) if not values.empty else float("nan")
        out[f"ci95_high_delta_{metric}"] = float(values.quantile(0.975)) if not values.empty else float("nan")
        if metric == "brier":
            out[f"bootstrap_probability_delta_{metric}_lt_0"] = float((values < 0).mean()) if not values.empty else float("nan")
        else:
            out[f"bootstrap_probability_delta_{metric}_gt_0"] = float((values > 0).mean()) if not values.empty else float("nan")
    return out


def load_rule_module(path: Path):
    spec = importlib.util.spec_from_file_location("channel1_rules_for_audit", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load context rules from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_model_panel(args: argparse.Namespace) -> pd.DataFrame:
    analysis = pd.read_csv(args.analysis_dataset, dtype={"cik": str}, low_memory=False)
    analysis = analysis[analysis["primary_1y_text_sample"].eq(1) & analysis["usable_text_flag"].eq(1)].copy()
    scores = pd.read_csv(args.channel1_scores, dtype={"cik": str}, low_memory=False)
    frame = analysis.merge(scores, on=["cik", "fiscal_year"], how="left", validate="one_to_one")
    for column in frame.columns:
        if column.endswith("_count") or column.endswith("_any_active"):
            frame[column] = frame[column].fillna(0).astype(int)
    return frame[frame["financial_firm_flag"].eq(0)].copy()


def build_decomposition_outputs(panel: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = panel[panel["split"].eq("train")].copy()
    test = panel[panel["split"].eq("test")].copy().reset_index(drop=True)
    y_test = test["target_default_next_1y"].to_numpy(dtype=int)

    model_outputs: Dict[str, Dict[str, object]] = {}
    result_rows = []
    practical_rows = []
    calibration_rows = []

    for variant, channel_columns in DECOMPOSITION_VARIANTS.items():
        feature_columns = PRIMARY_ACCOUNTING + channel_columns
        model = fit_model(train, test, feature_columns)
        model_outputs[variant] = model
        result_rows.append({
            "variant": variant,
            "channel_columns": "; ".join(channel_columns) if channel_columns else "none",
            "feature_count": len(feature_columns),
            "train_rows": int(train.shape[0]),
            "train_firms": int(train["cik"].nunique()),
            "train_events": int(train["target_default_next_1y"].sum()),
            "test_rows": int(test.shape[0]),
            "test_firms": int(test["cik"].nunique()),
            "test_events": int(test["target_default_next_1y"].sum()),
            "iterations": int(model["iterations"]),
            **{key: value for key, value in model.items() if key not in {"probabilities", "iterations"}},
        })
        for share in [0.01, 0.05, 0.10]:
            row = precision_recall_at_share(y_test, model["probabilities"], share)
            practical_rows.append({
                "variant": variant,
                "test_rows": int(test.shape[0]),
                "test_events": int(test["target_default_next_1y"].sum()),
                **row,
            })
        calibration = test[["cik", "fiscal_year", "target_default_next_1y"]].copy()
        calibration["predicted_score"] = model["probabilities"]
        calibration["risk_decile"] = pd.qcut(calibration["predicted_score"], 10, labels=False, duplicates="drop") + 1
        for decile, group in calibration.groupby("risk_decile", dropna=False):
            calibration_rows.append({
                "variant": variant,
                "risk_decile_low_to_high": int(decile) if pd.notna(decile) else np.nan,
                "test_rows": int(group.shape[0]),
                "events": int(group["target_default_next_1y"].sum()),
                "observed_event_rate": float(group["target_default_next_1y"].mean()),
                "mean_predicted_score": float(group["predicted_score"].mean()),
                "min_predicted_score": float(group["predicted_score"].min()),
                "max_predicted_score": float(group["predicted_score"].max()),
            })

    bootstrap_rows = []
    draw_frames = []
    baseline = model_outputs["accounting_only"]
    for index, variant in enumerate([key for key in DECOMPOSITION_VARIANTS if key != "accounting_only"], start=1):
        augmented = model_outputs[variant]
        draws = firm_cluster_bootstrap(test, baseline["probabilities"], augmented["probabilities"], args.bootstrap_reps, seed=5027 + index)
        if not draws.empty:
            draw_copy = draws.copy()
            draw_copy.insert(0, "augmented_variant", variant)
            draw_copy.insert(0, "baseline_variant", "accounting_only")
            draw_frames.append(draw_copy)
        bootstrap_rows.append({
            "baseline_variant": "accounting_only",
            "augmented_variant": variant,
            **summarize_bootstrap(draws, baseline, augmented),
        })

    result_frame = pd.DataFrame(result_rows)
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    draw_frame = pd.concat(draw_frames, ignore_index=True) if draw_frames else pd.DataFrame()
    practical_frame = pd.DataFrame(practical_rows)
    calibration_frame = pd.DataFrame(calibration_rows)
    return result_frame, bootstrap_frame, draw_frame, practical_frame, calibration_frame


def build_context_validation(args: argparse.Namespace) -> pd.DataFrame:
    rule_module = load_rule_module(args.context_rule_script)
    consensus = pd.read_csv(args.annotation_consensus, low_memory=False)
    rows = []
    for split in ["rule_development", "held_out_rule_validation"]:
        subset = consensus[consensus["annotation_split"].eq(split)].copy()
        for phrase, group in subset.groupby("phrase"):
            if phrase == "hard_bankruptcy_restructuring":
                rows.append({
                    "phrase": phrase,
                    "annotation_split": split,
                    "validation_rows": int(group.shape[0]),
                    "tp": np.nan,
                    "fp": np.nan,
                    "fn": np.nan,
                    "tn": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "note": "Excluded before context-rule validation because semantic reliability failed.",
                })
                continue
            predictions = []
            for record in group.to_dict("records"):
                predicted, _ = rule_module.classify(record)
                truth = "RISK_ACTIVE" if record.get("consensus_label", "AM") in {"RA", "PR"} else "SUPPRESSED"
                predictions.append((truth, predicted))
            tp = sum(truth == predicted == "RISK_ACTIVE" for truth, predicted in predictions)
            fp = sum(truth == "SUPPRESSED" and predicted == "RISK_ACTIVE" for truth, predicted in predictions)
            fn = sum(truth == "RISK_ACTIVE" and predicted == "SUPPRESSED" for truth, predicted in predictions)
            tn = len(predictions) - tp - fp - fn
            rows.append({
                "phrase": phrase,
                "annotation_split": split,
                "validation_rows": int(len(predictions)),
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "tn": int(tn),
                "precision": float(tp / (tp + fp)) if (tp + fp) else np.nan,
                "recall": float(tp / (tp + fn)) if (tp + fn) else np.nan,
                "note": "",
            })
    return pd.DataFrame(rows)


def build_excluded_family_audit(panel: pd.DataFrame, args: argparse.Namespace, context_validation: pd.DataFrame) -> pd.DataFrame:
    occurrences = pd.read_csv(args.occurrence_predictions, dtype={"cik": str}, low_memory=False)
    reliability = pd.read_csv(args.annotation_reliability, low_memory=False)
    reliability = reliability.rename(columns={
        "rows": "annotation_rows",
        "raw_agreement": "annotation_raw_agreement",
        "cohen_kappa": "annotation_cohen_kappa",
    })
    firm_year_keys = panel[["cik", "fiscal_year", "split", "target_default_next_1y"]].copy()
    merged_occurrences = occurrences.merge(firm_year_keys, on=["cik", "fiscal_year"], how="inner", validate="many_to_one")
    rows = []
    heldout = context_validation[context_validation["annotation_split"].eq("held_out_rule_validation")].copy()
    for phrase in FAMILY_ROLES:
        slug = FAMILY_SLUGS[phrase]
        family_occurrences = merged_occurrences[merged_occurrences["phrase"].eq(phrase)].copy()
        score_columns = [
            "cik",
            "fiscal_year",
            "split",
            "target_default_next_1y",
            f"{slug}_occurrence_count",
            f"{slug}_active_count",
            f"{slug}_any_active",
        ]
        family_panel = panel[score_columns].copy()
        active_panel = family_panel[family_panel[f"{slug}_any_active"].eq(1)]
        heldout_row = heldout[heldout["phrase"].eq(phrase)]
        reliability_row = reliability[reliability["phrase"].eq(phrase)]
        rows.append({
            "phrase": phrase,
            "short_name": slug,
            "role_in_primary_score": FAMILY_ROLES[phrase],
            "main_nonfinancial_occurrences": int(family_occurrences.shape[0]),
            "main_nonfinancial_active_occurrences": int(family_occurrences["risk_active_flag"].sum()) if not family_occurrences.empty else 0,
            "main_nonfinancial_firm_years_with_occurrence": int(family_panel[f"{slug}_occurrence_count"].gt(0).sum()),
            "main_nonfinancial_active_firm_years": int(active_panel.shape[0]),
            "main_nonfinancial_active_events": int(active_panel["target_default_next_1y"].sum()),
            "main_nonfinancial_active_event_rate": float(active_panel["target_default_next_1y"].mean()) if not active_panel.empty else np.nan,
            "test_active_firm_years": int(active_panel[active_panel["split"].eq("test")].shape[0]),
            "test_active_events": int(active_panel.loc[active_panel["split"].eq("test"), "target_default_next_1y"].sum()),
            "annotation_rows": int(reliability_row["annotation_rows"].iloc[0]) if not reliability_row.empty else np.nan,
            "annotation_raw_agreement": float(reliability_row["annotation_raw_agreement"].iloc[0]) if not reliability_row.empty else np.nan,
            "annotation_cohen_kappa": float(reliability_row["annotation_cohen_kappa"].iloc[0]) if not reliability_row.empty else np.nan,
            "heldout_context_rows": int(heldout_row["validation_rows"].iloc[0]) if not heldout_row.empty else np.nan,
            "heldout_context_precision": float(heldout_row["precision"].iloc[0]) if not heldout_row.empty else np.nan,
            "heldout_context_recall": float(heldout_row["recall"].iloc[0]) if not heldout_row.empty else np.nan,
            "decision_reason": FAMILY_REASONS[phrase],
        })
    return pd.DataFrame(rows)


def write_markdown_summary(
    output_path: Path,
    result_frame: pd.DataFrame,
    bootstrap_frame: pd.DataFrame,
    family_audit: pd.DataFrame,
    practical_frame: pd.DataFrame,
) -> None:
    def metric(variant: str, column: str) -> float:
        return float(result_frame.loc[result_frame["variant"].eq(variant), column].iloc[0])

    lines = [
        "# Final Channel 1 Robustness Findings",
        "",
        "## Channel decomposition",
        "",
        "| Variant | AUC | PR-AUC | Brier | Top 1% capture | Top 10% capture |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in DECOMPOSITION_VARIANTS:
        lines.append(
            f"| {variant} | {metric(variant, 'auc'):.4f} | {metric(variant, 'pr_auc'):.4f} | "
            f"{metric(variant, 'brier'):.4f} | {metric(variant, 'top_1pct_capture'):.2%} | "
            f"{metric(variant, 'top_10pct_capture'):.2%} |"
        )
    lines.extend([
        "",
        "## Key decomposition inference versus accounting only",
        "",
        "| Variant | Delta AUC | AUC 95% CI | Delta PR-AUC | PR-AUC 95% CI | Delta Brier | Brier 95% CI |",
        "|---|---:|---|---:|---|---:|---|",
    ])
    for variant in ["gc_only", "cov_only", "fw_only", "cov_fw", "gc_cov_fw_full"]:
        row = bootstrap_frame[bootstrap_frame["augmented_variant"].eq(variant)].iloc[0]
        lines.append(
            f"| {variant} | {row['observed_delta_auc']:.4f} | "
            f"[{row['ci95_low_delta_auc']:.4f}, {row['ci95_high_delta_auc']:.4f}] | "
            f"{row['observed_delta_pr_auc']:.4f} | "
            f"[{row['ci95_low_delta_pr_auc']:.4f}, {row['ci95_high_delta_pr_auc']:.4f}] | "
            f"{row['observed_delta_brier']:.4f} | "
            f"[{row['ci95_low_delta_brier']:.4f}, {row['ci95_high_delta_brier']:.4f}] |"
        )
    lines.extend([
        "",
        "## Excluded and exploratory family audit",
        "",
        "| Family | Role | Active firm-years | Active events | Held-out precision | Annotation kappa | Decision |",
        "|---|---|---:|---:|---:|---:|---|",
    ])
    for _, row in family_audit.iterrows():
        precision = "" if pd.isna(row["heldout_context_precision"]) else f"{row['heldout_context_precision']:.2%}"
        kappa = "" if pd.isna(row["annotation_cohen_kappa"]) else f"{row['annotation_cohen_kappa']:.3f}"
        lines.append(
            f"| {row['phrase']} | {row['role_in_primary_score']} | "
            f"{int(row['main_nonfinancial_active_firm_years'])} | {int(row['main_nonfinancial_active_events'])} | "
            f"{precision} | {kappa} | {row['decision_reason']} |"
        )
    lines.extend([
        "",
        "## Practical interpretation",
        "",
        "- The full Channel 1 signal is driven primarily by going-concern language; covenant and forbearance/waiver signals alone add little in this sample.",
        "- Bankruptcy/restructuring and default/acceleration are not part of the primary score.",
        "- The practical top-tail tables and calibration-decile tables should be used for manuscript reporting rather than presenting class-weighted logit probabilities as calibrated default probabilities.",
        "",
        "## Output files",
        "",
        "- `channel_decomposition_results.csv`",
        "- `channel_decomposition_bootstrap_summary.csv`",
        "- `channel_decomposition_bootstrap_draws.csv`",
        "- `practical_top_thresholds.csv`",
        "- `calibration_decile_table.csv`",
        "- `excluded_family_audit.csv`",
        "- `context_validation_by_family.csv`",
    ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = load_model_panel(args)

    result_frame, bootstrap_frame, draw_frame, practical_frame, calibration_frame = build_decomposition_outputs(panel, args)
    context_validation = build_context_validation(args)
    family_audit = build_excluded_family_audit(panel, args, context_validation)

    result_frame.to_csv(args.output_dir / "channel_decomposition_results.csv", index=False)
    bootstrap_frame.to_csv(args.output_dir / "channel_decomposition_bootstrap_summary.csv", index=False)
    if not draw_frame.empty:
        draw_frame.to_csv(args.output_dir / "channel_decomposition_bootstrap_draws.csv", index=False)
    practical_frame.to_csv(args.output_dir / "practical_top_thresholds.csv", index=False)
    calibration_frame.to_csv(args.output_dir / "calibration_decile_table.csv", index=False)
    context_validation.to_csv(args.output_dir / "context_validation_by_family.csv", index=False)
    family_audit.to_csv(args.output_dir / "excluded_family_audit.csv", index=False)

    summary = {
        "main_nonfinancial_rows": int(panel.shape[0]),
        "main_nonfinancial_events": int(panel["target_default_next_1y"].sum()),
        "test_rows": int(panel[panel["split"].eq("test")].shape[0]),
        "test_events": int(panel.loc[panel["split"].eq("test"), "target_default_next_1y"].sum()),
        "bootstrap_reps_requested": int(args.bootstrap_reps),
        "outputs": {
            "channel_decomposition_results": str(args.output_dir / "channel_decomposition_results.csv"),
            "channel_decomposition_bootstrap_summary": str(args.output_dir / "channel_decomposition_bootstrap_summary.csv"),
            "practical_top_thresholds": str(args.output_dir / "practical_top_thresholds.csv"),
            "calibration_decile_table": str(args.output_dir / "calibration_decile_table.csv"),
            "excluded_family_audit": str(args.output_dir / "excluded_family_audit.csv"),
            "context_validation_by_family": str(args.output_dir / "context_validation_by_family.csv"),
        },
    }
    (args.output_dir / "final_channel1_robustness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_summary(args.output_dir / "FINAL_CHANNEL1_ROBUSTNESS_FINDINGS.md", result_frame, bootstrap_frame, family_audit, practical_frame)

    print(result_frame.to_string(index=False))
    print("\nBootstrap summary:")
    print(bootstrap_frame.to_string(index=False))
    print("\nExcluded-family audit:")
    print(family_audit.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
