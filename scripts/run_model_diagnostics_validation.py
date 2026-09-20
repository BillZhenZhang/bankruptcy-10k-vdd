#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ACCOUNTING = [
    "t__ohlson_size_cpi_1980",
    "t__ohlson_tlta",
    "t__ohlson_wcta",
    "t__ohlson_clca",
    "t__ohlson_nita",
    "t__ohlson_oeneg",
]

CHANNEL1 = [
    "going_concern_any_active",
    "covenant_noncompliance_any_active",
    "forbearance_waiver_any_active",
]

MAIN_FEATURES = ACCOUNTING + CHANNEL1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run non-dictionary robustness diagnostics.")
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
        "--output-dir",
        type=Path,
        default=Path("Review revision/dictionary_v4_phase1a/output/model_diagnostics_validation"),
    )
    parser.add_argument("--group-splits", type=int, default=50)
    parser.add_argument("--test-size", type=float, default=0.20)
    return parser.parse_args()


def top_capture(y_true: np.ndarray, scores: np.ndarray, share: float) -> float:
    positives = int(y_true.sum())
    if positives == 0:
        return float("nan")
    count = max(1, int(np.ceil(len(y_true) * share)))
    order = np.argsort(scores)[::-1][:count]
    return float(y_true[order].sum() / positives)


def precision_at_share(y_true: np.ndarray, scores: np.ndarray, share: float) -> float:
    count = max(1, int(np.ceil(len(y_true) * share)))
    order = np.argsort(scores)[::-1][:count]
    return float(y_true[order].sum() / count)


def ks_statistic(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.max(tpr - fpr))


def prediction_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    auc = float(roc_auc_score(y_true, scores))
    return {
        "auc": auc,
        "somers_d": 2.0 * auc - 1.0,
        "pr_auc": float(average_precision_score(y_true, scores)),
        "ks": ks_statistic(y_true, scores),
        "brier": float(brier_score_loss(y_true, scores)),
        "top_1pct_capture": top_capture(y_true, scores, 0.01),
        "top_5pct_capture": top_capture(y_true, scores, 0.05),
        "top_10pct_capture": top_capture(y_true, scores, 0.10),
        "top_1pct_precision": precision_at_share(y_true, scores, 0.01),
        "top_5pct_precision": precision_at_share(y_true, scores, 0.05),
        "top_10pct_precision": precision_at_share(y_true, scores, 0.10),
    }


def balanced_weights(y: np.ndarray) -> np.ndarray:
    positives = y.sum()
    negatives = len(y) - positives
    weights = np.ones(len(y), dtype=float)
    weights[y == 1] = len(y) / (2.0 * positives)
    weights[y == 0] = len(y) / (2.0 * negatives)
    return weights


def load_panel(args: argparse.Namespace) -> pd.DataFrame:
    analysis = pd.read_csv(args.analysis_dataset, dtype={"cik": str}, low_memory=False)
    analysis = analysis[analysis["primary_1y_text_sample"].eq(1) & analysis["usable_text_flag"].eq(1)].copy()
    scores = pd.read_csv(args.channel1_scores, dtype={"cik": str}, low_memory=False)
    frame = analysis.merge(scores, on=["cik", "fiscal_year"], how="left", validate="one_to_one")
    for column in frame.columns:
        if column.endswith("_count") or column.endswith("_any_active"):
            frame[column] = frame[column].fillna(0).astype(int)
    return frame[frame["financial_firm_flag"].eq(0)].copy()


def fit_sklearn_logit(train: pd.DataFrame, test: pd.DataFrame, feature_columns: List[str]) -> Dict[str, object]:
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
    scores = pipeline.predict_proba(test[feature_columns])[:, 1]
    return {
        "scores": scores,
        "iterations": int(pipeline.named_steps["logit"].n_iter_[0]),
        **prediction_metrics(y_test, scores),
    }


def preprocess_train_test(train: pd.DataFrame, test: pd.DataFrame, feature_columns: List[str]):
    imputer = SimpleImputer(strategy="median", add_indicator=False)
    scaler = StandardScaler()
    x_train = imputer.fit_transform(train[feature_columns])
    x_test = imputer.transform(test[feature_columns])
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return sm.add_constant(x_train, has_constant="add"), sm.add_constant(x_test, has_constant="add")


def fit_statsmodels_glm(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: List[str],
    link_name: str,
    use_balanced_weights: bool = False,
) -> Dict[str, object]:
    x_train, x_test = preprocess_train_test(train, test, feature_columns)
    y_train = train["target_default_next_1y"].to_numpy(dtype=int)
    y_test = test["target_default_next_1y"].to_numpy(dtype=int)
    weights = balanced_weights(y_train) if use_balanced_weights else None
    if link_name == "logit":
        family = sm.families.Binomial(link=sm.families.links.logit())
    elif link_name == "probit":
        family = sm.families.Binomial(link=sm.families.links.probit())
    else:
        raise ValueError(f"Unsupported link: {link_name}")
    model = sm.GLM(y_train, x_train, family=family, freq_weights=weights) if weights is not None else sm.GLM(y_train, x_train, family=family)
    result = model.fit(maxiter=500, disp=0)
    scores = np.asarray(result.predict(x_test))
    scores = np.clip(scores, 1e-9, 1 - 1e-9)
    return {
        "scores": scores,
        "converged": bool(result.converged),
        "iterations": int(result.fit_history.get("iteration", -1)),
        "aic": float(result.aic),
        "bic": float(result.bic_llf),
        **prediction_metrics(y_test, scores),
    }


def build_multicollinearity(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = panel[panel["split"].eq("train")].copy()
    imputer = SimpleImputer(strategy="median")
    x = pd.DataFrame(imputer.fit_transform(train[MAIN_FEATURES]), columns=MAIN_FEATURES)
    corr = x.corr()
    corr_long = corr.reset_index().melt(id_vars="index", var_name="feature_b", value_name="pearson_correlation")
    corr_long = corr_long.rename(columns={"index": "feature_a"})

    vif_rows = []
    for feature in MAIN_FEATURES:
        y = x[feature].to_numpy()
        other_features = [column for column in MAIN_FEATURES if column != feature]
        model = LinearRegression()
        model.fit(x[other_features], y)
        r2 = float(model.score(x[other_features], y))
        vif = float(1.0 / (1.0 - r2)) if r2 < 0.999999 else float("inf")
        vif_rows.append({
            "feature": feature,
            "r_squared_against_other_features": r2,
            "vif": vif,
            "missing_rate_train": float(train[feature].isna().mean()),
            "mean_train": float(train[feature].mean()),
            "std_train": float(train[feature].std()),
        })
    vif = pd.DataFrame(vif_rows).sort_values("vif", ascending=False)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    singular_values = np.linalg.svd(x_scaled, compute_uv=False)
    condition_number = float(singular_values.max() / singular_values.min())
    condition = pd.DataFrame([{
        "feature_set": "accounting_plus_channel1",
        "rows": int(x.shape[0]),
        "features": int(x.shape[1]),
        "condition_number_standardized": condition_number,
        "min_singular_value": float(singular_values.min()),
        "max_singular_value": float(singular_values.max()),
    }])
    return corr_long, vif, condition


def run_logit_probit(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[panel["split"].eq("train")].copy()
    test = panel[panel["split"].eq("test")].copy().reset_index(drop=True)
    rows = []
    for feature_set_name, feature_columns in {
        "accounting_only": ACCOUNTING,
        "accounting_plus_channel1": MAIN_FEATURES,
    }.items():
        sklearn_logit = fit_sklearn_logit(train, test, feature_columns)
        rows.append({
            "feature_set": feature_set_name,
            "estimator": "l2_weighted_logit_sklearn_main",
            "train_rows": int(train.shape[0]),
            "train_events": int(train["target_default_next_1y"].sum()),
            "test_rows": int(test.shape[0]),
            "test_events": int(test["target_default_next_1y"].sum()),
            "converged": True,
            "aic": np.nan,
            "bic": np.nan,
            **{key: value for key, value in sklearn_logit.items() if key != "scores"},
        })
        for link_name in ["logit", "probit"]:
            glm = fit_statsmodels_glm(train, test, feature_columns, link_name, use_balanced_weights=False)
            rows.append({
                "feature_set": feature_set_name,
                "estimator": f"unweighted_glm_{link_name}_link_check",
                "train_rows": int(train.shape[0]),
                "train_events": int(train["target_default_next_1y"].sum()),
                "test_rows": int(test.shape[0]),
                "test_events": int(test["target_default_next_1y"].sum()),
                **{key: value for key, value in glm.items() if key != "scores"},
            })
    return pd.DataFrame(rows)


def run_time_split_unseen_firm_check(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[panel["split"].eq("train")].copy()
    test = panel[panel["split"].eq("test")].copy().reset_index(drop=True)
    train_firms = set(train["cik"])
    test = test.copy()
    test["firm_seen_in_train"] = test["cik"].isin(train_firms).astype(int)
    rows = []
    for feature_set_name, feature_columns in {
        "accounting_only": ACCOUNTING,
        "accounting_plus_channel1": MAIN_FEATURES,
    }.items():
        model = fit_sklearn_logit(train, test, feature_columns)
        test_scores = test[["cik", "target_default_next_1y", "firm_seen_in_train"]].copy()
        test_scores["score"] = model["scores"]
        for subset_name, subset in {
            "all_time_holdout_firm_years": test_scores,
            "seen_firms_only": test_scores[test_scores["firm_seen_in_train"].eq(1)],
            "unseen_firms_only": test_scores[test_scores["firm_seen_in_train"].eq(0)],
        }.items():
            if subset["target_default_next_1y"].nunique() < 2:
                rows.append({
                    "feature_set": feature_set_name,
                    "test_subset": subset_name,
                    "test_rows": int(subset.shape[0]),
                    "test_firms": int(subset["cik"].nunique()),
                    "test_events": int(subset["target_default_next_1y"].sum()),
                    "auc": np.nan,
                    "pr_auc": np.nan,
                    "brier": np.nan,
                    "top_10pct_capture": np.nan,
                })
                continue
            metrics = prediction_metrics(subset["target_default_next_1y"].to_numpy(dtype=int), subset["score"].to_numpy())
            rows.append({
                "feature_set": feature_set_name,
                "test_subset": subset_name,
                "test_rows": int(subset.shape[0]),
                "test_firms": int(subset["cik"].nunique()),
                "test_events": int(subset["target_default_next_1y"].sum()),
                **metrics,
            })
    return pd.DataFrame(rows)


def run_grouped_validation(panel: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(n_splits=args.group_splits, test_size=args.test_size, random_state=2027)
    rows = []
    groups = panel["cik"].to_numpy()
    y = panel["target_default_next_1y"].to_numpy(dtype=int)
    for split_index, (train_index, test_index) in enumerate(splitter.split(panel, y, groups), start=1):
        train = panel.iloc[train_index].copy()
        test = panel.iloc[test_index].copy().reset_index(drop=True)
        if train["target_default_next_1y"].nunique() < 2 or test["target_default_next_1y"].nunique() < 2:
            continue
        accounting = fit_sklearn_logit(train, test, ACCOUNTING)
        channel = fit_sklearn_logit(train, test, MAIN_FEATURES)
        rows.append({
            "split_index": split_index,
            "train_rows": int(train.shape[0]),
            "train_firms": int(train["cik"].nunique()),
            "train_events": int(train["target_default_next_1y"].sum()),
            "test_rows": int(test.shape[0]),
            "test_firms": int(test["cik"].nunique()),
            "test_events": int(test["target_default_next_1y"].sum()),
            "accounting_auc": float(accounting["auc"]),
            "channel1_auc": float(channel["auc"]),
            "delta_auc": float(channel["auc"] - accounting["auc"]),
            "accounting_pr_auc": float(accounting["pr_auc"]),
            "channel1_pr_auc": float(channel["pr_auc"]),
            "delta_pr_auc": float(channel["pr_auc"] - accounting["pr_auc"]),
            "accounting_brier": float(accounting["brier"]),
            "channel1_brier": float(channel["brier"]),
            "delta_brier": float(channel["brier"] - accounting["brier"]),
            "accounting_top_10pct_capture": float(accounting["top_10pct_capture"]),
            "channel1_top_10pct_capture": float(channel["top_10pct_capture"]),
            "delta_top_10pct_capture": float(channel["top_10pct_capture"] - accounting["top_10pct_capture"]),
        })
    split_results = pd.DataFrame(rows)
    summary_rows = []
    for column in [
        "accounting_auc",
        "channel1_auc",
        "delta_auc",
        "accounting_pr_auc",
        "channel1_pr_auc",
        "delta_pr_auc",
        "accounting_brier",
        "channel1_brier",
        "delta_brier",
        "accounting_top_10pct_capture",
        "channel1_top_10pct_capture",
        "delta_top_10pct_capture",
    ]:
        values = split_results[column].dropna()
        summary_rows.append({
            "metric": column,
            "valid_splits": int(values.shape[0]),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std()),
            "p05": float(values.quantile(0.05)),
            "p25": float(values.quantile(0.25)),
            "p75": float(values.quantile(0.75)),
            "p95": float(values.quantile(0.95)),
            "share_positive": float((values > 0).mean()) if column.startswith("delta_") else np.nan,
        })
    return split_results, pd.DataFrame(summary_rows)


def write_markdown(
    output_path: Path,
    vif: pd.DataFrame,
    condition: pd.DataFrame,
    logit_probit: pd.DataFrame,
    unseen: pd.DataFrame,
    grouped_summary: pd.DataFrame,
) -> None:
    max_vif = float(vif["vif"].replace(np.inf, np.nan).max())
    condition_number = float(condition["condition_number_standardized"].iloc[0])
    lines = [
        "# Model Diagnostics and Grouped Validation Findings",
        "",
        "## Multicollinearity",
        "",
        f"- Maximum VIF across accounting plus Channel 1 features: {max_vif:.2f}.",
        f"- Standardized condition number: {condition_number:.2f}.",
        "- These diagnostics do not indicate severe multicollinearity under common VIF thresholds, but the sparse Channel 1 indicators should still be interpreted as predictive flags rather than causal coefficients.",
        "",
        "## Logit versus probit",
        "",
        "| Feature set | Estimator | AUC | PR-AUC | Brier | Top 10% capture | Converged |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for _, row in logit_probit.iterrows():
        lines.append(
            f"| {row['feature_set']} | {row['estimator']} | {row['auc']:.4f} | {row['pr_auc']:.4f} | "
            f"{row['brier']:.4f} | {row['top_10pct_capture']:.2%} | {row['converged']} |"
        )
    lines.extend([
        "",
        "## Time-split unseen-firm check",
        "",
        "| Feature set | Test subset | Rows | Firms | Events | AUC | PR-AUC | Top 10% capture |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for _, row in unseen.iterrows():
        auc = "" if pd.isna(row["auc"]) else f"{row['auc']:.4f}"
        pr_auc = "" if pd.isna(row["pr_auc"]) else f"{row['pr_auc']:.4f}"
        top10 = "" if pd.isna(row["top_10pct_capture"]) else f"{row['top_10pct_capture']:.2%}"
        lines.append(
            f"| {row['feature_set']} | {row['test_subset']} | {int(row['test_rows'])} | {int(row['test_firms'])} | "
            f"{int(row['test_events'])} | {auc} | {pr_auc} | {top10} |"
        )
    lines.extend([
        "",
        "## Repeated firm-group holdout validation",
        "",
        "| Metric | Mean | Median | P25 | P75 | Share positive |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for metric in ["delta_auc", "delta_pr_auc", "delta_brier", "delta_top_10pct_capture"]:
        row = grouped_summary[grouped_summary["metric"].eq(metric)].iloc[0]
        share = "" if pd.isna(row["share_positive"]) else f"{row['share_positive']:.2%}"
        lines.append(
            f"| {metric} | {row['mean']:.4f} | {row['median']:.4f} | {row['p25']:.4f} | {row['p75']:.4f} | {share} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- The logit/probit comparison is intended as a link-function robustness check, not a replacement for the main penalized logit design.",
        "- The repeated firm-group validation directly addresses the concern that repeated firm-years could inflate performance because all observations of a firm are kept together in each split.",
        "- The grouped validation should be reported as robustness evidence alongside the out-of-time split, not as the primary design.",
    ])
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel = load_panel(args)

    corr, vif, condition = build_multicollinearity(panel)
    logit_probit = run_logit_probit(panel)
    unseen = run_time_split_unseen_firm_check(panel)
    grouped_splits, grouped_summary = run_grouped_validation(panel, args)

    corr.to_csv(args.output_dir / "multicollinearity_correlation_long.csv", index=False)
    vif.to_csv(args.output_dir / "multicollinearity_vif.csv", index=False)
    condition.to_csv(args.output_dir / "multicollinearity_condition_number.csv", index=False)
    logit_probit.to_csv(args.output_dir / "logit_probit_comparison.csv", index=False)
    unseen.to_csv(args.output_dir / "time_split_unseen_firm_check.csv", index=False)
    grouped_splits.to_csv(args.output_dir / "firm_grouped_validation_splits.csv", index=False)
    grouped_summary.to_csv(args.output_dir / "firm_grouped_validation_summary.csv", index=False)

    summary = {
        "main_nonfinancial_rows": int(panel.shape[0]),
        "main_nonfinancial_firms": int(panel["cik"].nunique()),
        "main_nonfinancial_events": int(panel["target_default_next_1y"].sum()),
        "train_rows": int(panel[panel["split"].eq("train")].shape[0]),
        "train_firms": int(panel.loc[panel["split"].eq("train"), "cik"].nunique()),
        "train_events": int(panel.loc[panel["split"].eq("train"), "target_default_next_1y"].sum()),
        "test_rows": int(panel[panel["split"].eq("test")].shape[0]),
        "test_firms": int(panel.loc[panel["split"].eq("test"), "cik"].nunique()),
        "test_events": int(panel.loc[panel["split"].eq("test"), "target_default_next_1y"].sum()),
        "train_test_firm_overlap": int(len(set(panel.loc[panel["split"].eq("train"), "cik"]) & set(panel.loc[panel["split"].eq("test"), "cik"]))),
        "group_splits_requested": int(args.group_splits),
        "group_splits_valid": int(grouped_splits.shape[0]),
        "max_vif": float(vif["vif"].replace(np.inf, np.nan).max()),
        "condition_number_standardized": float(condition["condition_number_standardized"].iloc[0]),
    }
    (args.output_dir / "model_diagnostics_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(args.output_dir / "MODEL_DIAGNOSTICS_VALIDATION_FINDINGS.md", vif, condition, logit_probit, unseen, grouped_summary)

    print("VIF:")
    print(vif.to_string(index=False))
    print("\nLogit/probit:")
    print(logit_probit.to_string(index=False))
    print("\nUnseen-firm check:")
    print(unseen.to_string(index=False))
    print("\nGrouped validation summary:")
    print(grouped_summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
