#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
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
CHANNEL1_PRIMARY = [
    "going_concern_any_active",
    "covenant_noncompliance_any_active",
    "forbearance_waiver_any_active",
]
CHANNEL1_NO_GC = [
    "covenant_noncompliance_any_active",
    "forbearance_waiver_any_active",
]
LM_SEPARATE = [
    "lm_negative_item7_per_1000",
    "lm_positive_item7_per_1000",
    "lm_uncertainty_item7_per_1000",
    "lm_litigious_item7_per_1000",
    "lm_strong_modal_item7_per_1000",
    "lm_weak_modal_item7_per_1000",
    "lm_constraining_item7_per_1000",
]
DOC_LENGTH = ["log_item7_words"]
READABILITY = [
    "item7_avg_sentence_words",
    "item7_complex_word_share",
    "item7_fog_index",
]
RISK_FACTOR_LENGTH = ["log_item1a_words"]
CONSTRAINT_ONLY = ["lm_constraining_item7_per_1000"]
MANUAL_STRONG_TEXT = LM_SEPARATE + DOC_LENGTH + READABILITY + RISK_FACTOR_LENGTH

PAIRWISE_COMPARISONS = [
    ("accounting_only", "channel1_primary_flags"),
    ("accounting_only", "channel1_any_active"),
    ("accounting_only", "channel1_count"),
    ("accounting_only", "channel1_no_gc"),
    ("accounting_only", "LM_separate_plus_length"),
    ("accounting_only", "manual_strong_text_benchmark"),
    ("LM_separate_plus_length", "channel1_plus_LM_separate_plus_length"),
    ("manual_strong_text_benchmark", "channel1_plus_manual_strong_text"),
    ("accounting_only", "TFIDF_item7"),
    ("TFIDF_item7", "TFIDF_item7_plus_channel1"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Channel 1 benchmark and leakage audits.")
    parser.add_argument("--analysis-dataset", type=Path, default=Path("Review revision/data_rebuild_v1/output/revised_analysis_dataset.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("Review revision/all_industry_expansion_v1/output/all_industry_manifest.csv"))
    parser.add_argument("--channel1-scores", type=Path, default=Path("Review revision/dictionary_v4_phase1a/output/channel1_primary_sample_firm_year_scores.csv"))
    parser.add_argument("--lm-detail", type=Path, default=Path("Review revision/stable_distress_dictionary_v3/output/scored_sample.csv"))
    parser.add_argument("--text-root", type=Path, default=Path("April 4th"))
    parser.add_argument("--output-dir", type=Path, default=Path("Review revision/dictionary_v4_phase1a/output/benchmark_leakage_audit"))
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--tfidf-max-features", type=int, default=5000)
    parser.add_argument("--tfidf-min-df", type=int, default=20)
    return parser.parse_args()


def top_capture(y_true: np.ndarray, scores: np.ndarray, share: float) -> float:
    positives = int(y_true.sum())
    if positives == 0:
        return float("nan")
    count = max(1, int(np.ceil(len(y_true) * share)))
    order = np.argsort(scores)[::-1][:count]
    return float(y_true[order].sum() / positives)


def ks_statistic(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.max(tpr - fpr))


def precision_recall_at_share(y_true: np.ndarray, scores: np.ndarray, share: float) -> tuple[float, float, int, int]:
    positives = int(y_true.sum())
    if positives == 0:
        return float("nan"), float("nan"), 0, 0
    count = max(1, int(np.ceil(len(y_true) * share)))
    order = np.argsort(scores)[::-1][:count]
    true_positives = int(y_true[order].sum())
    false_positives = int(count - true_positives)
    precision = true_positives / count
    recall = true_positives / positives
    return float(precision), float(recall), true_positives, false_positives


def prediction_metrics(y_true: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    auc = float(roc_auc_score(y_true, scores))
    p1, r1, tp1, fp1 = precision_recall_at_share(y_true, scores, 0.01)
    p5, r5, tp5, fp5 = precision_recall_at_share(y_true, scores, 0.05)
    p10, r10, tp10, fp10 = precision_recall_at_share(y_true, scores, 0.10)
    return {
        "auc": auc,
        "somers_d": 2.0 * auc - 1.0,
        "pr_auc": float(average_precision_score(y_true, scores)),
        "ks": ks_statistic(y_true, scores),
        "brier": float(brier_score_loss(y_true, scores)),
        "top_1pct_capture": r1,
        "top_5pct_capture": r5,
        "top_10pct_capture": r10,
        "top_1pct_precision": p1,
        "top_5pct_precision": p5,
        "top_10pct_precision": p10,
        "top_1pct_tp": tp1,
        "top_5pct_tp": tp5,
        "top_10pct_tp": tp10,
        "top_1pct_fp": fp1,
        "top_5pct_fp": fp5,
        "top_10pct_fp": fp10,
    }


def fit_dense_model(train: pd.DataFrame, test: pd.DataFrame, feature_columns: List[str], target_column: str) -> Dict[str, object]:
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("logit", LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=5000, class_weight="balanced", random_state=2027)),
    ])
    y_train = train[target_column].to_numpy(dtype=int)
    y_test = test[target_column].to_numpy(dtype=int)
    pipeline.fit(train[feature_columns], y_train)
    probabilities = pipeline.predict_proba(test[feature_columns])[:, 1]
    return {"probabilities": probabilities, "iterations": int(pipeline.named_steps["logit"].n_iter_[0]), **prediction_metrics(y_test, probabilities)}


def fit_tfidf_model(
    train: pd.DataFrame,
    test: pd.DataFrame,
    dense_columns: List[str],
    target_column: str,
    max_features: int,
    min_df: int,
) -> Dict[str, object]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 1),
        min_df=min_df,
        max_df=0.80,
        max_features=max_features,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z][a-zA-Z]+\b",
        dtype=np.float32,
    )
    print(f"Fitting TF-IDF benchmark with max_features={max_features}, min_df={min_df}", flush=True)
    x_train_text = vectorizer.fit_transform(train["item7_text"].fillna(""))
    x_test_text = vectorizer.transform(test["item7_text"].fillna(""))

    dense_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler(with_mean=False)),
    ])
    x_train_dense = dense_pipeline.fit_transform(train[dense_columns])
    x_test_dense = dense_pipeline.transform(test[dense_columns])
    x_train = sparse.hstack([sparse.csr_matrix(x_train_dense), x_train_text], format="csr")
    x_test = sparse.hstack([sparse.csr_matrix(x_test_dense), x_test_text], format="csr")

    model = LogisticRegression(C=1.0, penalty="l2", solver="liblinear", max_iter=5000, class_weight="balanced", random_state=2027)
    y_train = train[target_column].to_numpy(dtype=int)
    y_test = test[target_column].to_numpy(dtype=int)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    return {
        "probabilities": probabilities,
        "iterations": int(model.n_iter_[0]),
        "tfidf_vocabulary_size": int(len(vectorizer.vocabulary_)),
        **prediction_metrics(y_test, probabilities),
    }


def firm_cluster_bootstrap(test: pd.DataFrame, base_scores: np.ndarray, aug_scores: np.ndarray, target_column: str, reps: int, seed: int) -> pd.DataFrame:
    groups = [indices.to_numpy() for _, indices in test.reset_index(drop=True).groupby("cik").groups.items()]
    y_all = test[target_column].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    rows = []
    for repetition in range(reps):
        selected = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in selected])
        y_true = y_all[indices]
        if np.unique(y_true).size < 2:
            continue
        base = prediction_metrics(y_true, base_scores[indices])
        aug = prediction_metrics(y_true, aug_scores[indices])
        rows.append({
            "repetition": repetition,
            "delta_auc": aug["auc"] - base["auc"],
            "delta_somers_d": aug["somers_d"] - base["somers_d"],
            "delta_pr_auc": aug["pr_auc"] - base["pr_auc"],
            "delta_ks": aug["ks"] - base["ks"],
            "delta_brier": aug["brier"] - base["brier"],
        })
    return pd.DataFrame(rows)


def summarize_bootstrap(draws: pd.DataFrame, base: Dict[str, object], aug: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {"bootstrap_repetitions_valid": int(draws.shape[0])}
    for metric in ["auc", "somers_d", "pr_auc", "ks", "brier"]:
        observed = float(aug[metric] - base[metric])
        out[f"observed_delta_{metric}"] = observed
        if draws.empty:
            out[f"ci95_low_delta_{metric}"] = float("nan")
            out[f"ci95_high_delta_{metric}"] = float("nan")
            out[f"bootstrap_probability_delta_{metric}_gt_0"] = float("nan")
            continue
        values = draws[f"delta_{metric}"]
        out[f"ci95_low_delta_{metric}"] = float(values.quantile(0.025))
        out[f"ci95_high_delta_{metric}"] = float(values.quantile(0.975))
        if metric == "brier":
            out[f"bootstrap_probability_delta_{metric}_lt_0"] = float((values < 0).mean())
        else:
            out[f"bootstrap_probability_delta_{metric}_gt_0"] = float((values > 0).mean())
    return out


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?", text or ""))


def sentence_count(text: str) -> int:
    parts = [part for part in re.split(r"[.!?]+", text or "") if re.search(r"[A-Za-z]", part)]
    return max(1, len(parts))


@lru_cache(maxsize=100000)
def syllable_count(word: str) -> int:
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def readability_features(text: str) -> dict[str, float]:
    words = re.findall(r"[A-Za-z]+", text or "")
    n_words = len(words)
    n_sentences = sentence_count(text)
    if n_words == 0:
        return {"item7_avg_sentence_words": 0.0, "item7_complex_word_share": 0.0, "item7_fog_index": 0.0}
    complex_words = sum(1 for word in words if syllable_count(word) >= 3)
    avg_sentence = n_words / n_sentences
    complex_share = complex_words / n_words
    fog = 0.4 * (avg_sentence + 100.0 * complex_share)
    return {
        "item7_avg_sentence_words": float(avg_sentence),
        "item7_complex_word_share": float(complex_share),
        "item7_fog_index": float(fog),
    }


def locate_text(relative_path: object, preferred_root: Path) -> Path | None:
    if not isinstance(relative_path, str) or not relative_path or relative_path == "nan":
        return None
    for root in [preferred_root, Path("April 4th"), Path("Review revision/all_industry_expansion_v1/output/streamed_text"), Path(".")]:
        candidate = root / relative_path
        if candidate.exists():
            return candidate
    return None


def read_text(relative_path: object, preferred_root: Path) -> str:
    path = locate_text(relative_path, preferred_root)
    if path is None:
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def build_text_features(frame: pd.DataFrame, manifest: pd.DataFrame, text_root: Path) -> pd.DataFrame:
    keys = ["cik", "fiscal_year"]
    paths = frame[keys].merge(manifest[keys + ["item_1a_relpath", "item_7_relpath"]], on=keys, how="left", validate="one_to_one")
    rows = []
    for index, row in enumerate(paths.itertuples(index=False), start=1):
        if index % 1000 == 0:
            print(f"Built text benchmark features for {index:,}/{len(paths):,} filings", flush=True)
        item7_text = read_text(row.item_7_relpath, text_root)
        item1a_text = read_text(row.item_1a_relpath, text_root)
        item1a_words = word_count(item1a_text)
        read = readability_features(item7_text)
        rows.append({
            "cik": row.cik,
            "fiscal_year": row.fiscal_year,
            "item7_text": item7_text,
            "item1a_word_count": item1a_words,
            "log_item1a_words": math.log1p(item1a_words),
            **read,
        })
    return pd.DataFrame(rows)


def load_panel(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, object]]:
    analysis = pd.read_csv(args.analysis_dataset, dtype={"cik": str}, low_memory=False)
    analysis = analysis[analysis["primary_1y_text_sample"].eq(1) & analysis["usable_text_flag"].eq(1)].copy()
    if "log_text_chars" not in analysis.columns:
        analysis["log_text_chars"] = np.log(analysis["usable_text_total_chars"].clip(lower=1))
    channel1 = pd.read_csv(args.channel1_scores, dtype={"cik": str}, low_memory=False)
    lm_detail = pd.read_csv(args.lm_detail, dtype={"cik": str}, low_memory=False)
    manifest = pd.read_csv(args.manifest, dtype={"cik": str, "item_1a_relpath": str, "item_7_relpath": str}, low_memory=False)

    lm_columns = ["cik", "fiscal_year", "item7_word_count", "log_item7_words"] + LM_SEPARATE
    frame = analysis.merge(channel1, on=["cik", "fiscal_year"], how="left", validate="one_to_one")
    frame = frame.merge(lm_detail[lm_columns], on=["cik", "fiscal_year"], how="left", validate="one_to_one")
    for column in frame.columns:
        if column.endswith("_count") or column.endswith("_any_active"):
            frame[column] = frame[column].fillna(0)
    text_features = build_text_features(frame, manifest, args.text_root)
    frame = frame.merge(text_features, on=["cik", "fiscal_year"], how="left", validate="one_to_one")
    metadata = {
        "loaded_rows": int(frame.shape[0]),
        "loaded_events_1y": int(frame["target_default_next_1y"].sum()),
        "item7_text_missing_or_empty": int(frame["item7_text"].fillna("").eq("").sum()),
        "item1a_zero_word_rows": int(frame["item1a_word_count"].fillna(0).eq(0).sum()),
    }
    return frame, metadata


def evaluate_specification(
    sample: pd.DataFrame,
    specification: str,
    target_column: str,
    variants: Dict[str, List[str]],
    tfidf_variants: Dict[str, List[str]],
    args: argparse.Namespace,
) -> tuple[list[Dict[str, object]], list[Dict[str, object]], list[pd.DataFrame]]:
    train = sample[sample["split"].eq("train")].copy()
    test = sample[sample["split"].eq("test")].copy().reset_index(drop=True)
    if train[target_column].sum() == 0 or test[target_column].sum() == 0:
        return [], [], []
    y_test = test[target_column].to_numpy(dtype=int)
    result_rows: list[Dict[str, object]] = []
    model_outputs: Dict[str, Dict[str, object]] = {}
    draw_frames: list[pd.DataFrame] = []

    all_variants = {"accounting_only": []}
    all_variants.update(variants)
    for variant, text_columns in all_variants.items():
        features = PRIMARY_ACCOUNTING + text_columns
        model = fit_dense_model(train, test, features, target_column)
        model_outputs[variant] = model
        result_rows.append({
            "specification": specification,
            "target_column": target_column,
            "text_variant": variant,
            "feature_count": len(features),
            "train_rows": int(train.shape[0]),
            "train_firms": int(train["cik"].nunique()),
            "train_events": int(train[target_column].sum()),
            "test_rows": int(test.shape[0]),
            "test_firms": int(test["cik"].nunique()),
            "test_events": int(test[target_column].sum()),
            "test_event_rate": float(test[target_column].mean()),
            "tfidf_vocabulary_size": np.nan,
            **{k: v for k, v in model.items() if k != "probabilities"},
        })

    for variant, dense_columns in tfidf_variants.items():
        model = fit_tfidf_model(train, test, PRIMARY_ACCOUNTING + dense_columns, target_column, args.tfidf_max_features, args.tfidf_min_df)
        model_outputs[variant] = model
        result_rows.append({
            "specification": specification,
            "target_column": target_column,
            "text_variant": variant,
            "feature_count": len(PRIMARY_ACCOUNTING + dense_columns),
            "train_rows": int(train.shape[0]),
            "train_firms": int(train["cik"].nunique()),
            "train_events": int(train[target_column].sum()),
            "test_rows": int(test.shape[0]),
            "test_firms": int(test["cik"].nunique()),
            "test_events": int(test[target_column].sum()),
            "test_event_rate": float(test[target_column].mean()),
            **{k: v for k, v in model.items() if k != "probabilities"},
        })

    bootstrap_rows: list[Dict[str, object]] = []
    for comparison_index, (base_name, aug_name) in enumerate(PAIRWISE_COMPARISONS, start=1):
        if base_name not in model_outputs or aug_name not in model_outputs:
            continue
        base = model_outputs[base_name]
        aug = model_outputs[aug_name]
        draws = firm_cluster_bootstrap(test, base["probabilities"], aug["probabilities"], target_column, args.bootstrap_reps, seed=3027 + comparison_index)
        if not draws.empty:
            draw_copy = draws.copy()
            draw_copy.insert(0, "augmented_variant", aug_name)
            draw_copy.insert(0, "baseline_variant", base_name)
            draw_copy.insert(0, "target_column", target_column)
            draw_copy.insert(0, "specification", specification)
            draw_frames.append(draw_copy)
        bootstrap_rows.append({
            "specification": specification,
            "target_column": target_column,
            "baseline_variant": base_name,
            "augmented_variant": aug_name,
            **summarize_bootstrap(draws, base, aug),
        })
    return result_rows, bootstrap_rows, draw_frames


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame, metadata = load_panel(args)
    main_panel = frame[frame["financial_firm_flag"].eq(0)].copy()

    variants = {
        "document_length_only": DOC_LENGTH,
        "risk_factor_length_only": RISK_FACTOR_LENGTH,
        "readability_only": READABILITY,
        "financial_constraint_only": CONSTRAINT_ONLY,
        "LM_separate": LM_SEPARATE,
        "LM_separate_plus_length": LM_SEPARATE + DOC_LENGTH,
        "manual_strong_text_benchmark": MANUAL_STRONG_TEXT,
        "channel1_primary_flags": CHANNEL1_PRIMARY,
        "channel1_any_active": ["channel1_primary_any_active"],
        "channel1_count": ["channel1_primary_active_count"],
        "channel1_no_gc": CHANNEL1_NO_GC,
        "channel1_plus_LM_separate_plus_length": CHANNEL1_PRIMARY + LM_SEPARATE + DOC_LENGTH,
        "channel1_plus_manual_strong_text": CHANNEL1_PRIMARY + MANUAL_STRONG_TEXT,
    }
    tfidf_variants = {
        "TFIDF_item7": [],
        "TFIDF_item7_plus_channel1": CHANNEL1_PRIMARY,
    }

    all_results: list[Dict[str, object]] = []
    all_bootstraps: list[Dict[str, object]] = []
    all_draw_frames: list[pd.DataFrame] = []

    base_sample = main_panel.dropna(subset=["target_default_next_1y"])
    results, bootstraps, draws = evaluate_specification(
        base_sample,
        "benchmark_primary_nonfinancial",
        "target_default_next_1y",
        variants,
        tfidf_variants,
        args,
    )
    all_results.extend(results)
    all_bootstraps.extend(bootstraps)
    all_draw_frames.extend(draws)

    for days in [30, 60, 90, 180]:
        near_flag = f"default_within_{days}d_flag"
        target = "target_default_next_1y"
        leakage_sample = main_panel[~main_panel[near_flag].eq(1)].dropna(subset=[target]).copy()
        leakage_variants = {
            "LM_separate_plus_length": LM_SEPARATE + DOC_LENGTH,
            "manual_strong_text_benchmark": MANUAL_STRONG_TEXT,
            "channel1_primary_flags": CHANNEL1_PRIMARY,
            "channel1_no_gc": CHANNEL1_NO_GC,
            "channel1_plus_LM_separate_plus_length": CHANNEL1_PRIMARY + LM_SEPARATE + DOC_LENGTH,
            "channel1_plus_manual_strong_text": CHANNEL1_PRIMARY + MANUAL_STRONG_TEXT,
        }
        results, bootstraps, draws = evaluate_specification(
            leakage_sample,
            f"leakage_exclude_defaults_within_{days}d",
            target,
            leakage_variants,
            {},
            args,
        )
        all_results.extend(results)
        all_bootstraps.extend(bootstraps)
        all_draw_frames.extend(draws)

    result_frame = pd.DataFrame(all_results)
    bootstrap_frame = pd.DataFrame(all_bootstraps)
    result_frame.to_csv(args.output_dir / "benchmark_leakage_model_results.csv", index=False)
    bootstrap_frame.to_csv(args.output_dir / "benchmark_leakage_pairwise_bootstrap_summary.csv", index=False)
    if all_draw_frames:
        pd.concat(all_draw_frames, ignore_index=True).to_csv(args.output_dir / "benchmark_leakage_pairwise_bootstrap_draws.csv", index=False)

    feature_audit_columns = ["cik", "fiscal_year", "split", "target_default_next_1y", "financial_firm_flag", "item1a_word_count", "log_item1a_words"] + READABILITY + CHANNEL1_PRIMARY + ["channel1_primary_any_active", "channel1_primary_active_count"] + LM_SEPARATE + DOC_LENGTH
    main_panel[feature_audit_columns].to_csv(args.output_dir / "benchmark_feature_audit_panel.csv", index=False)

    summary = {
        "metadata": metadata,
        "main_nonfinancial_rows": int(main_panel.shape[0]),
        "main_nonfinancial_events": int(main_panel["target_default_next_1y"].sum()),
        "main_nonfinancial_test_rows": int(main_panel[main_panel["split"].eq("test")].shape[0]),
        "main_nonfinancial_test_events": int(main_panel.loc[main_panel["split"].eq("test"), "target_default_next_1y"].sum()),
        "outputs": {
            "model_results": str(args.output_dir / "benchmark_leakage_model_results.csv"),
            "bootstrap_summary": str(args.output_dir / "benchmark_leakage_pairwise_bootstrap_summary.csv"),
            "feature_audit_panel": str(args.output_dir / "benchmark_feature_audit_panel.csv"),
        },
    }
    (args.output_dir / "benchmark_leakage_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(result_frame.to_string(index=False))
    print("\nPairwise bootstrap summary:")
    print(bootstrap_frame.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
