#!/usr/bin/env python3
"""Build an unsampled all-industry SEC company roster."""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import random
import subprocess
import sys
import time
import threading
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List


OUTPUT_COLUMNS = [
    "cik", "companyfacts_file", "submission_file", "name", "sic", "sic_major_group",
    "sic_description", "tickers", "exchanges", "entity_type", "category", "fiscal_year_end",
    "state_of_incorporation", "ucla_default_cik_flag", "sampling_group",
]

_TEXT_WORKER_LM = None
_TEXT_WORKER_PB = None
_TEXT_WORKER_NEGATIVE: set[str] = set()
_TEXT_WORKER_UNCERTAINTY: set[str] = set()
_TEXT_WORKER_PB_CONFIG: Dict[str, object] = {}
_TEXT_WORKER_CACHE_ROOT = Path("April 4th")
_TEXT_WORKER_SECONDARY_ROOT = Path("Review revision/all_industry_expansion_v1/output/streamed_text")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--defaults-csv", type=Path, default=Path("Default_Record/Florida-UCLA-LoPucki Bankruptcy Research Database 1-12-2023.csv"))
    parser.add_argument("--submissions-dir", type=Path, default=Path("submissions"))
    parser.add_argument("--companyfacts-dir", type=Path, default=Path("companyfacts"))
    parser.add_argument("--output-csv", type=Path, default=Path("Review revision/all_industry_expansion_v1/output/all_industry_companies.csv"))
    parser.add_argument("--industry-audit-csv", type=Path, default=Path("Review revision/all_industry_expansion_v1/output/all_industry_company_audit.csv"))
    parser.add_argument("--metadata-json", type=Path, default=Path("Review revision/all_industry_expansion_v1/output/all_industry_companies_summary.json"))
    parser.add_argument("--prepare-targets", action="store_true")
    parser.add_argument("--build-manifest", action="store_true")
    parser.add_argument("--build-financial-eligibility", action="store_true")
    parser.add_argument("--score-all-text", action="store_true")
    parser.add_argument("--evaluate-text", action="store_true")
    parser.add_argument("--recalibrate-lm-label-tokens", action="store_true")
    parser.add_argument("--max-network-filings", type=int, default=0)
    parser.add_argument("--requests-per-second", type=float, default=2.0)
    parser.add_argument("--user-agent", default="Zhen Zhang zz8sp@virginia.edu")
    parser.add_argument("--financials-csv", type=Path, default=Path("Review revision/all_industry_expansion_v1/output/all_industry_financials_by_year.csv"))
    parser.add_argument("--financials-with-defaults-csv", type=Path, default=Path("Review revision/all_industry_expansion_v1/output/all_industry_financials_with_defaults.csv"))
    parser.add_argument("--target-years-csv", type=Path, default=Path("Review revision/all_industry_expansion_v1/output/all_industry_target_years.csv"))
    parser.add_argument("--manifest-csv", type=Path, default=Path("Review revision/all_industry_expansion_v1/output/all_industry_manifest.csv"))
    return parser.parse_args()


def load_source_module() -> object:
    path = Path("April 4th/build_april4_expanded_companies.py")
    spec = importlib.util.spec_from_file_location("april4_company_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_default_module() -> object:
    path = Path("merge_defaults_into_financials.py")
    spec = importlib.util.spec_from_file_location("default_merger", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_manifest_module() -> object:
    path = Path("build_10k_manifest_labeled_window.py")
    spec = importlib.util.spec_from_file_location("manifest_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_module(path: Path, name: str) -> object:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_company_allow_missing_sic(source: object, path: Path) -> Dict[str, str] | None:
    parsed = source.parse_submission_company(path)
    if parsed is not None:
        return parsed
    cik_digits = "".join(character for character in path.stem if character.isdigit())
    if not cik_digits:
        return None
    cik = str(int(cik_digits)).zfill(10)
    return {
        "cik": cik,
        "companyfacts_file": f"CIK{int(cik):010d}.json",
        "submission_file": path.name,
        "name": "",
        "sic": "",
        "sic_major_group": "",
        "sic_description": "",
        "tickers": "",
        "exchanges": "",
        "entity_type": "",
        "category": "",
        "fiscal_year_end": "",
        "state_of_incorporation": "",
    }


def attach_defaults_and_build_targets(args: argparse.Namespace) -> Dict[str, int]:
    if not args.financials_csv.exists():
        raise FileNotFoundError(args.financials_csv)
    merger = load_default_module()
    with args.defaults_csv.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        default_rows = list(csv.DictReader(handle))
    defaults_by_cik = defaultdict(list)
    defaults_by_name = defaultdict(list)
    for row in default_rows:
        ciks = {
            merger.normalize_cik(row.get("CikBefore", "")),
            merger.normalize_cik(row.get("CikEmerging", "")),
        }
        for cik in ciks - {""}:
            defaults_by_cik[cik].append(row)
        name = merger.normalize_name(row.get("NameCorp", ""))
        if name:
            defaults_by_name[name].append(row)
    output_rows = []
    target_rows = []
    match_counts = Counter()
    with args.financials_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        input_columns = list(reader.fieldnames or [])
        for row in reader:
            cik = merger.normalize_cik(row.get("cik", ""))
            candidates = defaults_by_cik.get(cik, [])
            match_method = "cik" if candidates else ""
            if not candidates:
                name = merger.normalize_name(row.get("name", ""))
                name_matches = defaults_by_name.get(name, []) if name else []
                if len(name_matches) == 1:
                    candidates = name_matches
                    match_method = "name_exact_unique"
            out = dict(row)
            out["default_flag"] = "1" if candidates else "0"
            out["default_date"] = merger.compute_default_date(candidates) if candidates else ""
            out["default_match_method"] = match_method
            out["brd_case_count"] = str(len(candidates))
            output_rows.append(out)
            match_counts[match_method or "unmatched"] += 1
            year_text = (row.get("fiscal_year") or "").strip()
            if cik and year_text.isdigit() and 2010 <= int(year_text) <= 2021:
                year = int(year_text)
                target_rows.append({"cik": cik, "fiscal_year": year, "split": "test" if year >= 2019 else "train"})
    args.financials_with_defaults_csv.parent.mkdir(parents=True, exist_ok=True)
    output_columns = input_columns + ["default_flag", "default_date", "default_match_method", "brd_case_count"]
    with args.financials_with_defaults_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_columns)
        writer.writeheader()
        writer.writerows(output_rows)
    unique_targets = {(row["cik"], row["fiscal_year"]): row for row in target_rows}
    target_rows = sorted(unique_targets.values(), key=lambda row: (row["cik"], row["fiscal_year"]))
    with args.target_years_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cik", "fiscal_year", "split"])
        writer.writeheader()
        writer.writerows(target_rows)
    return {
        "financial_rows_with_defaults": len(output_rows),
        "target_year_rows": len(target_rows),
        "target_firms": len({row["cik"] for row in target_rows}),
        "cik_matched_rows": match_counts["cik"],
        "unique_name_matched_rows": match_counts["name_exact_unique"],
    }


def build_manifest(args: argparse.Namespace) -> Dict[str, int]:
    if not args.target_years_csv.exists():
        raise FileNotFoundError(args.target_years_csv)
    manifest_builder = load_manifest_module()
    targets = manifest_builder.load_target_pairs(args.target_years_csv)
    output_rows = []
    status_counts = Counter()
    for cik, year_map in sorted(targets.items()):
        filings, submission_file = manifest_builder.load_all_filings_for_cik(cik, args.submissions_dir)
        for fiscal_year, split in sorted(year_map.items()):
            if not filings:
                status, selected, candidate_count = "missing_submission_file", None, 0
            else:
                status, selected, candidate_count = manifest_builder.select_filing_for_year(
                    filings,
                    fiscal_year=fiscal_year,
                    prefer_amended=False,
                    fallback_fy_plus1_max_month=3,
                )
            output_rows.append(manifest_builder.build_output_row(
                cik=cik,
                fiscal_year=fiscal_year,
                split=split,
                selection_status=status,
                selected=selected,
                candidate_count_same_year=candidate_count,
                submission_file=submission_file,
            ))
            status_counts[status] += 1
    output_rows.sort(key=lambda row: (row["cik"], int(row["fiscal_year"])))
    with args.manifest_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_builder.ROW_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)
    return {
        "manifest_rows": len(output_rows),
        "manifest_firms": len(targets),
        "manifest_matched_10k": status_counts["matched_10k"],
        "manifest_matched_10k_fy_plus1_q1": status_counts["matched_10k_fy_plus1_q1"],
        "manifest_missing_10k": status_counts["missing_10k_for_fiscal_year"],
        "manifest_missing_submission": status_counts["missing_submission_file"],
    }


def build_financial_eligibility(args: argparse.Namespace) -> Dict[str, str]:
    command = [
        sys.executable,
        "-u",
        "Review revision/data_rebuild_v1/build_filing_aligned_dataset.py",
        "--financial-features", str(args.financials_with_defaults_csv),
        "--manifest", str(args.manifest_csv),
        "--companyfacts-dir", str(args.companyfacts_dir),
        "--output-dir", "Review revision/all_industry_expansion_v1/output/financial_rebuild",
        "--workers", "10",
        "--skip-text-scores",
        "--include-fy-plus1-q1",
    ]
    subprocess.run(command, check=True)
    return {"financial_eligibility_output": "Review revision/all_industry_expansion_v1/output/financial_rebuild"}


def evaluate_text_models() -> Dict[str, str]:
    command = [
        sys.executable,
        "-u",
        "Review revision/data_rebuild_v1/evaluate_accounting_factor_sets.py",
        "--input-csv", "Review revision/all_industry_expansion_v1/output/financial_rebuild/revised_analysis_dataset.csv",
        "--text-scores", "Review revision/all_industry_expansion_v1/output/all_industry_text_scores.csv",
        "--output-dir", "Review revision/all_industry_expansion_v1/output/model_results",
        "--bootstrap-reps", "2000",
    ]
    subprocess.run(command, check=True)
    return {"model_results_output": "Review revision/all_industry_expansion_v1/output/model_results"}


def percentile(values: List[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def fit_bounds(rows: List[Dict[str, str]], column: str) -> tuple[float, float]:
    train_values = [
        float(row[column]) for row in rows
        if row.get("split") == "train" and row.get(column, "") not in {"", None}
    ]
    low = percentile(train_values, 0.01)
    high = percentile(train_values, 0.99)
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        values = [float(row[column]) for row in rows if row.get(column, "") not in {"", None}]
        low = min(values) if values else 0.0
        high = max(values) if values else 1.0
    if high <= low:
        high = low + 1.0
    return low, high


def format_number(value: float | None) -> str:
    return "" if value is None or not math.isfinite(value) else f"{value:.12g}"


def build_eligible_rows(args: argparse.Namespace) -> List[Dict[str, str]]:
    import pandas as pd

    financial_path = Path("Review revision/all_industry_expansion_v1/output/financial_rebuild/revised_analysis_dataset.csv")
    financial = pd.read_csv(financial_path, dtype={"cik": str}, low_memory=False)
    financial = financial[financial["primary_1y_text_sample"].eq(1)][
        ["cik", "fiscal_year", "split", "target_default_next_1y"]
    ]
    manifest = pd.read_csv(args.manifest_csv, dtype={"cik": str}, low_memory=False)
    eligible = financial.merge(manifest, on=["cik", "fiscal_year"], how="left", validate="one_to_one")
    eligible = eligible.sort_values(["cik", "fiscal_year"])
    return [
        {
            "cik": str(row.cik).zfill(10),
            "fiscal_year": str(int(row.fiscal_year)),
            "split": str(row.split_x),
            "target_default_next_1y": str(int(row.target_default_next_1y)),
            "accession_number": str(row.accession_number),
            "edgar_url": str(row.edgar_url),
            "filing_html_relpath": str(row.filing_html_relpath),
            "item_1_relpath": str(row.item_1_relpath),
            "item_1a_relpath": str(row.item_1a_relpath),
            "item_7_relpath": str(row.item_7_relpath),
        }
        for row in eligible.itertuples(index=False)
    ]


def score_sections(
    sections: Dict[str, str],
    lm_module: object,
    negative_words: set[str],
    uncertainty_words: set[str],
    pb_module: object,
    pb_config: Dict[str, object],
) -> Dict[str, str]:
    combined = (
        "[ITEM 1]\n" + sections.get("item_1", "").strip() + "\n\n"
        "[ITEM 1A]\n" + sections.get("item_1a", "").strip() + "\n\n"
        "[ITEM 7]\n" + sections.get("item_7", "").strip()
    ).strip()
    total_chars = sum(len(sections.get(key, "").strip()) for key in ["item_1", "item_1a", "item_7"])
    usable = total_chars >= 100 and bool(combined)
    output = {
        "usable_text_flag": "1" if usable else "0",
        "usable_text_total_chars": str(total_chars),
        "combined_text_sha256": hashlib.sha256(combined.encode("utf-8")).hexdigest() if combined else "",
        "lm_negative_count": "",
        "lm_uncertainty_count": "",
        "lm_token_count": "",
        "lm_negative_per_1000": "",
        "lm_uncertainty_per_1000": "",
        "lm_raw_signal": "",
        "lm_section_label_tokens_included": "1",
        "pb_raw_5pillar_equal_no_trigger": "",
    }
    if not usable:
        return output
    tokens = lm_module.TOKEN_RE.findall(combined.lower())
    token_count = len(tokens)
    negative_count = sum(1 for token in tokens if token in negative_words)
    uncertainty_count = sum(1 for token in tokens if token in uncertainty_words)
    negative_rate = 1000.0 * negative_count / token_count if token_count else 0.0
    uncertainty_rate = 1000.0 * uncertainty_count / token_count if token_count else 0.0
    lm_raw = 0.7 * negative_rate + 0.3 * uncertainty_rate
    paragraphs = pb_module.split_paragraphs_fast(combined)
    pb_raw = pb_module.compute_equal_no_trigger_raw(paragraphs, pb_config)
    output.update({
        "lm_negative_count": str(negative_count),
        "lm_uncertainty_count": str(uncertainty_count),
        "lm_token_count": str(token_count),
        "lm_negative_per_1000": format_number(negative_rate),
        "lm_uncertainty_per_1000": format_number(uncertainty_rate),
        "lm_raw_signal": format_number(lm_raw),
        "pb_raw_5pillar_equal_no_trigger": format_number(pb_raw),
    })
    return output


def read_cached_sections(row: Dict[str, str], cache_root: Path) -> Dict[str, str] | None:
    for root in [cache_root, _TEXT_WORKER_SECONDARY_ROOT]:
        paths = {
            "item_1": root / row["item_1_relpath"],
            "item_1a": root / row["item_1a_relpath"],
            "item_7": root / row["item_7_relpath"],
        }
        if all(path.exists() for path in paths.values()):
            return {key: path.read_text(encoding="utf-8", errors="ignore") for key, path in paths.items()}
    return None


def init_text_worker() -> None:
    global _TEXT_WORKER_LM, _TEXT_WORKER_PB, _TEXT_WORKER_NEGATIVE, _TEXT_WORKER_UNCERTAINTY, _TEXT_WORKER_PB_CONFIG
    _TEXT_WORKER_LM = load_module(Path("score_10k_distress_lm_full.py"), f"lm_worker_{mp.current_process().pid}")
    _TEXT_WORKER_PB = load_module(Path("April 4th/add_stress_event_equal_no_trigger_column.py"), f"pb_worker_{mp.current_process().pid}")
    _TEXT_WORKER_NEGATIVE, _TEXT_WORKER_UNCERTAINTY, _ = _TEXT_WORKER_LM.load_lm_master_dictionary(
        Path("April 4th/lm_master_dictionary.csv")
    )
    _TEXT_WORKER_PB_CONFIG = _TEXT_WORKER_PB.load_event_cfg(Path("April 4th/stress_event_updated_1_method_spec.json"))


def score_cached_worker(row: Dict[str, str]) -> Dict[str, str]:
    sections = read_cached_sections(row, _TEXT_WORKER_CACHE_ROOT)
    result = {
        "cik": row["cik"],
        "fiscal_year": row["fiscal_year"],
        "split": row["split"],
        "target_default_next_1y": row["target_default_next_1y"],
        "accession_number": row["accession_number"],
        "source": "cached_items",
        "status": "ok" if sections is not None else "error",
        "error": "" if sections is not None else "cached_items_missing_during_worker_read",
    }
    if sections is not None:
        result.update(score_sections(
            sections,
            _TEXT_WORKER_LM,
            _TEXT_WORKER_NEGATIVE,
            _TEXT_WORKER_UNCERTAINTY,
            _TEXT_WORKER_PB,
            _TEXT_WORKER_PB_CONFIG,
        ))
    return result


def recalibrate_lm_label_tokens() -> Dict[str, int]:
    path = Path("Review revision/all_industry_expansion_v1/output/all_industry_text_scores_raw_checkpoint.csv")
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    updated = 0
    for row in rows:
        if not row.get("lm_token_count") or row.get("lm_section_label_tokens_included") == "1":
            continue
        token_count = int(float(row["lm_token_count"]))
        negative_count = int(float(row["lm_negative_count"]))
        uncertainty_count = int(float(row["lm_uncertainty_count"]))
        corrected_tokens = token_count + 4
        negative_rate = 1000.0 * negative_count / corrected_tokens
        uncertainty_rate = 1000.0 * uncertainty_count / corrected_tokens
        row["lm_token_count"] = str(corrected_tokens)
        row["lm_negative_per_1000"] = format_number(negative_rate)
        row["lm_uncertainty_per_1000"] = format_number(uncertainty_rate)
        row["lm_raw_signal"] = format_number(0.7 * negative_rate + 0.3 * uncertainty_rate)
        row["lm_section_label_tokens_included"] = "1"
        updated += 1
    if "lm_section_label_tokens_included" not in fieldnames:
        fieldnames.append("lm_section_label_tokens_included")
    temp_path = path.with_suffix(".csv.tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)
    return {"lm_checkpoint_rows_recalibrated": updated}


def score_all_text(args: argparse.Namespace) -> Dict[str, object]:
    output_dir = Path("Review revision/all_industry_expansion_v1/output")
    checkpoint_path = output_dir / "all_industry_text_scores_raw_checkpoint.csv"
    final_path = output_dir / "all_industry_text_scores.csv"
    summary_path = output_dir / "all_industry_text_scores_summary.json"
    fieldnames = [
        "cik", "fiscal_year", "split", "target_default_next_1y", "accession_number", "source",
        "status", "error", "usable_text_flag", "usable_text_total_chars", "combined_text_sha256",
        "lm_negative_count", "lm_uncertainty_count", "lm_token_count", "lm_negative_per_1000",
        "lm_uncertainty_per_1000", "lm_raw_signal", "pb_raw_5pillar_equal_no_trigger",
        "lm_section_label_tokens_included",
    ]
    lm_module = load_module(Path("score_10k_distress_lm_full.py"), "lm_scorer")
    pb_module = load_module(Path("April 4th/add_stress_event_equal_no_trigger_column.py"), "pb_scorer")
    extractor = load_module(Path("extract_10k_items_from_manifest.py"), "item_extractor")
    downloader = load_module(Path("download_10k_filings_from_manifest.py"), "filing_downloader")
    negative_words, uncertainty_words, _ = lm_module.load_lm_master_dictionary(Path("April 4th/lm_master_dictionary.csv"))
    pb_config = pb_module.load_event_cfg(Path("April 4th/stress_event_updated_1_method_spec.json"))
    eligible_rows = build_eligible_rows(args)
    completed = set()
    if checkpoint_path.exists():
        with checkpoint_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("status") == "ok":
                    completed.add((row.get("cik", ""), row.get("fiscal_year", "")))
    write_header = not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0
    cache_root = Path("April 4th")
    cached_tasks = []
    for row in eligible_rows:
        key = (row["cik"], row["fiscal_year"])
        if key in completed:
            continue
        item_columns = ["item_1_relpath", "item_1a_relpath", "item_7_relpath"]
        cached_primary = all((cache_root / row[column]).exists() for column in item_columns)
        cached_secondary = all((_TEXT_WORKER_SECONDARY_ROOT / row[column]).exists() for column in item_columns)
        if cached_primary or cached_secondary:
            cached_tasks.append(row)
    cached_count = 0
    if cached_tasks:
        with checkpoint_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
                write_header = False
            with mp.Pool(processes=min(10, mp.cpu_count() or 1), initializer=init_text_worker) as pool:
                for result in pool.imap_unordered(score_cached_worker, cached_tasks, chunksize=20):
                    writer.writerow(result)
                    completed.add((result["cik"], result["fiscal_year"]))
                    cached_count += 1
                    if cached_count % 250 == 0:
                        handle.flush()
                        print(f"Cached text progress {cached_count:,}/{len(cached_tasks):,}", flush=True)
    network_candidates = []
    if args.max_network_filings != 0:
        for row in eligible_rows:
            key = (row["cik"], row["fiscal_year"])
            if key in completed:
                continue
            secondary_items_exist = all(
                (_TEXT_WORKER_SECONDARY_ROOT / row[column]).exists()
                for column in ["item_1_relpath", "item_1a_relpath", "item_7_relpath"]
            )
            if not (cache_root / row["filing_html_relpath"]).exists() and not secondary_items_exist:
                network_candidates.append(row)
        if args.max_network_filings > 0:
            network_candidates = network_candidates[:args.max_network_filings]
    request_lock = threading.Lock()
    last_request_start = [0.0]

    def network_worker(row: Dict[str, str]) -> Dict[str, str]:
        interval = 1.0 / max(args.requests_per_second, 0.1)
        with request_lock:
            elapsed = time.monotonic() - last_request_start[0]
            if elapsed < interval:
                time.sleep((interval - elapsed) + random.uniform(0.0, 0.03))
            last_request_start[0] = time.monotonic()
        ok, _, body, error = downloader.download_with_retries(row["edgar_url"], args.user_agent, 45.0, 4)
        result = {column: "" for column in fieldnames}
        result.update({column: row.get(column, "") for column in ["cik", "fiscal_year", "split", "target_default_next_1y", "accession_number"]})
        result["source"] = "network_stream"
        result["error"] = error
        if not ok:
            result["status"] = "error"
            return result
        raw_html = body.decode("utf-8", errors="ignore")
        normalized = extractor.normalize_text_from_html(raw_html)
        sections, _ = extractor.extract_items(normalized)
        for section_key, relpath_column in [("item_1", "item_1_relpath"), ("item_1a", "item_1a_relpath"), ("item_7", "item_7_relpath")]:
            path = _TEXT_WORKER_SECONDARY_ROOT / row[relpath_column]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(sections.get(section_key, ""), encoding="utf-8")
        combined = "\n\n".join(sections.get(key, "").strip() for key in ["item_1", "item_1a", "item_7"] if sections.get(key, "").strip())
        total_chars = sum(len(sections.get(key, "").strip()) for key in ["item_1", "item_1a", "item_7"])
        result["usable_text_flag"] = "1" if total_chars >= 100 and combined else "0"
        result["usable_text_total_chars"] = str(total_chars)
        result["combined_text_sha256"] = hashlib.sha256(combined.encode("utf-8")).hexdigest() if combined else ""
        result["status"] = "extracted"
        return result

    network_count = 0
    threaded_failures = 0
    if network_candidates:
        with checkpoint_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
                write_header = False
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(network_worker, row) for row in network_candidates]
                for future in as_completed(futures):
                    result = future.result()
                    writer.writerow(result)
                    completed.add((result["cik"], result["fiscal_year"]))
                    network_count += 1
                    threaded_failures += int(result["status"] == "error")
                    if network_count % 100 == 0:
                        handle.flush()
                        print(f"Network text progress {network_count:,}/{len(network_candidates):,}; failures={threaded_failures:,}", flush=True)
    cached_filing_count = 0
    failed_count = threaded_failures
    last_request_at = 0.0
    with checkpoint_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for index, row in enumerate(eligible_rows, start=1):
            key = (row["cik"], row["fiscal_year"])
            if key in completed:
                continue
            sections = read_cached_sections(row, cache_root)
            source = "cached_items"
            error = ""
            if sections is not None:
                cached_count += 1
            else:
                filing_path = cache_root / row["filing_html_relpath"]
                raw_html = ""
                if filing_path.exists():
                    raw_html = filing_path.read_text(encoding="utf-8", errors="ignore")
                    source = "cached_filing"
                    cached_filing_count += 1
                else:
                    if args.max_network_filings == 0:
                        continue
                    if args.max_network_filings > 0 and network_count >= args.max_network_filings:
                        continue
                    interval = 1.0 / max(args.requests_per_second, 0.1)
                    elapsed = time.monotonic() - last_request_at
                    if elapsed < interval:
                        time.sleep((interval - elapsed) + random.uniform(0.0, 0.1))
                    ok, _, body, error = downloader.download_with_retries(
                        row["edgar_url"], args.user_agent, 45.0, 4
                    )
                    last_request_at = time.monotonic()
                    network_count += 1
                    source = "network_stream"
                    raw_html = body.decode("utf-8", errors="ignore") if ok else ""
                if raw_html:
                    normalized = extractor.normalize_text_from_html(raw_html)
                    sections, _ = extractor.extract_items(normalized)
                else:
                    sections = None
            result = {column: "" for column in fieldnames}
            result.update({column: row.get(column, "") for column in ["cik", "fiscal_year", "split", "target_default_next_1y", "accession_number"]})
            result["source"] = source
            result["error"] = error
            if sections is None:
                result["status"] = "error"
                failed_count += 1
            else:
                result.update(score_sections(sections, lm_module, negative_words, uncertainty_words, pb_module, pb_config))
                result["status"] = "ok"
            writer.writerow(result)
            if index % 250 == 0:
                handle.flush()
                print(f"Text progress {index:,}/{len(eligible_rows):,}; cached={cached_count:,}; network={network_count:,}; failed={failed_count:,}", flush=True)
    latest_by_key: Dict[tuple[str, str], Dict[str, str]] = {}
    with checkpoint_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row.get("cik", ""), row.get("fiscal_year", ""))
            if row.get("status") in {"ok", "extracted"} or key not in latest_by_key:
                latest_by_key[key] = row
    final_rows = [latest_by_key[key] for key in sorted(latest_by_key)]
    lm_low, lm_high = fit_bounds(final_rows, "lm_raw_signal")
    pb_low, pb_high = fit_bounds(final_rows, "pb_raw_5pillar_equal_no_trigger")
    for row in final_rows:
        lm_raw = float(row["lm_raw_signal"]) if row.get("lm_raw_signal") else None
        pb_raw = float(row["pb_raw_5pillar_equal_no_trigger"]) if row.get("pb_raw_5pillar_equal_no_trigger") else None
        row["distress_score_0_100_LM"] = format_number(
            100.0 * (min(max(lm_raw, lm_low), lm_high) - lm_low) / (lm_high - lm_low)
        ) if lm_raw is not None else ""
        row["stress_event_5pillar_equal_no_trigger"] = format_number(
            100.0 * (min(max(pb_raw, pb_low), pb_high) - pb_low) / (pb_high - pb_low)
        ) if pb_raw is not None else ""
    final_fieldnames = fieldnames + ["distress_score_0_100_LM", "stress_event_5pillar_equal_no_trigger"]
    with final_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=final_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final_rows)
    eligible_keys = {(row["cik"], row["fiscal_year"]) for row in eligible_rows}
    scored_keys = {key for key, row in latest_by_key.items() if row.get("status") == "ok"}
    summary = {
        "eligible_rows": len(eligible_rows),
        "checkpoint_rows": len(final_rows),
        "successfully_scored_rows": len(scored_keys),
        "remaining_rows": len(eligible_keys - scored_keys),
        "cached_items_scored_this_run": cached_count,
        "cached_filings_scored_this_run": cached_filing_count,
        "network_filings_attempted_this_run": network_count,
        "failures_this_run": failed_count,
        "lm_train_scaler_p01": lm_low,
        "lm_train_scaler_p99": lm_high,
        "pb_train_scaler_p01": pb_low,
        "pb_train_scaler_p99": pb_high,
        "checkpoint_csv": str(checkpoint_path),
        "final_csv": str(final_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main() -> int:
    args = parse_args()
    for path in [args.defaults_csv, args.submissions_dir, args.companyfacts_dir]:
        if not path.exists():
            raise FileNotFoundError(path)
    source = load_source_module()
    case_count_by_group, default_ciks_by_group, all_default_ciks = source.build_ucla_industry_maps(args.defaults_csv)
    rows: List[Dict[str, str]] = []
    parse_failures = 0
    scanned = 0
    seen_ciks = set()
    for path in source.iter_submission_files(args.submissions_dir, args.companyfacts_dir):
        scanned += 1
        parsed = parse_company_allow_missing_sic(source, path)
        if parsed is None:
            parse_failures += 1
            continue
        cik = parsed["cik"]
        if cik in seen_ciks:
            continue
        seen_ciks.add(cik)
        parsed["ucla_default_cik_flag"] = "1" if cik in all_default_ciks else "0"
        parsed["sampling_group"] = "all_companyfacts_covered_sec_filers"
        rows.append(parsed)
    rows.sort(key=lambda row: (int(row["sic_major_group"] or 999), int(row["sic"] or 9999), row["cik"]))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    company_counts = Counter(row["sic_major_group"] or "missing" for row in rows)
    default_counts = Counter(row["sic_major_group"] or "missing" for row in rows if row["ucla_default_cik_flag"] == "1")
    audit_rows = []
    all_groups = sorted(set(company_counts) | {str(group) for group in case_count_by_group}, key=lambda value: (value == "missing", int(value) if value != "missing" else 999))
    for major_group in all_groups:
        audit_rows.append({
            "sic_major_group": major_group,
            "submission_company_count": company_counts.get(major_group, 0),
            "submission_default_cik_count": default_counts.get(major_group, 0),
            "brd_case_count": case_count_by_group.get(int(major_group), 0) if major_group != "missing" else 0,
            "brd_unique_default_ciks": len(default_ciks_by_group.get(int(major_group), set())) if major_group != "missing" else 0,
            "included_without_sampling": 1,
        })
    with args.industry_audit_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audit_rows[0]))
        writer.writeheader()
        writer.writerows(audit_rows)
    summary = {
        "design": "all companyfacts-covered SEC filers with matching main submissions file; no SIC or outcome-based sampling",
        "scanned_submission_files": scanned,
        "parse_failures": parse_failures,
        "company_rows": len(rows),
        "sic_major_groups_with_codes": len([group for group in company_counts if group != "missing"]),
        "companies_missing_sic": company_counts.get("missing", 0),
        "roster_default_ciks": sum(default_counts.values()),
        "brd_default_ciks_total": len(all_default_ciks),
        "output_csv": str(args.output_csv),
    }
    if args.prepare_targets:
        summary.update(attach_defaults_and_build_targets(args))
        summary["financials_with_defaults_csv"] = str(args.financials_with_defaults_csv)
        summary["target_years_csv"] = str(args.target_years_csv)
    if args.build_manifest:
        summary.update(build_manifest(args))
        summary["manifest_csv"] = str(args.manifest_csv)
    if args.build_financial_eligibility:
        summary.update(build_financial_eligibility(args))
    if args.recalibrate_lm_label_tokens:
        summary.update(recalibrate_lm_label_tokens())
    if args.score_all_text:
        summary["text_scoring"] = score_all_text(args)
    if args.evaluate_text:
        summary.update(evaluate_text_models())
    args.metadata_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
