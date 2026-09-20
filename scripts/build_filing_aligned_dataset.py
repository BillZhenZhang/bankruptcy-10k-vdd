#!/usr/bin/env python3
"""Build a filing-aligned, non-imputed bankruptcy analysis dataset."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


METRIC_SPECS: Dict[str, Dict[str, object]] = {
    "assets": {"kind": "instant", "tags": ["Assets"]},
    "liabilities": {"kind": "instant", "tags": ["Liabilities"]},
    "stockholders_equity": {
        "kind": "instant",
        "tags": [
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
            "StockholdersEquity",
        ],
    },
    "current_assets": {"kind": "instant", "tags": ["AssetsCurrent"]},
    "current_liabilities": {"kind": "instant", "tags": ["LiabilitiesCurrent"]},
    "net_income": {"kind": "duration", "tags": ["NetIncomeLoss", "ProfitLoss"]},
    "operating_cash_flow": {
        "kind": "duration",
        "tags": [
            "NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ],
    },
    "debt_current": {
        "kind": "instant",
        "tags": [
            "LongTermDebtAndFinanceLeaseObligationsCurrent",
            "LongTermDebtCurrent",
            "ShortTermDebtCurrent",
            "DebtCurrent",
            "ShortTermBorrowings",
        ],
    },
    "debt_noncurrent": {
        "kind": "instant",
        "tags": [
            "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
            "LongTermDebtNoncurrent",
        ],
    },
    "debt_total_reported": {
        "kind": "instant",
        "tags": ["LongTermDebtAndFinanceLeaseObligations", "LongTermDebt"],
    },
}

TEXT_COLUMNS = [
    "usable_text_flag",
    "usable_text_total_chars",
    "distress_score_0_100_LM",
    "lm_negative_count",
    "lm_uncertainty_count",
    "lm_token_count",
    "lm_negative_per_1000",
    "lm_uncertainty_per_1000",
    "stress_event_5pillar_equal_no_trigger",
]

PRIMARY_RAW_FACTORS = [
    "size_log_assets",
    "ohlson_tlta",
    "zmijewski_cacl",
    "ohlson_nita",
    "cashflow_ocfta",
]

OHLSON_FACTORS = [
    "ohlson_size_cpi_1980",
    "ohlson_tlta",
    "ohlson_wcta",
    "ohlson_clca",
    "ohlson_nita",
    "ohlson_futl_ocf_proxy",
    "ohlson_oeneg",
    "ohlson_intwo",
    "ohlson_chin",
]

ZMIJEWSKI_FACTORS = ["ohlson_nita", "ohlson_tlta", "zmijewski_cacl"]

TRANSFORM_MAP = {
    "size_log_assets": "identity",
    "ohlson_size_cpi_1980": "identity",
    "ohlson_tlta": "asinh",
    "ohlson_wcta": "asinh",
    "ohlson_clca": "log1p",
    "ohlson_nita": "asinh",
    "ohlson_futl_ocf_proxy": "asinh",
    "ohlson_oeneg": "identity",
    "ohlson_intwo": "identity",
    "ohlson_chin": "identity",
    "zmijewski_cacl": "log1p",
    "cashflow_ocfta": "asinh",
    "debt_to_assets_strict": "asinh",
}

CPI_ANNUAL_AVERAGE = {
    1980: 82.383333,
    2010: 218.076167,
    2011: 224.923000,
    2012: 229.586083,
    2013: 232.951750,
    2014: 236.715000,
    2015: 237.001750,
    2016: 240.005417,
    2017: 245.121000,
    2018: 251.099500,
    2019: 255.652583,
    2020: 258.855750,
    2021: 270.973417,
}

BRD_CPI_SOURCE_URL = "https://fred.stlouisfed.org/series/CPIAUCSL"
BRD_DEFINITION_URL = "https://lopucki.law.ufl.edu/contents_of_the_webbrd.php"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--financial-features", type=Path, default=Path("April 4th/financial_features.csv"))
    parser.add_argument("--manifest", type=Path, default=Path("April 4th/tenk_labeled_window_manifest.csv"))
    parser.add_argument("--text-scores", type=Path, default=Path("April 4th/tenk_text_scores_with_pb_v3_candidates.csv"))
    parser.add_argument("--companyfacts-dir", type=Path, default=Path("companyfacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("Review revision/data_rebuild_v1/output"))
    parser.add_argument("--censor-end-date", default="2023-01-12")
    parser.add_argument("--sample-start-year", type=int, default=2010)
    parser.add_argument("--sample-end-year", type=int, default=2021)
    parser.add_argument("--test-start-year", type=int, default=2019)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--skip-text-scores",
        action="store_true",
        help="Build the exact-accession financial eligibility sample before text collection.",
    )
    parser.add_argument(
        "--include-fy-plus1-q1",
        action="store_true",
        help="Include non-amended Q1 report-date matches assigned to the preceding fiscal year.",
    )
    return parser.parse_args()


def normalize_cik(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(10)


def normalize_accession(value: object) -> str:
    return str(value or "").replace("-", "").strip()


def finite_float(value: object) -> Optional[float]:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def parse_iso_date(value: object) -> Optional[date]:
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def candidate_rank(entry: Dict[str, object], report_date: str, kind: str) -> Tuple[float, ...]:
    end = str(entry.get("end") or "")
    exact_end = 0.0 if report_date and end == report_date else 1.0
    fp_rank = 0.0 if entry.get("fp") == "FY" else 1.0
    form_rank = 0.0 if entry.get("form") == "10-K" else 1.0
    if kind == "duration":
        start = parse_iso_date(entry.get("start"))
        end_date = parse_iso_date(entry.get("end"))
        duration = (end_date - start).days if start and end_date else -1
        annual_rank = 0.0 if 250 <= duration <= 450 else 1.0
        duration_distance = abs(duration - 365) if duration >= 0 else 9999.0
        return exact_end, fp_rank, form_rank, annual_rank, duration_distance
    return exact_end, fp_rank, form_rank


def extract_metric(
    facts: Dict[str, object],
    tags: Sequence[str],
    accession: str,
    report_date: str,
    kind: str,
) -> Tuple[float, str, str, str]:
    us_gaap = facts.get("us-gaap")
    if not isinstance(us_gaap, dict):
        return float("nan"), "", "", ""
    for tag in tags:
        tag_object = us_gaap.get(tag)
        if not isinstance(tag_object, dict):
            continue
        units = tag_object.get("units")
        if not isinstance(units, dict):
            continue
        entries = units.get("USD")
        if not isinstance(entries, list):
            continue
        matches: List[Dict[str, object]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if normalize_accession(entry.get("accn")) != accession:
                continue
            if entry.get("form") not in {"10-K", "10-K/A"}:
                continue
            if finite_float(entry.get("val")) is None:
                continue
            matches.append(entry)
        if not matches:
            continue
        if report_date:
            matches = [entry for entry in matches if str(entry.get("end") or "") == report_date]
            if not matches:
                continue
        if kind == "duration":
            annual_matches = []
            for entry in matches:
                start = parse_iso_date(entry.get("start"))
                end_date = parse_iso_date(entry.get("end"))
                duration = (end_date - start).days if start and end_date else -1
                if 250 <= duration <= 450:
                    annual_matches.append(entry)
            if not annual_matches:
                continue
            matches = annual_matches
        matches.sort(key=lambda entry: candidate_rank(entry, report_date, kind))
        selected = matches[0]
        selected_value = finite_float(selected.get("val"))
        if selected_value is None:
            continue
        return selected_value, tag, str(selected.get("start") or ""), str(selected.get("end") or "")
    return float("nan"), "", "", ""


def extract_company_rows(payload: Tuple[str, List[Dict[str, object]], Path]) -> List[Dict[str, object]]:
    cik, rows, companyfacts_dir = payload
    path = companyfacts_dir / f"CIK{cik}.json"
    if not path.exists():
        return [{"cik": cik, "fiscal_year": row["fiscal_year"], "companyfacts_file_missing": 1} for row in rows]
    try:
        payload_json = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [{"cik": cik, "fiscal_year": row["fiscal_year"], "companyfacts_file_missing": 1} for row in rows]
    facts = payload_json.get("facts")
    if not isinstance(facts, dict):
        facts = {}
    output: List[Dict[str, object]] = []
    for row in rows:
        out: Dict[str, object] = {"cik": cik, "fiscal_year": row["fiscal_year"], "companyfacts_file_missing": 0}
        accession = str(row.get("accession_number_normalized") or "")
        report_date = str(row.get("report_date") or "")
        for metric, spec in METRIC_SPECS.items():
            value, tag, start, end = extract_metric(
                facts=facts,
                tags=list(spec["tags"]),
                accession=accession,
                report_date=report_date,
                kind=str(spec["kind"]),
            )
            out[metric] = value
            out[f"source_tag__{metric}"] = tag
            out[f"source_start__{metric}"] = start
            out[f"source_end__{metric}"] = end
        output.append(out)
    return output


def extract_filing_aligned_facts(manifest: pd.DataFrame, companyfacts_dir: Path, workers: int) -> pd.DataFrame:
    grouped = [(str(cik), group.to_dict("records"), companyfacts_dir) for cik, group in manifest.groupby("cik", sort=False)]
    extracted: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for rows in pool.map(extract_company_rows, grouped):
            extracted.extend(rows)
    return pd.DataFrame(extracted)


def ensure_unique(table: pd.DataFrame, keys: Sequence[str]) -> Tuple[pd.DataFrame, int]:
    duplicate_count = int(table.duplicated(list(keys), keep=False).sum())
    if duplicate_count:
        table = table.sort_values(list(keys)).drop_duplicates(list(keys), keep="first")
    return table, duplicate_count


def derive_debt(row: pd.Series) -> Tuple[float, str]:
    total = finite_float(row.get("debt_total_reported"))
    if total is not None and total >= 0:
        return total, "reported_total"
    current = finite_float(row.get("debt_current"))
    noncurrent = finite_float(row.get("debt_noncurrent"))
    if current is not None and noncurrent is not None and current >= 0 and noncurrent >= 0:
        return current + noncurrent, "current_plus_noncurrent"
    return float("nan"), "missing_strict_components"


def add_labels(frame: pd.DataFrame, censor_end: date) -> pd.DataFrame:
    filing_dates = pd.to_datetime(frame["filing_date"], errors="coerce")
    default_dates = pd.to_datetime(frame["default_date"], errors="coerce")
    days = (default_dates - filing_dates).dt.days
    frame["days_from_filing_to_default"] = days
    frame["post_default_filing_flag"] = ((days < 0) & days.notna()).astype(int)
    for horizon, days_limit in [("1y", 365), ("2y", 730)]:
        target = days.between(0, days_limit, inclusive="both")
        enough_followup = filing_dates + pd.to_timedelta(days_limit, unit="D") <= pd.Timestamp(censor_end)
        observed = frame["post_default_filing_flag"].eq(0) & (target | enough_followup)
        frame[f"target_default_next_{horizon}"] = target.astype(int)
        frame[f"label_observed_next_{horizon}"] = observed.astype(int)
    for cutoff in [30, 60, 90, 180]:
        frame[f"default_within_{cutoff}d_flag"] = days.between(0, cutoff, inclusive="both").astype(int)
        frame[f"target_default_1y_excl_{cutoff}d"] = days.between(cutoff + 1, 365, inclusive="both").astype(int)
        enough_followup = filing_dates + pd.to_timedelta(365, unit="D") <= pd.Timestamp(censor_end)
        frame[f"label_observed_1y_excl_{cutoff}d"] = (
            frame["post_default_filing_flag"].eq(0)
            & frame[f"default_within_{cutoff}d_flag"].eq(0)
            & (frame[f"target_default_1y_excl_{cutoff}d"].eq(1) | enough_followup)
        ).astype(int)
    return frame


def add_factors(frame: pd.DataFrame) -> pd.DataFrame:
    for column in METRIC_SPECS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    debt_values = frame.apply(derive_debt, axis=1, result_type="expand")
    frame["total_debt_strict"] = pd.to_numeric(debt_values[0], errors="coerce")
    frame["total_debt_strict_method"] = debt_values[1]
    reported_assets = frame["assets"].where(frame["assets"] > 0)
    reported_liabilities = frame["liabilities"].where(frame["liabilities"] >= 0)
    reported_equity = frame["stockholders_equity"]
    identity_assets = (reported_liabilities + reported_equity).where((reported_liabilities + reported_equity) > 0)
    identity_liabilities = (reported_assets - reported_equity).where((reported_assets - reported_equity) >= 0)
    identity_equity = reported_assets - reported_liabilities
    frame["analysis_assets"] = reported_assets.fillna(identity_assets)
    frame["analysis_liabilities"] = reported_liabilities.fillna(identity_liabilities)
    frame["analysis_equity"] = reported_equity.fillna(identity_equity)
    frame["analysis_assets_method"] = np.where(
        reported_assets.notna(), "reported_assets", np.where(identity_assets.notna(), "liabilities_plus_equity", "missing")
    )
    frame["analysis_liabilities_method"] = np.where(
        reported_liabilities.notna(), "reported_liabilities", np.where(identity_liabilities.notna(), "assets_minus_equity", "missing")
    )
    frame["analysis_equity_method"] = np.where(
        reported_equity.notna(), "reported_equity", np.where(identity_equity.notna(), "assets_minus_liabilities", "missing")
    )
    valid_assets = frame["analysis_assets"].where(frame["analysis_assets"] > 0)
    nonnegative_liabilities = frame["analysis_liabilities"].where(frame["analysis_liabilities"] >= 0)
    valid_liabilities = nonnegative_liabilities.where(nonnegative_liabilities > 0)
    nonnegative_current_assets = frame["current_assets"].where(frame["current_assets"] >= 0)
    nonnegative_current_liabilities = frame["current_liabilities"].where(frame["current_liabilities"] >= 0)
    valid_current_assets = nonnegative_current_assets.where(nonnegative_current_assets > 0)
    valid_current_liabilities = nonnegative_current_liabilities.where(nonnegative_current_liabilities > 0)
    frame["size_log_assets"] = np.log(valid_assets)
    cpi_scale = frame["fiscal_year"].map(
        lambda year: CPI_ANNUAL_AVERAGE.get(int(year), float("nan")) / CPI_ANNUAL_AVERAGE[1980]
        if pd.notna(year) else float("nan")
    )
    frame["ohlson_size_cpi_1980"] = np.log(valid_assets / cpi_scale)
    frame["ohlson_tlta"] = nonnegative_liabilities / valid_assets
    frame["ohlson_wcta"] = (nonnegative_current_assets - nonnegative_current_liabilities) / valid_assets
    frame["ohlson_clca"] = nonnegative_current_liabilities / valid_current_assets
    frame["ohlson_nita"] = frame["net_income"] / valid_assets
    frame["ohlson_futl_ocf_proxy"] = frame["operating_cash_flow"] / valid_liabilities
    frame["zmijewski_cacl"] = nonnegative_current_assets / valid_current_liabilities
    frame["cashflow_ocfta"] = frame["operating_cash_flow"] / valid_assets
    frame["debt_to_assets_strict"] = frame["total_debt_strict"] / valid_assets
    equity_negative = frame["analysis_equity"].lt(0) & frame["analysis_equity"].notna()
    inferred_negative = nonnegative_liabilities.gt(valid_assets) & pd.concat([nonnegative_liabilities, valid_assets], axis=1).notna().all(axis=1)
    frame["ohlson_oeneg"] = np.where(
        frame["analysis_equity"].notna(),
        equity_negative.astype(float),
        np.where(pd.concat([nonnegative_liabilities, valid_assets], axis=1).notna().all(axis=1), inferred_negative.astype(float), np.nan),
    )
    frame = frame.sort_values(["cik", "fiscal_year"]).copy()
    prior_year = frame.groupby("cik")["fiscal_year"].shift(1)
    prior_income = frame.groupby("cik")["net_income"].shift(1)
    consecutive = prior_year.eq(frame["fiscal_year"] - 1)
    frame["ohlson_intwo"] = np.where(
        consecutive & frame["net_income"].notna() & prior_income.notna(),
        ((frame["net_income"] < 0) & (prior_income < 0)).astype(float),
        np.nan,
    )
    denominator = frame["net_income"].abs() + prior_income.abs()
    frame["ohlson_chin"] = np.where(
        consecutive & frame["net_income"].notna() & prior_income.notna() & denominator.gt(0),
        (frame["net_income"] - prior_income) / denominator,
        np.nan,
    )
    return frame


def transform_series(series: pd.Series, method: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if method == "asinh":
        return np.arcsinh(numeric)
    if method == "log1p":
        return np.log1p(numeric.where(numeric >= 0))
    return numeric


def add_treatments(frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    parameters: List[Dict[str, object]] = []
    train_mask = frame["split"].eq("train") & frame["primary_1y_text_sample"].eq(1)
    for raw_column, method in TRANSFORM_MAP.items():
        transformed = transform_series(frame[raw_column], method)
        frame[f"t__{raw_column}"] = transformed
        train_values = transformed[train_mask & transformed.notna()]
        lower = float(train_values.quantile(0.01)) if not train_values.empty else float("nan")
        upper = float(train_values.quantile(0.99)) if not train_values.empty else float("nan")
        frame[f"w__{raw_column}"] = transformed.clip(lower=lower, upper=upper)
        frame[f"mi__{raw_column}"] = frame[raw_column].isna().astype(int)
        parameters.append(
            {
                "raw_variable": raw_column,
                "transformation": method,
                "primary_treatment": "monotonic_transform_without_clipping",
                "sensitivity_treatment": "training_only_1st_99th_percentile_clip_after_transform",
                "training_nonmissing_n": int(train_values.shape[0]),
                "sensitivity_lower_bound": lower,
                "sensitivity_upper_bound": upper,
            }
        )
    return frame, pd.DataFrame(parameters)


def add_quality_flags(frame: pd.DataFrame) -> pd.DataFrame:
    sic = pd.to_numeric(frame["sic"], errors="coerce")
    frame["financial_firm_flag"] = sic.between(6000, 6799, inclusive="both").astype(int)
    frame["micro_assets_lt_1m_flag"] = (frame["analysis_assets"].notna() & frame["analysis_assets"].lt(1_000_000)).astype(int)
    frame["micro_assets_lt_100k_flag"] = (frame["analysis_assets"].notna() & frame["analysis_assets"].lt(100_000)).astype(int)
    frame["nonpositive_assets_flag"] = (frame["assets"].notna() & frame["assets"].le(0)).astype(int)
    frame["extreme_abs_roa_gt_10_flag"] = (frame["ohlson_nita"].abs() > 10).astype(int)
    frame["extreme_abs_ocfta_gt_10_flag"] = (frame["cashflow_ocfta"].abs() > 10).astype(int)
    frame["strict_debt_available_flag"] = frame["total_debt_strict"].notna().astype(int)
    frame["assets_identity_recovery_flag"] = frame["analysis_assets_method"].eq("liabilities_plus_equity").astype(int)
    frame["liabilities_identity_recovery_flag"] = frame["analysis_liabilities_method"].eq("assets_minus_equity").astype(int)
    frame["equity_identity_recovery_flag"] = frame["analysis_equity_method"].eq("assets_minus_liabilities").astype(int)
    frame["complete_core5_no_imputation"] = frame[[f"t__{column}" for column in PRIMARY_RAW_FACTORS]].notna().all(axis=1).astype(int)
    frame["complete_ohlson9_related_no_imputation"] = frame[[f"t__{column}" for column in OHLSON_FACTORS]].notna().all(axis=1).astype(int)
    frame["complete_zmijewski3_no_imputation"] = frame[[f"t__{column}" for column in ZMIJEWSKI_FACTORS]].notna().all(axis=1).astype(int)
    return frame


def summarize_variable(frame: pd.DataFrame, variable: str, sample_mask: pd.Series) -> Dict[str, object]:
    values = pd.to_numeric(frame.loc[sample_mask, variable], errors="coerce")
    nonmissing = values.dropna()
    result: Dict[str, object] = {
        "variable": variable,
        "sample_rows": int(sample_mask.sum()),
        "nonmissing_n": int(nonmissing.shape[0]),
        "missing_n": int(values.isna().sum()),
        "missing_rate": float(values.isna().mean()) if values.shape[0] else float("nan"),
        "zero_n": int((nonmissing == 0).sum()),
        "unique_n": int(nonmissing.nunique()),
    }
    for name, quantile in [("min", 0.0), ("p01", 0.01), ("p25", 0.25), ("median", 0.5), ("p75", 0.75), ("p99", 0.99), ("max", 1.0)]:
        result[name] = float(nonmissing.quantile(quantile)) if not nonmissing.empty else float("nan")
    result["mean"] = float(nonmissing.mean()) if not nonmissing.empty else float("nan")
    result["std"] = float(nonmissing.std()) if nonmissing.shape[0] > 1 else float("nan")
    return result


def write_outputs(frame: pd.DataFrame, treatment_parameters: pd.DataFrame, output_dir: Path, summary: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_columns = [
        "cik", "fiscal_year", "name", "sic", "sic_description", "split", "selection_status", "selected_form",
        "candidate_count_same_year", "report_date", "filing_date", "accession_number", "default_date",
        "days_from_filing_to_default", "post_default_filing_flag", "label_observed_next_1y", "target_default_next_1y",
        "label_observed_next_2y", "target_default_next_2y",
        *[f"default_within_{cutoff}d_flag" for cutoff in [30, 60, 90, 180]],
        *[f"label_observed_1y_excl_{cutoff}d" for cutoff in [30, 60, 90, 180]],
        *[f"target_default_1y_excl_{cutoff}d" for cutoff in [30, 60, 90, 180]],
        *TEXT_COLUMNS, "full_public_1y_text_sample", "brd_asset_threshold_usd", "brd_size_eligibility_basis",
        "brd_size_eligible_flag",
        "primary_1y_text_sample", *METRIC_SPECS.keys(),
        *[f"source_tag__{metric}" for metric in METRIC_SPECS],
        *[f"source_start__{metric}" for metric in METRIC_SPECS],
        *[f"source_end__{metric}" for metric in METRIC_SPECS],
        "total_debt_strict", "total_debt_strict_method",
        "analysis_assets", "analysis_liabilities", "analysis_equity", "analysis_assets_method",
        "analysis_liabilities_method", "analysis_equity_method",
        *list(dict.fromkeys(PRIMARY_RAW_FACTORS + OHLSON_FACTORS + ZMIJEWSKI_FACTORS + ["debt_to_assets_strict"])),
        *[f"t__{column}" for column in TRANSFORM_MAP], *[f"w__{column}" for column in TRANSFORM_MAP],
        *[f"mi__{column}" for column in TRANSFORM_MAP], "financial_firm_flag", "micro_assets_lt_1m_flag",
        "micro_assets_lt_100k_flag", "nonpositive_assets_flag", "extreme_abs_roa_gt_10_flag",
        "extreme_abs_ocfta_gt_10_flag", "strict_debt_available_flag", "complete_core5_no_imputation",
        "assets_identity_recovery_flag", "liabilities_identity_recovery_flag", "equity_identity_recovery_flag",
        "complete_ohlson9_related_no_imputation", "complete_zmijewski3_no_imputation",
    ]
    output_columns = [column for column in dict.fromkeys(output_columns) if column in frame.columns]
    frame[output_columns].to_csv(output_dir / "revised_analysis_dataset.csv", index=False)
    primary_mask = frame["primary_1y_text_sample"].eq(1)
    audit_variables = list(dict.fromkeys(
        list(METRIC_SPECS.keys()) + ["analysis_assets", "analysis_liabilities", "analysis_equity"]
        + PRIMARY_RAW_FACTORS + OHLSON_FACTORS + ZMIJEWSKI_FACTORS
        + ["debt_to_assets_strict"] + [f"t__{column}" for column in TRANSFORM_MAP]
    ))
    pd.DataFrame([summarize_variable(frame, variable, primary_mask) for variable in audit_variables]).to_csv(
        output_dir / "financial_variable_audit.csv", index=False
    )
    treatment_parameters.to_csv(output_dir / "data_treatment_parameters.csv", index=False)
    factor_sets = {
        "core5_ohlson_zm_related": "complete_core5_no_imputation",
        "ohlson9_related": "complete_ohlson9_related_no_imputation",
        "zmijewski3": "complete_zmijewski3_no_imputation",
    }
    coverage_rows: List[Dict[str, object]] = []
    for factor_set, indicator in factor_sets.items():
        eligible = primary_mask & frame[indicator].eq(1)
        coverage_rows.append({
            "factor_set": factor_set,
            "rows": int(eligible.sum()), "firms": int(frame.loc[eligible, "cik"].nunique()),
            "events_1y": int(frame.loc[eligible, "target_default_next_1y"].sum()),
            "train_rows": int((eligible & frame["split"].eq("train")).sum()),
            "train_events": int(frame.loc[eligible & frame["split"].eq("train"), "target_default_next_1y"].sum()),
            "test_rows": int((eligible & frame["split"].eq("test")).sum()),
            "test_events": int(frame.loc[eligible & frame["split"].eq("test"), "target_default_next_1y"].sum()),
        })
    pd.DataFrame(coverage_rows).to_csv(output_dir / "factor_set_coverage.csv", index=False)
    frame.loc[primary_mask].groupby(["fiscal_year", "split"], dropna=False).agg(
        rows=("cik", "size"), firms=("cik", "nunique"), events_1y=("target_default_next_1y", "sum"),
        financial_firms=("financial_firm_flag", "sum"), micro_assets_lt_1m=("micro_assets_lt_1m_flag", "sum"),
    ).reset_index().to_csv(output_dir / "sample_by_year.csv", index=False)
    sic_numeric = pd.to_numeric(frame.loc[primary_mask, "sic"], errors="coerce")
    sic_audit = frame.loc[primary_mask, ["cik", "sic", "sic_description", "split", "target_default_next_1y"]].copy()
    sic_audit["sic_major_group"] = (sic_numeric // 100).astype("Int64")
    sic_audit.groupby("sic_major_group", dropna=False).agg(
        rows=("cik", "size"),
        firms=("cik", "nunique"),
        events_1y=("target_default_next_1y", "sum"),
        train_rows=("split", lambda values: int(values.eq("train").sum())),
        train_events=("target_default_next_1y", lambda values: int(values[sic_audit.loc[values.index, "split"].eq("train")].sum())),
        test_rows=("split", lambda values: int(values.eq("test").sum())),
        test_events=("target_default_next_1y", lambda values: int(values[sic_audit.loc[values.index, "split"].eq("test")].sum())),
        sic_industries=("sic_description", "nunique"),
        sic_description_examples=("sic_description", lambda values: " | ".join(sorted(values.dropna().astype(str).unique())[:5])),
    ).reset_index().sort_values("sic_major_group").to_csv(
        output_dir / "sic_group_audit.csv", index=False
    )
    pd.DataFrame([
        {"stage": "Raw pre-imputation financial firm-years", "rows": summary["financial_rows_raw"]},
        {"stage": "Manifest rows", "rows": summary["manifest_rows_raw"]},
        {"stage": "Matched non-amended 10-K rows", "rows": summary["matched_10k_rows"]},
        {"stage": "Within 2010-2021 fiscal-year window", "rows": summary["within_year_window_rows"]},
        {"stage": "Valid filing date and non-post-default", "rows": summary["valid_timing_rows"]},
        {"stage": "Observable one-year outcome", "rows": summary["observed_1y_rows"]},
        {
            "stage": "Financially observable filing rows" if summary.get("text_scores_skipped") else "Usable text and both primary text scores",
            "rows": summary["full_public_1y_text_rows"],
        },
        {"stage": "BRD-consistent inflation-adjusted large-company universe", "rows": summary["primary_1y_text_rows"]},
    ]).to_csv(output_dir / "sample_flow.csv", index=False)
    quality_rows = []
    for flag in [
        "companyfacts_file_missing", "financial_firm_flag", "micro_assets_lt_1m_flag", "micro_assets_lt_100k_flag",
        "nonpositive_assets_flag", "extreme_abs_roa_gt_10_flag", "extreme_abs_ocfta_gt_10_flag",
        "strict_debt_available_flag", "assets_identity_recovery_flag", "liabilities_identity_recovery_flag",
        "equity_identity_recovery_flag",
    ]:
        quality_rows.append({
            "flag": flag,
            "primary_sample_count": int(frame.loc[primary_mask, flag].fillna(0).astype(float).sum()),
            "primary_sample_rate": float(frame.loc[primary_mask, flag].fillna(0).astype(float).mean()),
        })
    pd.DataFrame(quality_rows).to_csv(output_dir / "data_quality_flag_summary.csv", index=False)
    (output_dir / "data_rebuild_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    required_paths = [args.financial_features, args.manifest, args.companyfacts_dir]
    if not args.skip_text_scores:
        required_paths.append(args.text_scores)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    censor_end = date.fromisoformat(args.censor_end_date)
    financial = pd.read_csv(args.financial_features, dtype={"cik": str}, low_memory=False)
    manifest = pd.read_csv(args.manifest, dtype={"cik": str}, low_memory=False)
    text = (
        pd.DataFrame(columns=["cik", "fiscal_year"])
        if args.skip_text_scores
        else pd.read_csv(args.text_scores, dtype={"cik": str}, low_memory=False)
    )
    for table in [financial, manifest, text]:
        table["cik"] = normalize_cik(table["cik"])
        table["fiscal_year"] = pd.to_numeric(table["fiscal_year"], errors="coerce").astype("Int64")
    financial, financial_duplicates = ensure_unique(financial, ["cik", "fiscal_year"])
    manifest, manifest_duplicates = ensure_unique(manifest, ["cik", "fiscal_year"])
    text, text_duplicates = ensure_unique(text, ["cik", "fiscal_year"])
    manifest["accession_number_normalized"] = manifest["accession_number"].map(normalize_accession)
    matched_statuses = {"matched_10k"}
    if args.include_fy_plus1_q1:
        matched_statuses.add("matched_10k_fy_plus1_q1")
    manifest_subset = manifest[
        manifest["selection_status"].isin(matched_statuses) & manifest["selected_form"].eq("10-K")
        & manifest["accession_number_normalized"].ne("")
    ].copy()
    extracted = extract_filing_aligned_facts(manifest_subset, args.companyfacts_dir, args.workers)
    frame = manifest_subset.merge(
        financial[["cik", "fiscal_year", "name", "sic", "sic_description", "default_date"]],
        on=["cik", "fiscal_year"], how="left",
    )
    frame = frame.merge(extracted, on=["cik", "fiscal_year"], how="left", validate="one_to_one")
    available_text_columns = [column for column in TEXT_COLUMNS if column in text.columns]
    if not args.skip_text_scores:
        frame = frame.merge(text[["cik", "fiscal_year", *available_text_columns]], on=["cik", "fiscal_year"], how="left")
    frame["split"] = np.where(frame["fiscal_year"] >= args.test_start_year, "test", "train")
    frame = add_labels(frame, censor_end)
    within_year_window = frame["fiscal_year"].between(args.sample_start_year, args.sample_end_year, inclusive="both")
    valid_filing = pd.to_datetime(frame["filing_date"], errors="coerce").notna()
    timing_eligible = (
        within_year_window & valid_filing & frame["post_default_filing_flag"].eq(0)
        & frame["label_observed_next_1y"].eq(1)
    )
    if args.skip_text_scores:
        frame["full_public_1y_text_sample"] = timing_eligible.astype(int)
    else:
        usable_text = pd.to_numeric(frame.get("usable_text_flag"), errors="coerce").eq(1)
        lm_available = pd.to_numeric(frame.get("distress_score_0_100_LM"), errors="coerce").notna()
        pb_available = pd.to_numeric(frame.get("stress_event_5pillar_equal_no_trigger"), errors="coerce").notna()
        frame["full_public_1y_text_sample"] = (
            timing_eligible & usable_text & lm_available & pb_available
        ).astype(int)
    frame = add_factors(frame)
    frame["brd_asset_threshold_usd"] = frame["fiscal_year"].map(
        lambda year: 100_000_000 * CPI_ANNUAL_AVERAGE.get(int(year), float("nan")) / CPI_ANNUAL_AVERAGE[1980]
        if pd.notna(year) else float("nan")
    )
    observed_size_eligible = (
        frame["analysis_assets"].notna()
        & frame["brd_asset_threshold_usd"].notna()
        & frame["analysis_assets"].ge(frame["brd_asset_threshold_usd"])
    )
    case_membership_eligible = frame["target_default_next_1y"].eq(1) & frame["analysis_assets"].isna()
    frame["brd_size_eligibility_basis"] = np.where(
        observed_size_eligible,
        "filing_assets_meet_inflation_adjusted_threshold",
        np.where(case_membership_eligible, "BRD_case_membership_with_missing_filing_assets", "not_eligible_or_unobservable"),
    )
    frame["brd_size_eligible_flag"] = (observed_size_eligible | case_membership_eligible).astype(int)
    frame["primary_1y_text_sample"] = (
        frame["full_public_1y_text_sample"].eq(1) & frame["brd_size_eligible_flag"].eq(1)
    ).astype(int)
    frame, treatment_parameters = add_treatments(frame)
    frame = add_quality_flags(frame)
    primary_mask = frame["primary_1y_text_sample"].eq(1)
    summary: Dict[str, object] = {
        "financial_features_source": str(args.financial_features), "manifest_source": str(args.manifest),
        "text_scores_source": None if args.skip_text_scores else str(args.text_scores),
        "text_scores_skipped": bool(args.skip_text_scores),
        "companyfacts_directory": str(args.companyfacts_dir),
        "censor_end_date": censor_end.isoformat(),
        "censor_end_basis": "Florida-UCLA-LoPucki database snapshot date encoded in the source filename",
        "sample_start_year": args.sample_start_year, "sample_end_year": args.sample_end_year,
        "test_start_year": args.test_start_year, "financial_rows_raw": int(financial.shape[0]),
        "manifest_rows_raw": int(manifest.shape[0]), "text_rows_raw": int(text.shape[0]),
        "financial_duplicate_key_rows": financial_duplicates, "manifest_duplicate_key_rows": manifest_duplicates,
        "text_duplicate_key_rows": text_duplicates, "matched_10k_rows": int(manifest_subset.shape[0]),
        "within_year_window_rows": int(within_year_window.sum()),
        "valid_timing_rows": int((within_year_window & valid_filing & frame["post_default_filing_flag"].eq(0)).sum()),
        "observed_1y_rows": int((within_year_window & valid_filing & frame["post_default_filing_flag"].eq(0) & frame["label_observed_next_1y"].eq(1)).sum()),
        "primary_1y_text_rows": int(primary_mask.sum()),
        "primary_1y_events": int(frame.loc[primary_mask, "target_default_next_1y"].sum()),
        "primary_1y_firms": int(frame.loc[primary_mask, "cik"].nunique()),
        "companyfacts_files_missing": int(frame["companyfacts_file_missing"].fillna(1).sum()),
        "full_public_1y_text_rows": int(frame["full_public_1y_text_sample"].sum()),
        "full_public_1y_events": int(frame.loc[frame["full_public_1y_text_sample"].eq(1), "target_default_next_1y"].sum()),
        "brd_universe_definition_url": BRD_DEFINITION_URL,
        "cpi_source_url": BRD_CPI_SOURCE_URL,
        "brd_threshold_formula": "100,000,000 * annual average CPIAUCSL(fiscal year) / annual average CPIAUCSL(1980)",
        "treatment_policy": {
            "data": "no pre-imputation; exact-accession facts; deterministic accounting-identity recovery only",
            "primary_model_candidate": "training-only median after monotonic transformation plus explicit missing indicators",
            "robustness": "complete-case models and training-only 1st/99th percentile clipping after transformation",
            "debt": "require reported total debt or both current and noncurrent components",
            "outliers": "retain valid observations; use monotonic transforms; do not delete solely because a ratio is extreme",
        },
    }
    write_outputs(frame, treatment_parameters, args.output_dir, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
