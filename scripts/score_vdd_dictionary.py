#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RULE_SCRIPT = ROOT / "vdd_context_rules.py"
PRIMARY_FAMILIES = [
    "hard_going_concern",
    "hard_covenant_noncompliance",
    "hard_forbearance_waiver",
]
EXPLORATORY_FAMILIES = ["hard_unable_obligations"]
EXCLUDED_FAMILIES = [
    "hard_default_acceleration",
    "hard_bankruptcy_restructuring",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--occurrence-inventory",
        type=Path,
        default=Path("data/vdd_context_occurrence_inventory.csv"),
    )
    parser.add_argument(
        "--sentence-audit",
        type=Path,
        default=Path("data/vdd_sentence_extraction_audit.csv"),
    )
    parser.add_argument(
        "--analysis-dataset",
        type=Path,
        default=Path("data/revised_analysis_dataset.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/vdd_scores"),
    )
    return parser.parse_args()


def load_rule_module(path: Path):
    spec = importlib.util.spec_from_file_location("channel1_rules", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load context rules from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_mask_map(ciks: pd.Series) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for cik in sorted(set(ciks.dropna().astype(str))):
        masked = "FIRM_" + hashlib.sha256(cik.encode("utf-8")).hexdigest()[:12]
        mapping[masked] = cik
    return mapping


def phrase_slug(phrase: str) -> str:
    return phrase.removeprefix("hard_")


def attach_predictions(channel1: pd.DataFrame, rule_module) -> pd.DataFrame:
    records = []
    for row in channel1.to_dict("records"):
        predicted, rule_id = rule_module.classify(row)
        records.append({
            "predicted_binary": predicted,
            "rule_id": rule_id,
        })
    predicted_frame = pd.DataFrame(records)
    out = pd.concat([channel1.reset_index(drop=True), predicted_frame], axis=1)
    out["risk_active_flag"] = out["predicted_binary"].eq("RISK_ACTIVE").astype(int)
    out["primary_family_flag"] = out["phrase"].isin(PRIMARY_FAMILIES).astype(int)
    out["primary_risk_active_flag"] = (
        out["risk_active_flag"].eq(1) & out["primary_family_flag"].eq(1)
    ).astype(int)
    out["exploratory_family_flag"] = out["phrase"].isin(EXPLORATORY_FAMILIES).astype(int)
    out["exploratory_risk_active_flag"] = (
        out["risk_active_flag"].eq(1) & out["exploratory_family_flag"].eq(1)
    ).astype(int)
    return out


def build_firm_year_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["cik", "fiscal_year"]
    base = predictions.groupby(keys).agg(
        channel1_occurrence_count=("occurrence_id", "size"),
        channel1_risk_active_count=("risk_active_flag", "sum"),
        channel1_primary_occurrence_count=("primary_family_flag", "sum"),
        channel1_primary_active_count=("primary_risk_active_flag", "sum"),
        channel1_exploratory_active_count=("exploratory_risk_active_flag", "sum"),
    ).reset_index()
    base["channel1_any_risk_active"] = base["channel1_risk_active_count"].gt(0).astype(int)
    base["channel1_primary_any_active"] = base["channel1_primary_active_count"].gt(0).astype(int)
    base["channel1_exploratory_any_active"] = base["channel1_exploratory_active_count"].gt(0).astype(int)
    for phrase in PRIMARY_FAMILIES + EXPLORATORY_FAMILIES + EXCLUDED_FAMILIES:
        slug = phrase_slug(phrase)
        subset = predictions[predictions["phrase"].eq(phrase)].groupby(keys).agg(
            active_count=("risk_active_flag", "sum"),
            occurrence_count=("occurrence_id", "size"),
        ).reset_index()
        subset[f"{slug}_active_count"] = subset.pop("active_count")
        subset[f"{slug}_occurrence_count"] = subset.pop("occurrence_count")
        subset[f"{slug}_any_active"] = subset[f"{slug}_active_count"].gt(0).astype(int)
        base = base.merge(subset, on=keys, how="left", validate="one_to_one")
    fill_zero_columns = [column for column in base.columns if column not in {"cik", "fiscal_year"}]
    base[fill_zero_columns] = base[fill_zero_columns].fillna(0)
    integer_columns = [column for column in fill_zero_columns if column.endswith("_count") or column.endswith("_active") or column.endswith("_flag")]
    for column in integer_columns:
        base[column] = base[column].astype(int)
    return base.sort_values(keys).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    rule_module = load_rule_module(RULE_SCRIPT)
    sentence_audit = pd.read_csv(args.sentence_audit, dtype={"cik": str}, usecols=["cik"], low_memory=False)
    analysis = pd.read_csv(args.analysis_dataset, dtype={"cik": str}, low_memory=False)
    mask_map = build_mask_map(pd.concat([sentence_audit["cik"], analysis["cik"]], ignore_index=True))
    inventory = pd.read_csv(args.occurrence_inventory, dtype={"fiscal_year": int}, low_memory=False)
    channel1 = inventory[inventory["feature_source"].eq("channel1_pilot")].copy()
    channel1["cik"] = channel1["firm_id_masked"].map(mask_map)
    unresolved = channel1["cik"].isna()
    if unresolved.any():
        sample = channel1.loc[unresolved, "firm_id_masked"].drop_duplicates().tolist()[:5]
        raise ValueError(f"Unable to map {int(unresolved.sum())} Channel 1 rows back to CIKs; sample masks={sample}")
    predictions = attach_predictions(channel1, rule_module)
    predictions["retained_primary_family"] = predictions["phrase"].isin(PRIMARY_FAMILIES).astype(int)
    predictions["excluded_primary_family"] = predictions["phrase"].isin(EXCLUDED_FAMILIES).astype(int)
    firm_year = build_firm_year_scores(predictions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "channel1_occurrence_predictions.csv", index=False)
    firm_year.to_csv(args.output_dir / "channel1_firm_year_scores.csv", index=False)

    family_summary = (
        predictions.groupby("phrase")
        .agg(
            occurrences=("occurrence_id", "size"),
            firms=("cik", "nunique"),
            firm_years=("occurrence_id", lambda values: values.shape[0]),
            risk_active_occurrences=("risk_active_flag", "sum"),
        )
        .reset_index()
        .sort_values("phrase")
    )
    family_rows = []
    for row in family_summary.to_dict("records"):
        row["status"] = (
            "primary"
            if row["phrase"] in PRIMARY_FAMILIES
            else "exploratory"
            if row["phrase"] in EXPLORATORY_FAMILIES
            else "excluded"
        )
        family_rows.append(row)
    summary = {
        "rule_script": str(RULE_SCRIPT),
        "rule_script_sha256": hashlib.sha256(RULE_SCRIPT.read_bytes()).hexdigest(),
        "occurrence_inventory": str(args.occurrence_inventory),
        "sentence_audit": str(args.sentence_audit),
        "analysis_dataset": str(args.analysis_dataset),
        "channel1_occurrences": int(predictions.shape[0]),
        "channel1_firms": int(predictions["cik"].nunique()),
        "channel1_firm_years": int(predictions[["cik", "fiscal_year"]].drop_duplicates().shape[0]),
        "primary_active_firm_years": int(firm_year["channel1_primary_any_active"].sum()),
        "primary_active_occurrences": int(predictions["primary_risk_active_flag"].sum()),
        "exploratory_active_firm_years": int(firm_year["channel1_exploratory_any_active"].sum()),
        "family_summary": family_rows,
    }
    (args.output_dir / "channel1_score_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
