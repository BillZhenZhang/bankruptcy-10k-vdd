# File Manifest

## Documentation

- `README.md`: overview, publication citation, and usage notes.
- `PUBLISHED_ARTICLE.md`: published article DOI, URL, and citation details.
- `CITATION.bib`: BibTeX citation for the published article.
- `DATA_SOURCES.md`: source-data requirements and excluded files.
- `CODEBOOK.md`: variable definitions.
- `requirements.txt`: Python dependencies.

## Configuration

- `config/vdd_rule_registry.csv`: fixed phrase-pattern registry used by the VDD scoring script.

## Scripts

- `scripts/build_filing_aligned_dataset.py`: builds accession-aligned accounting and outcome panel.
- `scripts/build_all_industry_companies.py`: creates the broader nonfinancial sample and sample audits.
- `scripts/evaluate_accounting_factor_sets.py`: evaluates accounting baseline factor sets.
- `scripts/score_vdd_dictionary.py`: applies VDD phrase patterns and context filters to Item 7 text.
- `scripts/vdd_context_rules.py`: deterministic context-suppression rules imported by the VDD scoring script.
- `scripts/run_benchmark_analysis.py`: runs LM, disclosure-control, TF-IDF, and incremental VDD benchmark checks.
- `scripts/run_robustness_checks.py`: runs near-event, SIC, family-decomposition, and related robustness checks.
- `scripts/run_model_diagnostics_validation.py`: runs multicollinearity, link-function, and grouped-validation diagnostics.

## Empty placeholders

- `data/.gitkeep`: placeholder only. Data files are intentionally not distributed in this repository.
