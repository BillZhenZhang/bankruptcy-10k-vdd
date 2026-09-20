# Bankruptcy 10-K VDD Replication Code

This repository contains the code and documentation used to reproduce the data construction, text scoring, benchmark tests, and robustness checks for the published paper:

> Zhang, Z.; Zheng, M.; Zhang, T.; Lin, L.; Lin, L. **Bankruptcy Prediction from 10-K Narratives: Evidence from Interpretable Text Scores and Accounting Baselines.** *Risks* 2026, 14(8), 179. https://doi.org/10.3390/risks14080179

Published article: https://www.mdpi.com/2227-9091/14/8/179

The repository is intentionally code-first. It does not redistribute full raw 10-K filings, full Item 7 text, occurrence-level sentence extracts, annotation packets, or the final modeling dataset. Raw filings can be regenerated from SEC EDGAR accessions, and bankruptcy events are obtained from the Florida--UCLA--LoPucki Bankruptcy Research Database.

## What is included

- `scripts/`: code for accession-aligned sample construction, VDD scoring, benchmark tests, and robustness checks.
- `scripts/vdd_context_rules.py`: deterministic sentence-level context filters used by the VDD scoring code.
- `config/vdd_rule_registry.csv`: fixed candidate phrase patterns used by the VDD scoring code.
- `CODEBOOK.md`: definitions of the main variables and outputs used in the paper.
- `DATA_SOURCES.md`: source data needed to rerun the pipeline.
- `FILE_MANIFEST.md`: description of the included files.
- `CITATION.bib`: BibTeX entry for the published article.
- `PUBLISHED_ARTICLE.md`: DOI, article URL, and citation details.
- `requirements.txt`: Python package requirements used by the scripts.

## What is not included

The following files are not included because they are large, derived from public filings, third-party sourced, or contain sentence-level filing excerpts:

- full raw SEC 10-K filings;
- full extracted Item 1, Item 1A, or Item 7 text;
- occurrence-level sentence extracts;
- human annotation packets;
- final modeling datasets and prediction-score files.

The scripts are provided so that these files can be regenerated from the public and third-party source materials described in `DATA_SOURCES.md`.

## Typical workflow

1. Download SEC Company Facts files and matched 10-K filing metadata.
2. Prepare the BRD bankruptcy-event file.
3. Run the accession-aligned data construction script.
4. Extract Item 7 text from annual filings.
5. Run the VDD scoring script using `config/vdd_rule_registry.csv`.
6. Run the benchmark, robustness, and diagnostic scripts.

The scripts assume local input/output paths. Before running them on another machine, update the path variables near the top of each script to point to your local SEC, BRD, filing-text, and output folders. Some internal variable names use the project-development label `channel1`; in the manuscript, these are the VDD indicators.

## Data availability language

A concise manuscript statement can read:

> Code and documentation for the replication workflow are available at this repository. Raw SEC filings are publicly available through EDGAR, and bankruptcy-event information is obtained from the Florida--UCLA--LoPucki Bankruptcy Research Database. Full raw 10-K text and occurrence-level sentence extracts are not redistributed, but they can be regenerated from public filing accessions using the included code.
