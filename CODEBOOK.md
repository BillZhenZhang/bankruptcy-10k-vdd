# Codebook

This file summarizes the main variables created by the replication code. Exact column names can differ across intermediate files, but the definitions below match the variables used in the final manuscript.

## Outcome

- `bankruptcy_within_1y`: indicator equal to one if the firm files for bankruptcy within one year after the 10-K filing date.

## Sample identifiers

- `cik`: SEC Central Index Key.
- `accession`: SEC filing accession number.
- `fiscal_year`: fiscal year associated with the 10-K filing.
- `filing_date`: 10-K filing date.
- `sic`: Standard Industrial Classification code.

## Accounting variables

- `size`: natural log of total assets restated in 1980 dollars.
- `tlta`: total liabilities divided by total assets.
- `wcta`: working capital divided by total assets.
- `clca`: current liabilities divided by current assets.
- `nita`: net income divided by total assets.
- `oeneg`: indicator equal to one if shareholders' equity is negative or liabilities exceed assets.

## VDD variables

- `gc_flag`: indicator equal to one when Item 7 contains a scored going-concern uncertainty disclosure after context filtering.
- `cov_flag`: indicator equal to one when Item 7 contains a scored covenant noncompliance disclosure after context filtering.
- `fw_flag`: indicator equal to one when Item 7 contains a scored lender forbearance or covenant-waiver disclosure after context filtering.
- `any_vdd_flag`: indicator equal to one when any retained VDD family is scored.

## Text benchmark variables

- Loughran--McDonald dictionary rates: negative, positive, uncertainty, litigious, strong modal, weak modal, and constraining word rates per 1000 words.
- Disclosure controls: Item 7 length, Item 1A length, sentence-length measures, complex-word share, and Gunning Fog index.
- TF-IDF benchmark: Item 7 unigram features with training-period regularization tuning.

## Notes

The final models use training-period transformations, training-sample median fill after transformation, missingness indicators, and standard scaling. Class-weighted logistic scores are used for ranking and are not calibrated probability estimates.
