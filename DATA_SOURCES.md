# Data Sources

The analysis combines public SEC filing data, SEC Company Facts accounting data, and bankruptcy-event information from the Florida--UCLA--LoPucki Bankruptcy Research Database.

## Required inputs

1. **SEC annual filings**
   - Source: SEC EDGAR.
   - Used for: Form 10-K filing dates, accession numbers, and Item 7 text.
   - Raw filing text is not redistributed here.

2. **SEC Company Facts**
   - Source: SEC EDGAR Company Facts API.
   - Used for: accession-aligned accounting variables.
   - Local files are expected as per-CIK JSON files.

3. **Florida--UCLA--LoPucki Bankruptcy Research Database**
   - Source: Florida--UCLA--LoPucki Bankruptcy Research Database.
   - Used for: bankruptcy-event labels and BRD large-company universe alignment.
   - The BRD file is not redistributed here.

4. **Inflation series for BRD threshold alignment**
   - Source: CPIAUCSL, Federal Reserve Economic Data.
   - Used for: restating the BRD large-company asset threshold and size variable.

## Excluded data

This release does not include full raw filings, extracted full-text sections, sentence-level occurrence extracts, annotation packets, or final modeling datasets. Those files are either large, derived directly from public filings, or not necessary to inspect the scoring rules and analysis code.
