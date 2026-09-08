# WRDS CRSP Data Pipeline

The project uses live WRDS CRSP monthly extraction while keeping authentication outside the repository.

The extractor discovers accessible WRDS libraries at runtime and prefers the current CRSP CIZ monthly format (`msf_v2`). Real WRDS pulls are cached as Parquet under `data/raw/`, which is gitignored.

The live connection has been validated against `crsp.msf_v2` and successfully returns common-stock observations including PERMNO, ticker, date, return, price, shares outstanding, and market equity.

Compustat/CCM book-to-market enters in the later complete-data construction stage and is not approximated or fabricated in the current partial-feature specification.
