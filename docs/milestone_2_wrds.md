# Milestone 2 — WRDS CRSP Pipeline

Milestone 2 activates real CRSP monthly extraction while keeping authentication outside the repository.

## Current CRSP format

CRSP discontinued updated legacy SIZ releases after the December 2024 release. The extractor therefore discovers the user's accessible WRDS libraries at runtime and prefers the current CIZ monthly format (`msf_v2`). It can fall back to the legacy `msf` + name-history tables if that is the only accessible interface.

For CIZ data, the research universe corresponding to legacy SHRCD 10/11 is implemented using WRDS' current mapping:

- `ShareType = 'NS'`
- `SecurityType = 'EQTY'`
- `SecuritySubType = 'COM'`
- `USIncFlg = 'Y'`
- `IssuerType in ('ACOR', 'CORP')`

The NYSE/AMEX/NASDAQ active regular-way filter uses:

- `PrimaryExch in ('N', 'A', 'Q')`
- `ConditionalType = 'RW'`
- `TradingStatusFlg = 'A'`

The pull is filtered on the WRDS server by date and optional ticker/price constraints before data are transferred.

## Authentication

No function accepts a password. `wrds.Connection()` uses the user's normal WRDS-supported authentication mechanism. Do not commit credentials, `.pgpass`, `.env`, keys, or passwords.

## Cache

Real WRDS pulls are cached as Parquet under `data/raw/`, which is gitignored. Re-running the same extraction command reads the local cache unless `--refresh` is specified.

## Validation

`pull_wrds.py` generates a CSV under `data/interim/` containing sample dates, row counts, unique stocks/months, duplicate counts, missing rates, and selected return/price/market-equity quantiles.

## Scope

This milestone establishes CRSP market-data extraction and caching. Compustat/CCM book-to-market and the empirical VIX loader are later data-construction steps and are not fabricated here.
