# WRDS Data Pipeline

The project uses live WRDS extraction while keeping authentication outside the repository. Raw licensed data are cached locally as Parquet and are excluded from public version control.

## Validated runtime sources

The production build resolved:

- CRSP monthly common stocks: `crsp.msf_v2` (CIZ);
- Compustat annual fundamentals: `comp.funda`;
- CCM link history: `crsp.ccmxpf_lnkhist`; and
- Cboe VIX: `cboe_all.cboe`.

The source resolvers inspect the libraries, tables, and relevant columns exposed by the connected WRDS account instead of relying only on historical tutorial table names.

## Local validation

Validate accounting sources with:

```powershell
.\.venv\Scripts\python.exe scripts\validate_accounting_sources.py
```

The script connects using the normal WRDS authentication workflow, verifies Compustat and CCM access, prints the resolved table names and relevant fields, pulls tiny samples, and closes the connection cleanly.

Do not place a WRDS password in source code, YAML, Git, or chat.

## Complete predictor panel

Quick construction:

```powershell
.\.venv\Scripts\python.exe scripts\build_modeling_panel.py --quick
```

Production construction:

```powershell
.\.venv\Scripts\python.exe scripts\build_modeling_panel.py --production
```

Quick and production modes use the same accounting methodology. Quick mode changes only computational scale.

The production panel is written locally to:

`data/processed/modeling_panel.parquet`

and the quick panel to:

`data/processed/modeling_panel_quick.parquet`.

These files contain licensed row-level WRDS-derived data and must not be committed publicly.

Aggregate, non-confidential diagnostics are written under:

`results/data_validation/quick/`

and

`results/data_validation/production/`.

See `docs/data_sources.md` for the exact book-equity, CCM, December-market-equity, June-assignment, and unit conventions.
