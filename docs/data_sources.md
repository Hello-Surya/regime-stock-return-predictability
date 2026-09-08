# Data Sources and Predictor Construction

## Overview

The complete baseline predictor panel combines CRSP monthly U.S. common-stock data, Compustat annual fundamentals, the CRSP/Compustat Merged link history, and Cboe VIX data. Raw licensed WRDS observations and the row-level processed modeling panel remain local and are excluded from public version control.

The validated production build spans January 1990 through December 2025. Construction uses pre-sample history beginning in 1988 for CRSP and 1987 for Compustat so that momentum, prior-December market equity, and June accounting assignments are available at the start of the research sample.

## CRSP monthly stock data

The runtime-resolved production source is:

`crsp.msf_v2` (current CRSP CIZ monthly format).

The extraction retains U.S. ordinary common equity listed on the NYSE, AMEX, or NASDAQ using the available CIZ classifications:

- `sharetype = 'NS'`;
- `securitytype = 'EQTY'`;
- `securitysubtype = 'COM'`;
- `usincflg = 'Y'`;
- `issuertype IN ('ACOR', 'CORP')`;
- `primaryexch IN ('N', 'A', 'Q')`;
- `conditionaltype = 'RW'`; and
- `tradingstatusflg = 'A'`.

The core feature pipeline normalizes price to its absolute value and constructs monthly market equity as

`market_equity = abs(prc) * shrout`.

CRSP shares outstanding are in thousands, so this market-equity variable is in USD thousands. `log_me` is the natural logarithm of strictly positive market equity. Nonpositive market equity produces missing `log_me`.

`me_lag` is the previous calendar month's market equity. Calendar gaps are not treated as adjacent observations.

The next-month target is the return in the next actual calendar month. Momentum 12–2 compounds returns from months t-12 through t-2 and excludes months t and t-1.

The $5 price screen is applied at feature month t after the full return history has been assembled so that the t+1 price does not determine whether the t observation survives.

## Compustat annual fundamentals

The runtime-resolved production source is:

`comp.funda`.

The production extraction uses annual records from the buffered Compustat history and, where the source exposes the fields, applies the standard filters:

- `indfmt = 'INDL'`;
- `datafmt = 'STD'`;
- `popsrc = 'D'`;
- `consol = 'C'`; and
- `curcd = 'USD'`.

Relevant fields include `gvkey`, `datadate`, `fyear`, `fyr`, `seq`, `ceq`, `at`, `lt`, `txditc`, `txdb`, `itcb`, `pstkrv`, `pstkl`, and `pstk`.

If multiple standard annual records exist for the same GVKEY and calendar fiscal-year-end year, the latest fiscal-year-end observation is retained.

### Stockholders' equity

The baseline hierarchy is:

1. `SEQ`;
2. otherwise `CEQ + PSTK`; and
3. otherwise `AT - LT`.

If none of these constructions is available, stockholders' equity remains missing.

### Preferred stock

Preferred stock uses:

1. `PSTKRV`;
2. otherwise `PSTKL`; and
3. otherwise `PSTK`.

If all preferred-stock fields are missing, the baseline explicitly assigns preferred stock equal to zero.

### Deferred taxes and investment tax credit

Deferred taxes / investment tax credit use:

1. `TXDITC`;
2. otherwise `TXDB + ITCB`, treating a missing component within this fallback sum as zero.

If all deferred-tax / investment-tax-credit fields are missing, the baseline explicitly assigns this component equal to zero.

### Book equity

Book equity is

`book_equity = stockholders_equity + deferred_taxes_itc - preferred_stock`.

Compustat accounting values are in USD millions. Book equity is retained for lineage even when zero or negative, but only strictly positive book equity can generate a valid baseline book-to-market ratio.

## CRSP/Compustat Merged linkage

The runtime-resolved production source is:

`crsp.ccmxpf_lnkhist`.

The baseline keeps:

- link types `LC` and `LU`; and
- primary indicators `P` and `C`.

A link must be effective on the Compustat `datadate`. Missing link start or end dates are treated as open-ended on the corresponding side.

Duplicate links for a GVKEY / characteristic-year / PERMNO are resolved deterministically with the following priority:

1. primary indicator `P` before `C`;
2. link type `LC` before `LU`; and
3. the latest applicable link start date.

If a PERMNO has competing GVKEY assignments in the same characteristic year, the same link priorities are applied, followed by the latest accounting date and GVKEY as a deterministic final tie-breaker.

Compustat is never merged to CRSP by ticker or company name.

## Accounting-information timing

The baseline uses an explicit conservative June convention rather than immediately forward-filling newly observed Compustat records.

For a Compustat fiscal-year end in calendar year y-1:

- the characteristic year is y;
- the accounting record first becomes eligible at the end of June y;
- it is assigned to monthly observations from June y through May y+1.

Thus a fiscal-year-end observation in calendar 2023 can first enter the monthly predictor panel in June 2024 and remains the annual accounting characteristic through May 2025.

A hard timing assertion prevents an accounting record from entering any monthly observation before its allowed June date.

## December market equity and multiple securities

For a characteristic year y, book equity is paired with CRSP market equity from December y-1.

The monthly CRSP feature panel stores market equity in USD thousands. Required December security-level market equity is divided by 1,000 to convert it to USD millions before it is combined with Compustat book equity.

Where a GVKEY has multiple eligible linked PERMNOs represented in the research universe, December market equity is summed across those securities to form the firm-level denominator. Firm-level book equity is therefore not mechanically divided by each share class separately.

If any represented linked security required for that firm-year denominator lacks its December market-equity observation, the denominator is treated as incomplete and book-to-market remains missing rather than being guessed.

## Book-to-market

For valid observations,

`book_to_market = book_equity / me_dec`,

where both numerator and denominator are in USD millions.

A valid baseline B/M observation requires:

- strictly positive book equity;
- strictly positive prior-December firm market equity; and
- a complete required December denominator.

Nonpositive or unavailable inputs produce missing B/M.

No winsorization, trimming, or predictive-performance-driven accounting adjustment is performed during data construction.

## VIX and volatility regimes

The runtime-resolved production VIX source is:

`cboe_all.cboe`.

Daily VIX observations are converted to a monthly series using the final observed VIX close in each calendar month. The volatility regime at month t is HIGH when VIX at t exceeds the expanding historical median calculated using observations through t, and LOW otherwise.

Future VIX observations do not enter the regime at t.

## Production validation

The validated 1990–2025 production build contains:

- 2,140,091 stock-month observations;
- 18,898 unique PERMNOs;
- 1,818,006 valid book-to-market observations;
- 1,798,891 observations with size, book-to-market, and momentum 12–2 all present; and
- zero duplicate `permno`/`date` rows.

After sequentially imposing a valid next-month target, the $5 feature-month price screen, size, momentum, CCM match, positive book equity, and valid book-to-market, the complete baseline sample flow contains 1,340,823 stock-months.

The production B/M median is approximately 0.558 and the 99.9th percentile is approximately 11.60. The automated unit-sanity check passes. Extreme observations are reported rather than automatically deleted.
