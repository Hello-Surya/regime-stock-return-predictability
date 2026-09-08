# Data Construction Validation Summary

This report describes the CRSP-Compustat-CCM baseline predictor panel. It contains aggregate diagnostics only; licensed row-level WRDS data are not included.

## Panel coverage

- Stock-month observations: 43,774
- Unique PERMNOs: 300
- Sample: 2012-01-31 through 2025-12-31
- Valid book-to-market months: 41,565
- Complete baseline predictor rows: 41,556
- Duplicate PERMNO/date rows: 0

## Coverage fractions

- log_me: 1.0000
- mom_12_2: 0.9994
- book_to_market: 0.9495
- next_month_return: 0.9931
- regime: 1.0000

## Link and unit checks

- CCM rows retained after baseline link filters: 33,324
- Accounting firm-years unmatched to a valid CCM link: 22
- GVKEY-years with multiple eligible PERMNOs: 0
- GVKEY-years with incomplete December ME: 346
- B/M unit sanity: pass
- B/M median: 0.3505365856927575
- B/M 99.9th percentile: 2.6596954648692726

## Distribution percentiles

| variable | n | missing | min | max | p0.1 | p1 | p5 | p25 | p50 | p75 | p95 | p99 | p99.9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| book_equity | 43406 | 368 | -17306 | 732931 | -14781 | -6328 | 257 | 4456.9 | 10677.5 | 24839 | 93404 | 231558 | 516425 |
| me_dec | 43406 | 368 | 128.58615 | 3766499.857 | 408.99252 | 2387.76285 | 7970.85156 | 16630.40229 | 32593.64522 | 71556.45636 | 240223.5394 | 525497.88 | 2902367.977 |
| book_to_market | 41565 | 2209 | 5.157662352e-05 | 5.713288504 | 0.002298137886 | 0.01300953117 | 0.05523725694 | 0.1850980436 | 0.3505365857 | 0.64812639 | 1.215562493 | 1.819685875 | 2.659695465 |
| log_me | 43774 | 0 | 9.945414936 | 22.31688315 | 12.44177935 | 14.64880379 | 15.80665799 | 16.68521002 | 17.37023856 | 18.14582672 | 19.35755474 | 20.35535072 | 21.84869644 |
| mom_12_2 | 43747 | 27 | -0.9685966163 | 5.380044807 | -0.7800002641 | -0.5222438059 | -0.2991665294 | -0.043884011 | 0.107539173 | 0.2659694308 | 0.5808345171 | 1.045807127 | 2.511105014 |
