# Data Construction Validation Summary

This report describes the CRSP-Compustat-CCM baseline predictor panel. It contains aggregate diagnostics only; licensed row-level WRDS data are not included.

## Panel coverage

- Stock-month observations: 2,140,091
- Unique PERMNOs: 18,898
- Sample: 1990-01-31 through 2025-12-31
- Valid book-to-market months: 1,818,006
- Complete baseline predictor rows: 1,798,891
- Duplicate PERMNO/date rows: 0

## Coverage fractions

- log_me: 0.9996
- mom_12_2: 0.9216
- book_to_market: 0.8495
- next_month_return: 0.9904
- regime: 1.0000

## Link and unit checks

- CCM rows retained after baseline link filters: 33,324
- Accounting firm-years unmatched to a valid CCM link: 140,411
- GVKEY-years with multiple eligible PERMNOs: 0
- GVKEY-years with incomplete December ME: 11,246
- B/M unit sanity: pass
- B/M median: 0.5580521349402687
- B/M 99.9th percentile: 11.596283521332463

## Distribution percentiles

| variable | n | missing | min | max | p0.1 | p1 | p5 | p25 | p50 | p75 | p95 | p99 | p99.9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| book_equity | 1898870 | 241221 | -96620 | 732931 | -2574 | -159.862 | 1.161 | 23.936 | 107.079 | 510.709 | 4948.7 | 22924 | 128159 |
| me_dec | 1917474 | 222617 | 0.15 | 3766499.857 | 0.646875 | 2.32875 | 7.1114 | 45.73794 | 216.7005 | 1150.7307 | 12590.09834 | 66253.02225 | 297828.2969 |
| book_to_market | 1818006 | 322085 | 7.782916494e-06 | 53.96362068 | 0.004386180177 | 0.02961350186 | 0.09404701873 | 0.2994005276 | 0.5580521349 | 0.9249617639 | 1.999694899 | 4.202379594 | 11.59628352 |
| log_me | 2139259 | 832 | 2.251291799 | 22.31688315 | 6.246106765 | 7.628272631 | 8.817371855 | 10.69090056 | 12.21981775 | 13.89457117 | 16.30859473 | 17.9911837 | 19.56342978 |
| mom_12_2 | 1972307 | 167784 | -0.9999999994 | 105.711191 | -0.9705882713 | -0.8712572315 | -0.6693065136 | -0.240538912 | 0.03260869953 | 0.3140500225 | 1.132483174 | 2.783781152 | 8.062898239 |
