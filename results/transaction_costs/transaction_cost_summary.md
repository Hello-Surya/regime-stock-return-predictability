# Turnover and Transaction-Cost Analysis

## Methodology
Authoritative M8 assignments are reused unchanged (tie broken deterministically by PERMNO). EW is 1/N and VW uses positive, finite lagged formation ME. Previous targets drift through the prior realized holding-period returns before rebalance.
One-sided turnover equals 0.5 times the L1 target-minus-pretrade distance; each newly cash-funded leg incurs initial turnover 1. The D10-D1 spread is per $1 long and $1 short and incurs costs on both legs: 0.005 × (turnover D10 + turnover D1).
The D1_net field is the **long-only** D1 net return; D1_short_leg_net_contribution is the actual short-side contribution to D10-D1 net. Therefore D10_net - D1_net is **not** the net long-short spread.
Gross and net cumulative lines use M8 cumulative arithmetic returns, not compounded wealth.

## 50-basis-point overall and LOW/MIDDLE/HIGH results
- Elastic Net EW OVERALL: N=311, turnover=0.256, gross=0.3401%, net=0.2119%, net HAC t=0.66, p=0.5067.
- Elastic Net EW LOW: N=104, turnover=0.176, gross=0.4864%, net=0.3985%, net HAC t=1.43, p=0.1533.
- Elastic Net EW MIDDLE: N=108, turnover=0.263, gross=0.6702%, net=0.5389%, net HAC t=0.99, p=0.3201.
- Elastic Net EW HIGH: N=99, turnover=0.334, gross=-0.1739%, net=-0.3409%, net HAC t=-0.51, p=0.6130.
- Elastic Net VW OVERALL: N=311, turnover=0.243, gross=-0.0734%, net=-0.1952%, net HAC t=-0.61, p=0.5403.
- Elastic Net VW LOW: N=104, turnover=0.132, gross=0.3445%, net=0.2785%, net HAC t=0.91, p=0.3615.
- Elastic Net VW MIDDLE: N=108, turnover=0.234, gross=-0.3136%, net=-0.4306%, net HAC t=-0.93, p=0.3503.
- Elastic Net VW HIGH: N=99, turnover=0.371, gross=-0.2504%, net=-0.4360%, net HAC t=-0.60, p=0.5506.
- XGBoost EW OVERALL: N=311, turnover=0.481, gross=0.6153%, net=0.3746%, net HAC t=1.16, p=0.2464.
- XGBoost EW LOW: N=104, turnover=0.400, gross=0.8133%, net=0.6135%, net HAC t=2.45, p=0.0143.
- XGBoost EW MIDDLE: N=108, turnover=0.482, gross=1.2009%, net=0.9601%, net HAC t=1.80, p=0.0714.
- XGBoost EW HIGH: N=99, turnover=0.567, gross=-0.2316%, net=-0.5152%, net HAC t=-0.77, p=0.4388.
- XGBoost VW OVERALL: N=311, turnover=0.572, gross=0.0406%, net=-0.2453%, net HAC t=-0.64, p=0.5253.
- XGBoost VW LOW: N=104, turnover=0.505, gross=0.1708%, net=-0.0820%, net HAC t=-0.26, p=0.7964.
- XGBoost VW MIDDLE: N=108, turnover=0.568, gross=0.0888%, net=-0.1952%, net HAC t=-0.32, p=0.7476.
- XGBoost VW HIGH: N=99, turnover=0.646, gross=-0.1488%, net=-0.4717%, net HAC t=-0.56, p=0.5783.

## Three-state net regime differences (Holm primary, BH FDR secondary)
- Elastic Net EW MIDDLE - LOW: difference=0.00140, raw p=0.7969, Holm p=0.7969.
- Elastic Net EW HIGH - LOW: difference=-0.00739, raw p=0.2854, Holm p=0.7151.
- Elastic Net EW HIGH - MIDDLE: difference=-0.00880, raw p=0.2384, Holm p=0.7151.
- Elastic Net VW MIDDLE - LOW: difference=-0.00709, raw p=0.1743, Holm p=0.5230.
- Elastic Net VW HIGH - LOW: difference=-0.00715, raw p=0.3668, Holm p=0.7336.
- Elastic Net VW HIGH - MIDDLE: difference=-0.00005, raw p=0.9951, Holm p=0.9951.
- XGBoost EW MIDDLE - LOW: difference=0.00347, raw p=0.5253, Holm p=0.5253.
- XGBoost EW HIGH - LOW: difference=-0.01129, raw p=0.1029, Holm p=0.2058.
- XGBoost EW HIGH - MIDDLE: difference=-0.01475, raw p=0.0462, Holm p=0.1386.
- XGBoost VW MIDDLE - LOW: difference=-0.00113, raw p=0.8638, Holm p=1.0000.
- XGBoost VW HIGH - LOW: difference=-0.00390, raw p=0.6536, Holm p=1.0000.
- XGBoost VW HIGH - MIDDLE: difference=-0.00276, raw p=0.7754, Holm p=1.0000.

## Break-even one-way costs (descriptive)
- Elastic Net EW: 132.67 bps
- Elastic Net VW: undefined
- XGBoost EW: 127.80 bps
- XGBoost VW: 7.10 bps

## Limitations
50 bps is a stylized proportional one-way trading-cost robustness assumption, **not** a measured execution cost. Spread, liquidity, nonlinear market impact, trade size and period can all affect realizable implementation costs. Short borrow/locate fees, rebates, financing, and nonlinear market impact are not separately modeled. Positive modeled net returns do not establish an implementable trading strategy.
