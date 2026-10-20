"""Idempotently synchronize M10 glossary, manuscript, and README after production validation.

Only called after successful scientific checks and output generation. Does not edit
M8/M9 result files or the M9-generated tables.
"""
from __future__ import annotations

from pathlib import Path

MARKER = "<!-- Validated transaction-cost analysis -->"


def _insert_once(text: str, anchor: str, new_text: str, marker: str) -> str:
    if marker in text:
        return text
    if anchor not in text:
        raise RuntimeError(f"Documentation anchor missing: {anchor!r}")
    return text.replace(anchor, new_text + anchor, 1)


def update_transaction_cost_docs(root: Path) -> None:
    glossary = root / "docs" / "research_glossary.md"
    readme = root / "README.md"
    manuscript = root / "paper" / "sections" / "05_economic_value.tex"
    gitignore = root / ".gitignore"
    for p in (glossary, readme, manuscript):
        if not p.is_file():
            raise FileNotFoundError(p)
    raw = glossary.read_text(encoding="utf-8")
    index_marker = "| bp / bps | Basis points |"
    new_index = """| bp / bps | Basis points | One bp equals 0.0001 of capital; frozen 50-bp one-way cost is 0.005. | Current turnover/cost stage |
| TC | Transaction cost | One-way proportional trading fee charged to one-sided portfolio turnover. | Current turnover/cost stage |
"""
    new_index += "| L1 | L1 norm | Sum of absolute component differences in portfolio-weight turnover. | Current turnover/cost stage |\n"
    new_index += "| M8 | Milestone 8 | Frozen three-state portfolio formation stage. | Historical stage marker |\n"
    new_index += "| M9 | Milestone 9 | Frozen gross HAC statistical inference stage. | Historical stage marker |\n"
    new_index += "| M10 | Milestone 10 | Turnover and transaction-cost evaluation on frozen M8 portfolios. | Current evaluation stage |\n"
    new_index += "| SHA | Secure Hash Algorithm | SHA-256 hashes protect frozen inputs against mutation. | Current audit stage |\n"
    if index_marker not in raw:
        raw = raw.replace("\n\n**Acronym collisions:**", "\n" + new_index + "\n**Acronym collisions:**", 1)
    # The frozen three-state M8 sort breaks exact prediction ties by PERMNO.
    old = ("Current production economic-value module uses average percentile ranks and maps those "
           "percentiles to deciles, keeping identical predictions in the same portfolio. "
           "Intermediate deciles may be empty under large ties, while both extreme legs are required.")
    replacement = ("The authoritative Milestone 8 three-state production portfolios sort by prediction "
                   "and then PERMNO, assigning approximately equal membership to each decile. "
                   "The earlier baseline economic-value module's average-percentile/tie-preserving "
                   "rule is a historical, separately defined portfolio implementation and is not "
                   "used to recompute transaction-cost-stage memberships.")
    raw = raw.replace(old, replacement)
    section = """
## Validated Turnover and Trading-Cost Methodology

**Status:** Implemented on the frozen three-state production decile assignments. Actual quantitative findings require a completed local production run; methodology is separate from the older binary-regime cost exercise.

### Basis point (bp, bps)
One basis point is 0.01 percentage point of capital, equal to 0.0001 in decimal returns. The frozen one-way TC assumption is 50 bps = 0.50% = 0.005. The 0-bps scenario exactly reproduces M8 gross returns.

### Transaction cost (TC) / one-way transaction cost
A stylized proportional fee applied to traded portfolio notional. It is not a claim that true execution costs equal 50 bps. Financing, borrow fees, short rebates, locate fees and nonlinear market impact are not explicitly included.

### Portfolio rebalancing / turnover / portfolio turnover
At each formation month t, rebalancing sets the portfolio to the *existing saved* M8 decile assignments. For each **existing long-only leg**, the one-sided turnover is $0.5\\sum_i |w^{target}_{i,t}-w^{pretrade}_{i,t}|$, calculated on the union of entering, continuing and exiting PERMNOs; no constituent-count proxy is used. A leg initially formed from cash incurs a turnover of 1.0.

### Target weight
The desired post-trade portfolio weight. EW is $1/N$ over M8-eligible stocks. VW is positive finite `weighting_variable` (`me_lag`, lagged market equity from t-1) divided by its eligible-decile total. VW does not substitute EW weights for invalid market equity.

### Pre-trade weight / drifted weight
The weight carried into the current rebalance from the preceding month, adjusted by the return observed during that **preceding holding period**: $w^{pretrade}_{i,t} = w^{target}_{i,t-1}(1+R_{i,t})/\\sum_j w^{target}_{j,t-1}(1+R_{j,t})$. Here the saved assignment at t-1 records the realized calendar-t stock return. It is permissible at the end-of-t rebalance; current assignment's realized t+1 return is never used to form turnover at t. Missing or impossible drift returns fail closed.

### Traded notional / gross exposure / net exposure
`turnover_D10 + turnover_D1` measures **one-sided traded notional per $1 long plus $1 short**, not a rate per $2 gross exposure. The D10-D1 spread carries $2 of gross exposure and zero target net dollar exposure. A one-way 50-bps fee costs `0.005 * (D10_turnover + D1_turnover)` per $1 long-short spread convention.

### Long leg / short leg / self-financing portfolio
The D10 long leg contributes `D10_gross - D10_cost`. The D1 short leg contributes `-D1_gross - D1_cost`. Thus the combined net strategy return is `D10_gross - D1_gross - D10_cost - D1_cost`. `D1_net` in the M10 monthly output means the **long-only** D1 return after costs, not the short-leg contribution; the latter has a separate explicitly named field.

### Gross return / net return
Gross D10-D1 retains the exact saved M8 arithmetic D10 minus D1 series; net subtracts costs on both legs. Annualized arithmetic means multiply monthly average by 12. Cumulative spread figures sum monthly returns, following M8, and are not compounded wealth indices.

### Break-even transaction cost
Descriptive one-way threshold equal to the mean positive gross spread divided by mean long-short one-sided traded notional; results are undefined when average gross is nonpositive or the denominator is invalid.

### Transaction-cost attrition
Difference between mean gross and mean net monthly spread; an optional attrition ratio divides by the absolute average gross spread only when it is meaningfully nonzero.

### Net HAC inference
The identical M9 six-lag Newey--West/HAC procedure is applied to the 50-bp net monthly spread: within-state means, LOW-reference dummy regression contrasts (MIDDLE minus LOW, HIGH minus LOW, HIGH minus MIDDLE), raw p-values, primary Holm correction by model/weighting family, and secondary Benjamini--Hochberg FDR p-values. Gross M9 inference is never overwritten.

### Implementation reference
`scripts/run_transaction_costs.py`, `src/rdsrp/transaction_costs/pipeline.py`, `results/transaction_costs/`, and generated `paper/generated/table_transaction_costs.tex`.

"""
    section_marker = "## Validated Turnover and Trading-Cost Methodology"
    raw = _insert_once(raw, "## Implementation Locations", section, section_marker)
    variable_rows = """| target_weight | M8 EW/VW target allocation | Frozen assignments and positive lagged ME | t | Turnover and cost |
| pretrade_weight | Last holdings' return-drifted allocation | M8 prior assignment and observed t return | End of t | Turnover and cost |
| absolute_weight_change | Absolute target-minus-pretrade change | Union of exiting and entering holdings | t | Audit only |
| long_short_total_turnover | D10 one-sided turnover plus D1 one-sided turnover | M8 assignment carry-forward | t | Net D10-D1 |
| long_short_cost | Cost rate times total one-sided notional | 0 / 50-bp assumption | t | Net D10-D1 |
| D10_minus_D1_net | Frozen gross spread less both leg costs | M8/M10 | t+1 | Net HAC inference |
"""
    if "| target_weight | M8 EW/VW target allocation |" not in raw:
        raw = raw.replace("\n\n## Model Hyperparameter Reference", "\n"+variable_rows+"\n## Model Hyperparameter Reference",1)
    change_row = "| Validated turnover and transaction-cost stage | M8 assignment-conserving EW/VW target and drifted pre-trade weights, 0/50-bps one-way costs, both long and short costs, traded notional, net M9 HAC inference, break-even costs, and cost attrition |\n"
    if "| Validated turnover and transaction-cost stage |" not in raw:
        raw = raw.replace("\n\n## Terminology Maintenance Requirement", "\n"+change_row+"\n## Terminology Maintenance Requirement",1)
    glossary.write_text(raw, encoding="utf-8")

    if not gitignore.is_file():
        raise FileNotFoundError(gitignore)
    ignored = gitignore.read_text(encoding="utf-8")
    holdings_line = "results/transaction_costs/portfolio_weights_monthly.csv"
    if holdings_line not in ignored:
        gitignore.write_text(ignored.rstrip() + "\n\n# Restricted row-level turnover holdings audit; never publish\n" + holdings_line + "\n", encoding="utf-8")

    r = readme.read_text(encoding="utf-8")
    extra = f"""{MARKER}

## Turnover and Transaction-Cost Evaluation

The frozen Milestone 8 LOW/MIDDLE/HIGH decile assignments now have a separate, validation-gated turnover and trading-cost analysis. The original forecasts, model tuning, deciles, M8 gross returns, and M9 gross HAC inference are unchanged. EW and VW turnovers are computed separately for D1 and D10 from prior return-drifted holdings, with costs on both long and short legs. The 0-bp case reproduces gross returns; the baseline robustness assumption is 50 bps one way, not a claim about observed execution costs.

Production **net** results, net HAC tests, regime contrasts and figures (generated only after protected-input validation) are in `results/transaction_costs/transaction_cost_summary.md` and `results/transaction_costs/`. For the current primary table see `paper/generated/table_transaction_costs.tex`. This exercise excludes borrow fees, funding costs and nonlinear market impact.

The next research stage is pre-specified **robustness analysis** of alternative volatility classifications, subsamples, eligibility restrictions, predictor treatments, and subperiod stability, without retroactively selecting specifications based on cost-adjusted results.

"""
    if MARKER not in r:
        r = r.replace("## Reproducibility", extra + "## Reproducibility", 1)
        readme.write_text(r, encoding="utf-8")

    p = manuscript.read_text(encoding="utf-8")
    p = p.replace("Deciles use tie-preserving percentile ranks.",
                  "The frozen three-state deciles use deterministic predicted-return order with PERMNO tie-breaking.")
    latex_marker = "% Validated transaction-cost analysis"
    new_latex = r"""
% Validated transaction-cost analysis
\subsection{Trading intensity and cost-adjusted three-state results}

To isolate implementation frictions without re-estimating forecasts, the turnover analysis
consumes the frozen month-by-month M8 assignments. The equal-weighted target is $1/N_{d,t}$;
value-weighted targets use the previously validated positive lagged market equity of stocks
in each decile. The current pre-trade portfolio weights are last month's targets drifted
through the return realized by the current formation date:
\[
 \widetilde{w}_{i,t}=\frac{w_{i,t-1}(1+R_{i,t})}{\sum_jw_{j,t-1}(1+R_{j,t})},
 \qquad
 T_{d,t}=\frac12\sum_i |w_{i,t}-\widetilde{w}_{i,t}|.
\]
The first formation from cash has $T_{d,t}=1$. The sum runs over the union of
old and new holdings, including full liquidations and new purchases. Returns
$R_{i,t}$ in the drift are the prior formation's observed holding-period returns;
formation-date forecasts do not depend on their own not-yet-realized $t+1$ returns.

The frozen convention is one dollar long in D10 and one dollar short in D1
(two dollars gross and zero dollars net exposure). With one-way cost $c$ the net spread is
\[
 R^{LS,\mathrm{net}}_{t+1}=R^{D10}_{t+1}-R^{D1}_{t+1}
 -c\left(T_{D10,t}+T_{D1,t}\right),\qquad c\in\{0,0.005\}.
\]
The 0-basis-point analysis must exactly reproduce the M8 gross series.
Both long and short legs incur rebalancing fees; the assumed 50-basis-point
rate is stylized and omits financing, borrow fees and nonlinear market impact.
All net-return HAC tests use the original six-lag M9 Newey--West covariance
and the same LOW-reference pairwise contrasts and Holm testing families.

\input{generated/table_transaction_costs}
"""
    if latex_marker not in p:
        p += "\n" + new_latex
        manuscript.write_text(p, encoding="utf-8")
