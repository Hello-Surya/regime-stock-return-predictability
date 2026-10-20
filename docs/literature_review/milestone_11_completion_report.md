# Milestone 11 Completion and Verification Report

Prepared 2026-10-20. **Status: integrated documentation on isolated GitHub branch; scientific acceptance and local technical checks PENDING.** The established M1–M10 forecasting and portfolio work was not changed.

## A. Executive summary
M11 constructed a 38-study academic bibliography/literature matrix; a literature search protocol, direct-competitor comparison, research-gap analysis, five formal hypotheses, economic mechanisms, three contribution statements, journal-readiness report, evidence map, future milestone mapping, and an explicit retrospective research-transparency protocol. It integrated a new scholarly related-literature/hypotheses section into the pre-existing LaTeX manuscript. Citations have been linked consistently but a complete paper compile and core-paper full-text review remain outstanding.

## B. Repository status
- GitHub source: https://github.com/Hello-Surya/regime-stock-return-predictability
- Prior main head verified before M11: f89d8970ef794eba83cfca5bc2fef87c795320ba, dated 2026-10-20, M10 turnover/cost stage.
- M9 inference commit: 2021382f4c2dd12370f3025e8000860a9dca4fcb, dated 2026-09-29.
- M11 work committed to branch **milestone-11-literature-and-hypotheses**, originally branched from the above main head. No merge to main or force push occurred.
- Local uncommitted Windows changes and ignored licensed files cannot be inspected through the remote GitHub connection.

## C. Literature coverage
- **38** matrix studies, **27** columns, **37 published records** (including reviews and a corrigendum) and **1 non-refereed working paper**.
- **27 priority descriptions** in literature_matrix.md; 11 other documented records.
- **12 closest or adjacent competitors** in closest_competitors.md.
- Covered: characteristics, ML-based pricing, VIX/conditional volatility, cross-sectional versus aggregate targets, conditional model evaluation, market stress, trade friction, microcaps, HAC, and multiple-testing design.
- Bibliographic identity and abstract-level claims were checked against publisher and scholarly indexes, with inaccessible sample periods, predictor lists, and methods explicitly marked unverified. A full-text review of 15–20 priority articles and exhaustive later-version/priority check **remain incomplete**.

## D. Novelty finding
**Proceed, but narrow the contribution.** Gu–Kelly–Xiu (2020) already establishes nonlinear ML stock forecasting and Avramov–Cheng–Metzker (2023) already connects ML profits, microcaps, high-volatility states and trading costs. Chen–Pelger–Zhu (2024) and Fang–Ślepaczuk (2026 working paper) introduce further conditional ML overlap. The candidate narrower question is whether fixed identical-predictor forecasts show significant *direct* between-regime IC/loss and *net* portfolio contrasts under rigorous historical classification, additional benchmarks and investability restrictions. Current evidence does not yet demonstrate distinct originality.

## E. Research questions
Central: How does statistical cross-sectional forecast performance and net investment value vary by historically observed VIX regimes, and is XGBoost's *paired* relative performance against Elastic Net different across those states? Secondary: IC regime contrasts, model-to-model accuracy/rank differences, EW/VW net investment consequences. Exploratory: intermediate-state peak, economic channels, investability and stability.

## F. Hypothesis register
| ID | Status | Testable estimand and procedure | Related milestone |
|---|---|---|---|
| H1 regime heterogeneity | Retrospective on existing results | Direct equality of LOW/MIDDLE/HIGH monthly IC or return means, HAC joint tests and Holm contrasts | M9/M16 |
| H2 nonlinear relative advantage | New tests planned after observed baseline | Paired monthly XGB−EN IC and EN−XGB squared-loss differences by state | M12/M16 |
| H3 net economic significance | Retrospective for existing positive EW LOW signal | Net D10−D1 mean, appropriate uncertainty, separate direct state differences; trade costs and risk controls | M10/M15/M17 |
| H4 middle volatility peak | **Post-hoc exploratory** | MIDDLE−LOW > 0 AND MIDDLE−HIGH > 0 under joint/simultaneous inference | M14/M16 |
| H5 investability survival | Future plan after historical result inspection | Predeclared investability screens, net means, matched cost scenarios | M15/M16 |

These definitions postdate the M1–M10 results and must **never** be called historical preregistration.

## G. Files added or updated
- Under docs/literature_review/: research_status_at_m11.md, literature_search_protocol.md, literature_matrix.csv, literature_matrix.md, closest_competitors.md, research_gap_analysis.md, formal_hypotheses.md, economic_mechanisms.md, research_contribution.md, journal_readiness_assessment.md, hypothesis_milestone_mapping.md, claims_evidence_map.md, research_transparency_protocol.md, milestone_11_completion_report.md.
- Under paper/: new sections/01b_related_literature.tex; updated main.tex (one additional input) and populated existing references.bib (37 entries).
- Updated docs/research_glossary.md and README.md with scientific terminology, links and status qualifications.
- Added scripts/validate_literature_m11.py as a standard-library structural verification utility.

No protected row-level WRDS source data, credentials, private monthly holdings, local virtual environment or generated binaries were committed.

## H. Validation: actual performed checks versus pending tasks
**Actually performed in connected-repository static audit:**
- Parsed 38 CSV records with exactly 27 fields per record, with essential identity fields populated.
- Verified no duplicate CSV paper IDs, BibTeX keys or DOI identifiers.
- Found 38 unique bibliography entries, 33 citation-key occurrences and 31 distinct cited keys in the new section, with zero unresolved citations relative to the bibliography.
- Confirmed section included from paper/main.tex and simple new section/BibTeX brace counts balanced.
- Compared committed prior results with narrative; did not rewrite empirical values.

**NOT executed or verified from the remote read/write environment:**
- Python validator scripts/validate_literature_m11.py on a local checkout.
- Full repository pytest suite, strict terminology audit and git diff --check.
- Real manuscript compilation (pdflatex/bibtex cycles) or external cross-platform build.
- Independent WRDS row-level reproduction or scientific re-estimation.
- Every important paper's full text and the exact passage supporting each literature claim.

Passing static metadata checks is not scientific citation validation. **Do not mark full Milestone 11 acceptance or merge before the remaining review.**

### Local PowerShell commands (run from project root)
Preserve any unrelated uncommitted changes before switching branches. Then:
~~~powershell
git status --short
git fetch origin
git switch milestone-11-literature-and-hypotheses
git pull --ff-only origin milestone-11-literature-and-hypotheses

.\.venv\Scripts\python.exe scripts\validate_literature_m11.py
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\audit_terminology.py --strict
git diff --check
git status --short

# If a TeX distribution with pdflatex and bibtex is installed:
Push-Location paper
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
Pop-Location
~~~
These are unexecuted local validation commands, **not** a claim that the tests passed.

## I. Remaining scientific concerns
Novelty versus Avramov and newer work; strong simple benchmarks not yet compared; direct state tests imprecise after Holm; dependence and broad multiplicity; post-hoc selection of three states; EW/VW divergence; historical VIX proxy and threshold arbitrariness; incomplete borrow/funding/impact costs; microcap exposures; unstable subperiods; observational mechanism ambiguity; and limited licensed-data redistribution.

## J. What M12 should consume (without starting M12)
Use formal_hypotheses.md and hypothesis_milestone_mapping.md as the design record; retain frozen M7 forecasts and M8–M10 outputs. Add simple historical-mean, linear and characteristic-based benchmarks; prespecify calendar-month loss/IC paired-difference metrics and appropriate conditional HAC/model-comparison tests with sample and multiple-testing rules before new outcomes. Start M12 only after Milestone 11 acceptance criteria and author/advisor literature review are explicitly checked.
