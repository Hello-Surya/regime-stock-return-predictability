# Reproducibility

Use Python 3.11, fixed random seeds, YAML configuration, time-respecting tests, and generated artifacts. Local licensed data and credentials are gitignored. See RESEARCH_RUN.md for the full Windows/PowerShell execution guide.

## Research terminology maintenance

docs/research_glossary.md is the canonical Research Terminology and Methodology Appendix. It must remain synchronized with the code, configuration, README, validation artifacts, and manuscript.

Every future research stage has the following acceptance item:

- [ ] Research terminology appendix reviewed and updated

### Terminology Appendix Review

Before a research stage is considered complete:

1. identify newly introduced acronyms;
2. identify newly introduced variables and aliases;
3. identify newly introduced econometric or statistical terms and formulas;
4. identify newly introduced portfolio and performance terminology;
5. identify new datasets, identifiers, or source-table conventions;
6. update general definitions and project-specific implementation details;
7. update implementation-status labels, including planned or deprecated terms;
8. reconcile inconsistent language across README, documentation, code/configuration, results, and paper; and
9. confirm that the academic appendix and comprehensive Markdown reference do not conflict.

Run the lightweight terminology audit before completion:

~~~powershell
.\.venv\Scripts\python.exe scripts\audit_terminology.py --strict
~~~

The audit warns about potentially undocumented uppercase research acronyms on the main documentation, paper, configuration, and baseline-model-contract surfaces. It is intentionally conservative and does not modify files. A warning requires human review; it is not permission to add generic glossary filler or redefine stable code merely to satisfy the audit.

Implementation-specific glossary statements must be checked against current code and frozen configuration. Historical preliminary terminology should remain labeled historical rather than being silently rewritten as current production methodology.
