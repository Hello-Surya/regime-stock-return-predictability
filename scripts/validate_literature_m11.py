r"""Static metadata and citation audit for Milestone 11.

This script verifies *file consistency*, not the validity of academic conclusions,
the accuracy of published studies, or independent replication of local WRDS results.
Run from anywhere with Python 3.11+:
    .\.venv\Scripts\python.exe scripts\validate_literature_m11.py
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs" / "literature_review"
CSV_PATH = BASE / "literature_matrix.csv"
BIB_PATH = ROOT / "paper" / "references.bib"
MAIN = ROOT / "paper" / "main.tex"
SECTION = ROOT / "paper" / "sections" / "01b_related_literature.tex"

REQUIRED = (
    "Paper ID", "Authors", "Year", "Title", "Journal / Working Paper",
    "Publication Status", "DOI / Verified URL", "Literature Category",
    "Research Question", "Dataset", "Sample Period", "Data Frequency",
    "Prediction Target", "Forecast Horizon", "Predictors", "Models",
    "Market-Regime Definition", "Out-of-Sample Methodology",
    "Statistical Evaluation", "Economic Evaluation", "Principal Findings",
    "Limitations", "Relationship to Our Research",
    "Overlap With Our Contribution", "Potential Novelty Implications",
    "Evidence / Access Status", "BibTeX Citation Key",
)
ESSENTIAL = (
    "Paper ID", "Authors", "Year", "Title", "Journal / Working Paper",
    "Publication Status", "DOI / Verified URL", "Literature Category",
    "Evidence / Access Status", "BibTeX Citation Key",
)
BIB_KEY = re.compile(r"@\w+\s*{\s*([^,\s]+)\s*,")
DOI = re.compile(r"\bdoi\s*=\s*{\s*([^}]+)\s*}", flags=re.IGNORECASE)
CITATION = re.compile(r"\\cite(?:\[[^\]]*\])?\{([^}]+)\}")


def duplicates(values: list[str]) -> list[str]:
    return sorted(value for value, count in Counter(values).items() if count > 1)


def main() -> int:
    failures: list[str] = []
    try:
        with CSV_PATH.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            rows = list(reader)
    except OSError as exc:
        print(f"Unable to load literature matrix: {exc}", file=sys.stderr)
        return 1

    missing_columns = sorted(set(REQUIRED) - set(fields))
    if missing_columns:
        failures.append(f"Missing required columns: {missing_columns}")
    if len(rows) < 25:
        failures.append(f"Only {len(rows)} papers, expected at least 25")
    if any(None in row for row in rows):
        failures.append("Malformed CSV record with excess columns")
    for idx, row in enumerate(rows, start=2):
        for field in ESSENTIAL:
            if not str(row.get(field) or "").strip():
                failures.append(f"Row {idx} missing essential field: {field}")
        value = str(row.get("Year") or "")
        if value and not re.fullmatch(r"(?:19|20)\d{2}", value):
            failures.append(f"Row {idx} unexpected publication year: {value}")

    paper_ids = [str(row.get("Paper ID") or "").strip() for row in rows]
    csv_keys = [str(row.get("BibTeX Citation Key") or "").strip() for row in rows]
    if repeated := duplicates(paper_ids):
        failures.append(f"Duplicate paper IDs: {repeated}")
    if repeated := duplicates(csv_keys):
        failures.append(f"Duplicate CSV BibTeX keys: {repeated}")

    dois = []
    for row in rows:
        identifier = str(row.get("DOI / Verified URL") or "").strip()
        if identifier and not identifier.startswith(("https://", "http://")):
            dois.append(identifier.lower().replace("https://doi.org/", ""))
    if repeated := duplicates(dois):
        failures.append(f"Duplicate DOI identifiers in CSV: {repeated}")

    try:
        bib_text = BIB_PATH.read_text(encoding="utf-8")
        tex_text = SECTION.read_text(encoding="utf-8")
        main_text = MAIN.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Unable to load manuscript or bibliography: {exc}", file=sys.stderr)
        return 1

    bib_keys = BIB_KEY.findall(bib_text)
    if repeated := duplicates(bib_keys):
        failures.append(f"Duplicate bibliography keys: {repeated}")
    if (unknown := sorted(set(csv_keys) - set(bib_keys))):
        failures.append(f"Matrix keys missing from bibliography: {unknown}")
    if (unlisted := sorted(set(bib_keys) - set(csv_keys))):
        failures.append(f"Bibliography keys absent from matrix: {unlisted}")
    if repeated := duplicates([x.strip().lower() for x in DOI.findall(bib_text)]):
        failures.append(f"Duplicate bibliography DOIs: {repeated}")

    citations = [
        key.strip()
        for texfile in (ROOT / "paper" / "sections").glob("*.tex")
        for m in CITATION.findall(texfile.read_text(encoding="utf-8"))
        for key in m.split(",")
    ]
    if unknown := sorted(set(citations) - set(bib_keys)):
        failures.append(f"Unresolved citation keys: {unknown}")
    if r"\input{sections/01b_related_literature}" not in main_text:
        failures.append("The new related literature section is not included in paper/main.tex")
    if r"\operatorname" in tex_text:
        failures.append("New section uses amsmath-specific \\operatorname without known package support")

    print(
        "Milestone 11 static audit: "
        f"{len(rows)} matrix records, {len(fields)} columns, "
        f"{len(bib_keys)} bibliography entries, {len(citations)} manuscript citations."
    )
    if failures:
        print("FAILED:")
        for problem in failures:
            print(f"- {problem}")
        return 1
    print("PASS: metadata structure, duplicate identifiers and internal citations.")
    print("NOTE: this is NOT full-text scientific verification, pytest or LaTeX compilation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
