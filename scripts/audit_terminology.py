"""Warn about potentially undocumented research acronyms in repository-facing text.

This is deliberately a lightweight documentation audit, not a natural-language
parser. It scans high-value research documentation/configuration surfaces and
compares uppercase tokens with the acronym registry in docs/research_glossary.md.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GLOSSARY = REPO_ROOT / "docs" / "research_glossary.md"

# Infrastructure, file-format, source-code, and provider classification tokens
# that are meaningful in context but are not research acronyms requiring glossary
# entries. One-character CCM/CIZ classification codes are not captured by the
# token regex and therefore do not need to be listed here.
ALLOWLIST = {
    "AAPL",
    "ACOR",
    "API",
    "CLI",
    "COM",
    "CORP",
    "CPU",
    "CSV",
    "EQTY",
    "GB",
    "HTML",
    "HTTP",
    "HTTPS",
    "INDL",
    "JSON",
    "MB",
    "NA",
    "NAN",
    "NULL",
    "OVERALL",
    "PDF",
    "PNG",
    "README",
    "RW",
    "SQL",
    "STD",
    "TEX",
    "TIFF",
    "URL",
    "US",
    "USD",
    "UTF",
    "UTC",
    "YAML",
}

TOKEN_RE = re.compile(r"(?<![A-Za-z0-9_])([A-Z][A-Z0-9]{1,})(?![A-Za-z0-9_])")
TABLE_ROW_RE = re.compile(r"^\|\s*([^|]+?)\s*\|", re.MULTILINE)


def _acronym_section(text: str) -> str:
    start = text.find("## Acronym Index")
    if start < 0:
        raise ValueError("Glossary does not contain an '## Acronym Index' section.")
    tail = text[start:]
    next_heading = re.search(r"\n## (?!Acronym Index)", tail[1:])
    if next_heading:
        return tail[: next_heading.start() + 1]
    return tail


def documented_acronyms(glossary_text: str) -> set[str]:
    section = _acronym_section(glossary_text)
    documented: set[str] = set()
    for raw in TABLE_ROW_RE.findall(section):
        cell = raw.strip().strip("`*")
        if cell.lower() in {"acronym", "---"}:
            continue
        parts = re.split(r"\s+/\s+", cell) if re.search(r"\s/\s", cell) else [cell]
        for part in parts:
            token = part.strip().strip("`*")
            if token:
                documented.add(token.upper())
    # Common typographic/notation aliases for registry entries.
    if "B/M" in documented:
        documented.add("BM")
    if "R²" in documented:
        documented.add("R2")
    return documented


def scan_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    candidates = [root / "README.md", root / "RESEARCH_RUN.md"]
    candidates.extend(sorted((root / "docs").glob("*.md")))
    candidates.extend(sorted((root / "paper").rglob("*.tex")))
    candidates.extend(sorted((root / "configs").glob("*.yaml")))
    candidates.append(root / "src" / "rdsrp" / "model_spec.py")
    for path in candidates:
        if path.exists() and path != GLOSSARY:
            paths.append(path)
    return paths


def potential_undocumented(root: Path, documented: set[str]) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    known = documented | ALLOWLIST
    for path in scan_paths(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        tokens = set(TOKEN_RE.findall(text))
        # Explicitly capture two forms the simple token regex separates or misses.
        if "B/M" in text:
            tokens.add("B/M")
        if "R²" in text or "R^2" in text or "R-squared" in text or "R-squared" in text.lower():
            tokens.add("R²")
        unknown = {token for token in tokens if token.upper() not in known and token not in known}
        for token in unknown:
            found.setdefault(token, set()).add(str(path.relative_to(root)))
    return dict(sorted(found.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit status when potential undocumented acronyms are found.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not GLOSSARY.exists():
        print(f"Research terminology audit\n\nMissing glossary: {GLOSSARY.relative_to(REPO_ROOT)}")
        return 2

    glossary_text = GLOSSARY.read_text(encoding="utf-8")
    documented = documented_acronyms(glossary_text)
    unknown = potential_undocumented(REPO_ROOT, documented)

    print("Research terminology audit")
    print()
    print(f"Documented acronym/alias entries: {len(documented)}")
    print(f"Potential undocumented acronyms: {len(unknown)}")
    if unknown:
        print()
        for token, paths in unknown.items():
            locations = ", ".join(sorted(paths))
            print(f"{token}: {locations}")
        print()
        print("Review required before completing the research stage.")
    else:
        print("No potential undocumented research acronyms were found on the scanned surfaces.")

    return 1 if args.strict and unknown else 0


if __name__ == "__main__":
    raise SystemExit(main())
