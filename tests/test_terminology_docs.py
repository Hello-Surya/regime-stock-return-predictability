from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.audit_terminology import documented_acronyms

ROOT = Path(__file__).resolve().parents[1]
GLOSSARY = ROOT / "docs" / "research_glossary.md"


def test_glossary_contains_required_current_baseline_terms() -> None:
    text = GLOSSARY.read_text(encoding="utf-8")
    required = [
        "CRSP",
        "Compustat",
        "CCM",
        "PERMNO",
        "GVKEY",
        "Market Equity",
        "Book-to-Market",
        "mom_12_2",
        "VIX",
        "Elastic Net",
        "XGBoost",
        "OOS R²",
        "RMSE",
        "Rank IC",
    ]
    missing = [term for term in required if term not in text]
    assert not missing, f"Glossary is missing required terms: {missing}"


def test_acronym_registry_has_no_duplicate_keys() -> None:
    text = GLOSSARY.read_text(encoding="utf-8")
    registry = documented_acronyms(text)
    # The parser returns a set, so verify table keys separately before slash aliases
    # are expanded.
    section = text.split("## Acronym Index", 1)[1].split("\n## ", 1)[0]
    keys = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        key = line.split("|", 2)[1].strip().strip("`*")
        if key.lower() in {"acronym", "---"}:
            continue
        keys.append(key)
    assert len(keys) == len(set(keys))
    assert {"CRSP", "CCM", "OOS", "RMSE", "VIX"}.issubset(registry)


def test_readme_links_to_existing_glossary() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "docs/research_glossary.md" in readme
    assert GLOSSARY.exists()


def test_paper_includes_academic_terminology_appendix() -> None:
    main = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    appendix = ROOT / "paper" / "sections" / "appendix_glossary.tex"
    assert appendix.exists()
    assert "\\input{sections/appendix_glossary}" in main


def test_terminology_audit_script_executes() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "audit_terminology.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Research terminology audit" in result.stdout
