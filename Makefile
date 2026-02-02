.PHONY: help install lint format typecheck test run-pipeline build-paper

help:
    @echo "Targets: install lint format typecheck test run-pipeline build-paper"

install:
    python -m pip install -U pip
    python -m pip install -e ".[dev]"

lint:
    ruff check src scripts tests

format:
    ruff format src scripts tests

typecheck:
    mypy src

test:
    pytest

run-pipeline:
    python scripts/run_pipeline.py

build-paper:
    python scripts/build_paper_artifacts.py
    latexmk -pdf -interaction=nonstopmode -cd paper/main.tex
