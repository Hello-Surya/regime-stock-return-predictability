.PHONY: test synthetic

test:
	pytest

synthetic:
	python scripts/run_single_stock.py --synthetic --ticker STK001
