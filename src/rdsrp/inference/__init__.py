"""Formal statistical inference for frozen monthly research outputs."""

from rdsrp.inference.hac import BASELINE_HAC_LAG, CONFIDENCE_LEVEL, hac_mean, significance_stars
from rdsrp.inference.pipeline import run_formal_inference

__all__ = ["BASELINE_HAC_LAG", "CONFIDENCE_LEVEL", "hac_mean", "significance_stars", "run_formal_inference"]
