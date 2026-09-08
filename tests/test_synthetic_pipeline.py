from pathlib import Path

import pandas as pd

from rdsrp.pipeline import run_synthetic_validation


def test_synthetic_pipeline_generates_artifacts(tmp_path: Path) -> None:
    artifacts = run_synthetic_validation(
        tmp_path,
        ticker="STK001",
        n_stocks=15,
        n_months=32,
        oos_months=4,
        random_state=3,
    )
    assert artifacts["predictions"].exists()
    assert artifacts["metrics"].exists()
    assert artifacts["summary"].exists()
    predictions = pd.read_csv(artifacts["predictions"])
    assert {"actual_next_month_return", "elastic_net_prediction", "xgboost_prediction"}.issubset(predictions.columns)
