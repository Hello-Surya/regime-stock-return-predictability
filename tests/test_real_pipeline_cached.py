from pathlib import Path

import pandas as pd
import yaml

from rdsrp.data.synthetic import make_synthetic_market
from rdsrp.pipeline import run_wrds_single_stock


def test_cached_real_pipeline_generates_preliminary_artifacts(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "configs").mkdir()
    config = {
        "run": {
            "mode": "quick",
            "random_state": 5,
            "real": {
                "crsp_start": "2015-01-01",
                "crsp_end": "2020-12-31",
                "vix_history_start": "2014-01-01",
                "oos_start": "2019-01-31",
                "top_n_by_market_cap": 20,
                "universe_formation_date": "2015-01-31",
                "min_price": 5.0,
                "predictors": ["log_me", "mom_12_2"],
                "min_train_months": 24,
                "retrain_every_months": 2,
                "xgboost_n_jobs": 1,
                "tuning": {
                    "n_folds": 2,
                    "min_train_months": 24,
                    "elastic_net": {"alpha": [0.001], "l1_ratio": [0.5]},
                    "xgboost": {
                        "n_estimators": [5],
                        "max_depth": [2],
                        "learning_rate": [0.05],
                        "subsample": [0.8],
                        "colsample_bytree": [0.8],
                    },
                },
            },
        }
    }
    (tmp_path / "configs" / "quick.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")

    crsp, vix_monthly = make_synthetic_market(n_stocks=20, n_months=72, start="2015-01-31", random_state=5)
    crsp.loc[crsp["ticker"] == "STK001", "ticker"] = "AAPL"
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    crsp_cache = raw / "crsp_monthly_2015_2020_fixedtop20_p5.parquet"
    vix_cache = raw / "vix_daily_2014_2020.parquet"
    crsp_cache.touch()
    vix_cache.touch()

    def fake_read(path: Path) -> pd.DataFrame:
        return crsp.copy() if "crsp_monthly" in path.name else vix_monthly.copy()

    monkeypatch.setattr("rdsrp.pipeline.read_parquet_cache", fake_read)
    artifacts = run_wrds_single_stock(tmp_path, ticker="AAPL", quick=True)
    assert artifacts["predictions"].exists()
    assert artifacts["metrics"].exists()
    assert artifacts["summary"].exists()
    out = pd.read_csv(artifacts["predictions"])
    assert {"elastic_net_prediction", "xgboost_prediction", "historical_mean_benchmark"}.issubset(out.columns)
