from pathlib import Path

from rdsrp.config import load_all


def test_config_loading() -> None:
    cfg = load_all(Path("configs"))
    assert cfg.universe["target"]["horizon_months"] == 1
    assert cfg.regime["regime"]["lookback"] == "expanding"
