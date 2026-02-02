def test_config_loading():
    from rdsrp.config import load_all
    from pathlib import Path
    # This test simply ensures configs load without errors.
    cfg = load_all(Path('configs'))
    assert cfg is not None
