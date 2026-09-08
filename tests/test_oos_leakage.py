from rdsrp.data.synthetic import make_synthetic_market
from rdsrp.eval.oos import run_expanding_oos
from rdsrp.features.build import build_features
from rdsrp.models.elastic_net import ElasticNetModel
from rdsrp.regimes.vix_regime import label_vix_regime


def test_oos_training_dates_are_strictly_historical() -> None:
    crsp, vix = make_synthetic_market(n_stocks=12, n_months=30, random_state=7)
    panel = build_features(crsp).merge(label_vix_regime(vix), on="date", how="left")
    predictions = run_expanding_oos(
        panel,
        feature_cols=["log_me", "book_to_market", "mom_12_2"],
        models={"elastic_net": ElasticNetModel(params={"alpha": 0.01})},
        start_test=panel["date"].sort_values().unique()[-4],
        min_train_months=12,
    )
    assert not predictions.empty
    assert (predictions["train_end"] < predictions["date"]).all()
