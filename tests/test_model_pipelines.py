import numpy as np

from rdsrp.models.elastic_net import ElasticNetModel
from rdsrp.models.xgboost_model import XGBoostModel


def test_models_fit_and_predict() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 3))
    y = 0.2 * X[:, 0] - 0.1 * X[:, 1] + rng.normal(scale=0.1, size=80)
    for model in (
        ElasticNetModel(params={"alpha": 0.001}),
        XGBoostModel(params={"n_estimators": 10, "n_jobs": 1}),
    ):
        fit = model.fit(X, y)
        pred = model.predict(fit, X[:5])
        assert pred.shape == (5,)
        assert np.isfinite(pred).all()
