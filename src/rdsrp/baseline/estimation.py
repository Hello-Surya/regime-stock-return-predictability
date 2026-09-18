"""Historical tuning and leakage-safe expanding-window baseline prediction."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rdsrp.model_spec import FORMATION_DATE, REALIZED_RETURN_DATE
from rdsrp.models.base import FitResult, ModelSpec
from rdsrp.models.elastic_net import ElasticNetModel
from rdsrp.models.tuning import select_params_time_cv
from rdsrp.models.xgboost_model import XGBoostModel

from .data import BASELINE_PREDICTORS


def deterministic_tuning_sample(
    frame: pd.DataFrame, *, max_rows_per_month: int | None, random_state: int
) -> pd.DataFrame:
    if not max_rows_per_month or max_rows_per_month <= 0:
        return frame.sort_values([FORMATION_DATE, "permno"]).reset_index(drop=True)
    pieces = []
    for date, group in frame.groupby(FORMATION_DATE, sort=True):
        if len(group) > max_rows_per_month:
            seed = int(pd.Timestamp(date).strftime("%Y%m")) + int(random_state) * 1_000_003
            group = group.sample(n=max_rows_per_month, random_state=seed, replace=False)
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True).sort_values([FORMATION_DATE, "permno"]).reset_index(drop=True)


def tune_baseline_models(
    initial_train: pd.DataFrame, real_cfg: Mapping[str, Any], *, random_state: int
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    if list(real_cfg.get("predictors", [])) != list(BASELINE_PREDICTORS):
        raise ValueError(f"Baseline predictors must be exactly {list(BASELINE_PREDICTORS)}")
    cfg = real_cfg["tuning"]
    cap = int(cfg.get("max_rows_per_month", 500))
    frame = deterministic_tuning_sample(initial_train, max_rows_per_month=cap, random_state=random_state)
    grids = {
        "elastic_net": {**dict(cfg["elastic_net"]), "max_iter": [20_000]},
        "xgboost": {
            **dict(cfg["xgboost"]),
            "n_jobs": [int(real_cfg.get("xgboost_n_jobs", 4))],
            "verbosity": [0],
            "tree_method": ["hist"],
        },
    }
    selected: dict[str, dict[str, Any]] = {}
    tables = []
    for name in ("elastic_net", "xgboost"):
        best, table = select_params_time_cv(
            frame,
            list(BASELINE_PREDICTORS),
            name,
            grids[name],
            date_col=FORMATION_DATE,
            realized_return_date_col=REALIZED_RETURN_DATE,
            n_splits=int(cfg["n_folds"]),
            min_train_months=int(cfg["min_train_months"]),
            random_state=random_state,
        )
        selected[name] = best
        table = table.copy()
        table["selected"] = table["params"].map(lambda p: p == best)
        table["tuning_start"] = str(pd.to_datetime(frame[FORMATION_DATE]).min().date())
        table["tuning_end"] = str(pd.to_datetime(frame[FORMATION_DATE]).max().date())
        table["tuning_rows"] = len(frame)
        tables.append(table)
    result = pd.concat(tables, ignore_index=True)
    result["params"] = result["params"].map(lambda p: json.dumps(p, sort_keys=True))
    return selected, result


def build_models(selected: Mapping[str, Mapping[str, Any]], *, random_state: int) -> dict[str, ModelSpec]:
    return {
        "elastic_net": ElasticNetModel(dict(selected["elastic_net"]), random_state=random_state),
        "xgboost": XGBoostModel(dict(selected["xgboost"]), random_state=random_state),
    }


def _interpretability(name: str, fit: FitResult, refit_date: pd.Timestamp) -> list[dict[str, Any]]:
    estimator = fit.estimator
    if not hasattr(estimator, "named_steps") or "model" not in estimator.named_steps:
        return []
    model = estimator.named_steps["model"]
    if name == "elastic_net":
        values = np.asarray(model.coef_, dtype=float).reshape(-1)
        field = "standardized_coefficient"
    else:
        values = getattr(model, "feature_importances_", None)
        if values is None:
            return []
        values = np.asarray(values, dtype=float).reshape(-1)
        field = "feature_importance"
    return [
        {"refit_date": refit_date, "model": name, "predictor": feature, field: float(value)}
        for feature, value in zip(BASELINE_PREDICTORS, values, strict=True)
    ]


def validate_prediction_panel(predictions: pd.DataFrame) -> None:
    if predictions.empty:
        raise ValueError("No baseline OOS predictions were generated")
    if predictions.duplicated(["permno", FORMATION_DATE]).any():
        raise AssertionError("Duplicate predictions exist for the same permno/formation_date")
    formation = pd.to_datetime(predictions[FORMATION_DATE])
    train_formation = pd.to_datetime(predictions["train_formation_end_date"])
    train_realized = pd.to_datetime(predictions["train_realized_return_end_date"])
    if not (train_formation < formation).all():
        raise AssertionError("training formation_date must be strictly before prediction formation_date")
    if not (train_realized <= formation).all():
        raise AssertionError("training realized_return_date must be on or before prediction formation_date")
    if not (pd.to_datetime(predictions[REALIZED_RETURN_DATE]) > formation).all():
        raise AssertionError("realized next-month return must follow formation_date")
    values = predictions[["elastic_net_prediction", "xgboost_prediction"]].to_numpy(float)
    if not np.isfinite(values).all():
        raise AssertionError("NaN or infinite baseline predictions detected")


def run_expanding_baseline_oos(
    sample: pd.DataFrame,
    models: Mapping[str, ModelSpec],
    *,
    start_test: str | pd.Timestamp,
    end_test: str | pd.Timestamp | None,
    min_train_months: int,
    retrain_every_months: int,
    checkpoint_dir: Path | None = None,
    run_fingerprint: str | None = None,
    resume: bool = True,
    max_oos_months: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if retrain_every_months < 1:
        raise ValueError("retrain_every_months must be at least 1")
    if set(models) != {"elastic_net", "xgboost"}:
        raise ValueError("Baseline estimation requires exactly Elastic Net and XGBoost")
    df = sample.sort_values([FORMATION_DATE, "permno"]).reset_index(drop=True)
    matrix = df[list(BASELINE_PREDICTORS)].to_numpy(float)
    if not np.isfinite(matrix).all():
        raise AssertionError("Baseline estimation is complete-case; missing predictors are not allowed")
    start = pd.Timestamp(start_test) + pd.offsets.MonthEnd(0)
    end = pd.Timestamp(end_test) + pd.offsets.MonthEnd(0) if end_test is not None else df[FORMATION_DATE].max()
    oos_dates = [pd.Timestamp(d) for d in sorted(df[FORMATION_DATE].unique()) if start <= pd.Timestamp(d) <= end]
    if max_oos_months is not None:
        oos_dates = oos_dates[: int(max_oos_months)]
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        manifest = checkpoint_dir / "manifest.json"
        if manifest.exists() and resume:
            old = json.loads(manifest.read_text(encoding="utf-8"))
            if old.get("run_fingerprint") != run_fingerprint:
                raise RuntimeError("Checkpoint fingerprint mismatch; use --refresh")
        elif run_fingerprint is not None:
            manifest.write_text(json.dumps({"run_fingerprint": run_fingerprint}, indent=2), encoding="utf-8")

    outputs = []
    elastic_rows: list[dict[str, Any]] = []
    xgb_rows: list[dict[str, Any]] = []
    fitted: dict[str, tuple[FitResult, pd.Timestamp, pd.Timestamp]] = {}

    for test_index, test_date in enumerate(oos_dates):
        checkpoint = checkpoint_dir / f"predictions_{test_date:%Y_%m}.parquet" if checkpoint_dir else None
        if checkpoint is not None and checkpoint.exists() and resume:
            cached = pd.read_parquet(checkpoint)
            validate_prediction_panel(cached)
            outputs.append(cached)
            continue

        train = df[
            (df[FORMATION_DATE] < test_date)
            & (df[REALIZED_RETURN_DATE] <= test_date)
        ]
        test = df[df[FORMATION_DATE] == test_date]
        if test.empty or train[FORMATION_DATE].nunique() < min_train_months:
            continue
        train_formation_end = pd.Timestamp(train[FORMATION_DATE].max())
        train_realized_end = pd.Timestamp(train[REALIZED_RETURN_DATE].max())
        if not train_formation_end < test_date:
            raise AssertionError("Leakage guard failed for training formation_date")
        if not train_realized_end <= test_date:
            raise AssertionError("Leakage guard failed for training realized_return_date")

        x_train = train[list(BASELINE_PREDICTORS)].to_numpy(np.float32)
        y_train = train["next_month_return"].to_numpy(np.float32)
        x_test = test[list(BASELINE_PREDICTORS)].to_numpy(np.float32)
        predictions: dict[str, np.ndarray] = {}
        for name in ("elastic_net", "xgboost"):
            model = models[name]
            if name not in fitted or test_index % retrain_every_months == 0:
                fit = model.fit(x_train, y_train)
                fitted[name] = (fit, train_formation_end, train_realized_end)
                rows = _interpretability(name, fit, test_date)
                (elastic_rows if name == "elastic_net" else xgb_rows).extend(rows)
            fit, fit_formation_end, fit_realized_end = fitted[name]
            if not fit_formation_end < test_date or not fit_realized_end <= test_date:
                raise AssertionError("Stored fitted model violates timing constraints")
            predictions[name] = model.predict(fit, x_test)

        carry = [
            "date", FORMATION_DATE, REALIZED_RETURN_DATE, "permno", "ticker", "regime", "vix",
            *BASELINE_PREDICTORS, "next_month_return", "me_lag", "me_lag_date", "market_equity",
        ]
        out = test[[c for c in dict.fromkeys(carry) if c in test]].copy()
        out = out.rename(columns={"next_month_return": "actual_next_month_return"})
        out["benchmark_prediction"] = float(np.mean(y_train, dtype=np.float64))
        out["elastic_net_prediction"] = predictions["elastic_net"]
        out["xgboost_prediction"] = predictions["xgboost"]
        out["train_formation_end_date"] = train_formation_end
        out["train_realized_return_end_date"] = train_realized_end
        out["train_feature_end_date"] = train_formation_end
        out["train_target_end_date"] = train_realized_end
        outputs.append(out)
        if checkpoint is not None:
            out.to_parquet(checkpoint, index=False)

    if not outputs:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    result = pd.concat(outputs, ignore_index=True).sort_values([FORMATION_DATE, "permno"]).reset_index(drop=True)
    validate_prediction_panel(result)
    return result, pd.DataFrame(elastic_rows), pd.DataFrame(xgb_rows)
