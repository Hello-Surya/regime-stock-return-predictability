"""Historical tuning and expanding-window production prediction."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rdsrp.models.base import FitResult, ModelSpec
from rdsrp.models.elastic_net import ElasticNetModel
from rdsrp.models.tuning import select_params_time_cv
from rdsrp.models.xgboost_model import XGBoostModel

from .data import BASELINE_PREDICTORS


def deterministic_tuning_sample(
    frame: pd.DataFrame, *, max_rows_per_month: int | None, random_state: int
) -> pd.DataFrame:
    """Deterministic historical-only cross-sectional sample used only for tuning."""
    if not max_rows_per_month or max_rows_per_month <= 0:
        return frame.sort_values(["date", "permno"], kind="mergesort").reset_index(drop=True)
    pieces: list[pd.DataFrame] = []
    for date, group in frame.groupby("date", sort=True):
        if len(group) <= max_rows_per_month:
            pieces.append(group)
            continue
        date_seed = int(pd.Timestamp(date).strftime("%Y%m")) + int(random_state) * 1_000_003
        pieces.append(group.sample(n=max_rows_per_month, random_state=date_seed, replace=False))
    return (
        pd.concat(pieces, ignore_index=True)
        .sort_values(["date", "permno"], kind="mergesort")
        .reset_index(drop=True)
    )


def tune_baseline_models(
    initial_train: pd.DataFrame, real_cfg: Mapping[str, Any], *, random_state: int
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    if list(real_cfg.get("predictors", [])) != list(BASELINE_PREDICTORS):
        raise ValueError(f"Production predictors must be exactly {list(BASELINE_PREDICTORS)}")
    tuning_cfg = real_cfg["tuning"]
    cap = int(tuning_cfg.get("max_rows_per_month", 500))
    tune_frame = deterministic_tuning_sample(
        initial_train, max_rows_per_month=cap, random_state=random_state
    )
    grids = {
        "elastic_net": {**dict(tuning_cfg["elastic_net"]), "max_iter": [20_000]},
        "xgboost": {
            **dict(tuning_cfg["xgboost"]),
            "n_jobs": [int(real_cfg.get("xgboost_n_jobs", 4))],
            "verbosity": [0],
            "tree_method": ["hist"],
        },
    }
    selected: dict[str, dict[str, Any]] = {}
    tables: list[pd.DataFrame] = []
    for model_name in ("elastic_net", "xgboost"):
        best, table = select_params_time_cv(
            tune_frame,
            list(BASELINE_PREDICTORS),
            model_name,
            grids[model_name],
            n_splits=int(tuning_cfg["n_folds"]),
            min_train_months=int(tuning_cfg["min_train_months"]),
            random_state=random_state,
        )
        selected[model_name] = best
        table = table.copy()
        table["selected"] = table["params"].map(lambda p: p == best)
        table["tuning_start"] = str(pd.to_datetime(tune_frame["date"]).min().date())
        table["tuning_end"] = str(pd.to_datetime(tune_frame["date"]).max().date())
        table["tuning_rows"] = int(len(tune_frame))
        table["max_rows_per_month"] = cap
        tables.append(table)
    result = pd.concat(tables, ignore_index=True)
    result["params"] = result["params"].map(lambda p: json.dumps(p, sort_keys=True))
    return selected, result


def build_models(
    selected: Mapping[str, Mapping[str, Any]], *, random_state: int
) -> dict[str, ModelSpec]:
    return {
        "elastic_net": ElasticNetModel(dict(selected["elastic_net"]), random_state=random_state),
        "xgboost": XGBoostModel(dict(selected["xgboost"]), random_state=random_state),
    }


def _interpretability(
    model_name: str, fit: FitResult, refit_date: pd.Timestamp
) -> list[dict[str, Any]]:
    estimator = fit.estimator
    if not hasattr(estimator, "named_steps") or "model" not in estimator.named_steps:
        return []
    rows: list[dict[str, Any]] = []
    model = estimator.named_steps["model"]
    if model_name == "elastic_net":
        values = np.asarray(model.coef_, dtype=float).reshape(-1)
        value_name = "standardized_coefficient"
    else:
        values = getattr(model, "feature_importances_", None)
        if values is None:
            return []
        values = np.asarray(values, dtype=float).reshape(-1)
        value_name = "feature_importance"
    for feature, value in zip(BASELINE_PREDICTORS, values, strict=True):
        rows.append(
            {
                "refit_date": refit_date,
                "model": model_name,
                "predictor": feature,
                value_name: float(value),
            }
        )
    return rows


def validate_prediction_panel(predictions: pd.DataFrame) -> None:
    if predictions.empty:
        raise ValueError("No production OOS predictions were generated.")
    if predictions.duplicated(["permno", "date"]).any():
        raise AssertionError("Duplicate production predictions exist for the same permno/date.")
    if not (
        pd.to_datetime(predictions["train_feature_end_date"]) < pd.to_datetime(predictions["date"])
    ).all():
        raise AssertionError("Training feature date is not strictly before formation.")
    if not (
        pd.to_datetime(predictions["train_target_end_date"]) <= pd.to_datetime(predictions["date"])
    ).all():
        raise AssertionError("A training target was not observable by formation.")
    if not (
        pd.to_datetime(predictions["realized_return_date"]) > pd.to_datetime(predictions["date"])
    ).all():
        raise AssertionError("Realized next-month return date must follow formation date.")
    numeric = predictions[["elastic_net_prediction", "xgboost_prediction"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        raise AssertionError("NaN or infinite production predictions detected.")


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
        raise ValueError("retrain_every_months must be at least 1.")
    df = sample.sort_values(["date", "permno"], kind="mergesort").reset_index(drop=True)
    if df[list(BASELINE_PREDICTORS)].isna().any().any():
        raise AssertionError("Baseline estimation is complete-case; missing predictors are not allowed.")
    dates = pd.to_datetime(df["date"])
    realized = pd.to_datetime(df["realized_return_date"])
    start = pd.Timestamp(start_test) + pd.offsets.MonthEnd(0)
    end = (pd.Timestamp(end_test) + pd.offsets.MonthEnd(0)) if end_test is not None else dates.max()
    oos_dates = [pd.Timestamp(d) for d in sorted(dates.unique()) if start <= pd.Timestamp(d) <= end]
    if max_oos_months is not None:
        oos_dates = oos_dates[: int(max_oos_months)]

    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        manifest = checkpoint_dir / "manifest.json"
        if manifest.exists() and resume:
            old = json.loads(manifest.read_text(encoding="utf-8"))
            if old.get("run_fingerprint") != run_fingerprint:
                raise RuntimeError(
                    "Existing checkpoints use a different data/configuration fingerprint; use --refresh."
                )
        elif run_fingerprint is not None:
            manifest.write_text(
                json.dumps({"run_fingerprint": run_fingerprint}, indent=2), encoding="utf-8"
            )

    x = df.loc[:, list(BASELINE_PREDICTORS)].to_numpy(dtype=np.float32, copy=False)
    y = df["next_month_return"].to_numpy(dtype=np.float32, copy=False)
    date_values = dates.to_numpy(dtype="datetime64[ns]")
    unique_train_dates = np.array(sorted(dates.unique()), dtype="datetime64[ns]")
    fitted: dict[str, tuple[FitResult, pd.Timestamp, pd.Timestamp]] = {}
    outputs: list[pd.DataFrame] = []
    elastic_rows: list[dict[str, Any]] = []
    xgb_rows: list[dict[str, Any]] = []

    for test_index, test_date in enumerate(oos_dates):
        checkpoint = (
            checkpoint_dir / f"predictions_{test_date:%Y_%m}.parquet" if checkpoint_dir else None
        )
        coef_file = (
            checkpoint_dir / f"elastic_net_coefficients_{test_date:%Y_%m}.csv"
            if checkpoint_dir
            else None
        )
        importance_file = (
            checkpoint_dir / f"xgboost_feature_importance_{test_date:%Y_%m}.csv"
            if checkpoint_dir
            else None
        )
        if checkpoint is not None and checkpoint.exists() and resume:
            outputs.append(pd.read_parquet(checkpoint))
            if coef_file is not None and coef_file.exists():
                elastic_rows.extend(pd.read_csv(coef_file).to_dict("records"))
            if importance_file is not None and importance_file.exists():
                xgb_rows.extend(pd.read_csv(importance_file).to_dict("records"))
            continue

        test_np = np.datetime64(test_date.to_datetime64())
        lo = int(np.searchsorted(date_values, test_np, side="left"))
        hi = int(np.searchsorted(date_values, test_np, side="right"))
        train_months = int(np.searchsorted(unique_train_dates, test_np, side="left"))
        if lo == hi or lo <= 0 or train_months < min_train_months:
            continue
        train_feature_end = pd.Timestamp(date_values[lo - 1])
        train_target_end = pd.Timestamp(realized.iloc[:lo].max())
        if not train_feature_end < test_date:
            raise AssertionError("Leakage guard failed: training feature date is not before formation.")
        if not train_target_end <= test_date:
            raise AssertionError("Leakage guard failed: training target is not observable by formation.")

        x_train, y_train, x_test = x[:lo], y[:lo], x[lo:hi]
        benchmark = float(np.mean(y_train, dtype=np.float64))
        predictions: dict[str, np.ndarray] = {}
        for model_name in ("elastic_net", "xgboost"):
            model = models[model_name]
            should_refit = model_name not in fitted or test_index % retrain_every_months == 0
            if should_refit:
                fit = model.fit(x_train, y_train)
                fitted[model_name] = (fit, train_feature_end, train_target_end)
                rows = _interpretability(model_name, fit, test_date)
                (elastic_rows if model_name == "elastic_net" else xgb_rows).extend(rows)
            fit, fit_feature_end, fit_target_end = fitted[model_name]
            if not fit_feature_end < test_date or not fit_target_end <= test_date:
                raise AssertionError("Stored fitted model violates production timing constraints.")
            predictions[model_name] = model.predict(fit, x_test)

        carry = [
            "date",
            "permno",
            "regime",
            "vix",
            *BASELINE_PREDICTORS,
            "next_month_return",
            "realized_return_date",
        ]
        for optional in ("ticker", "me_lag", "me_lag_date", "market_equity"):
            if optional in df.columns:
                carry.append(optional)
        out = df.iloc[lo:hi][carry].copy().rename(
            columns={"next_month_return": "actual_next_month_return"}
        )
        out["benchmark_prediction"] = benchmark
        out["elastic_net_prediction"] = predictions["elastic_net"]
        out["xgboost_prediction"] = predictions["xgboost"]
        out["train_feature_end_date"] = train_feature_end
        out["train_target_end_date"] = train_target_end
        if out.duplicated(["permno", "date"]).any():
            raise AssertionError("Duplicate predictions created within an OOS month.")
        outputs.append(out)
        if checkpoint is not None:
            out.to_parquet(checkpoint, index=False)
            month_coef = [r for r in elastic_rows if pd.Timestamp(r["refit_date"]) == test_date]
            month_imp = [r for r in xgb_rows if pd.Timestamp(r["refit_date"]) == test_date]
            if coef_file is not None and month_coef:
                pd.DataFrame(month_coef).to_csv(coef_file, index=False)
            if importance_file is not None and month_imp:
                pd.DataFrame(month_imp).to_csv(importance_file, index=False)

    if not outputs:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    result = pd.concat(outputs, ignore_index=True).sort_values(["date", "permno"]).reset_index(drop=True)
    validate_prediction_panel(result)
    return result, pd.DataFrame(elastic_rows), pd.DataFrame(xgb_rows)
