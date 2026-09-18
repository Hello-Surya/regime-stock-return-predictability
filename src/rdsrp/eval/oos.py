"""Strict expanding-window out-of-sample prediction engine."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from rdsrp.models.base import FitResult, ModelSpec


def run_expanding_oos(
    panel: pd.DataFrame,
    feature_cols: list[str],
    models: Mapping[str, ModelSpec],
    start_test: str | pd.Timestamp,
    end_test: str | pd.Timestamp | None = None,
    min_train_months: int = 12,
    retrain_every_months: int = 1,
) -> pd.DataFrame:
    if retrain_every_months < 1:
        raise ValueError("retrain_every_months must be >= 1.")
    required = {"date", "permno", "next_month_return", "regime", "vix", *feature_cols}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Modeling panel missing required columns: {sorted(missing)}")
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    if "formation_date" in df:
        df["formation_date"] = pd.to_datetime(df["formation_date"])
        mismatch = df["formation_date"].notna() & df["date"].notna() & df["formation_date"].ne(df["date"])
        if mismatch.any():
            raise AssertionError("date and formation_date disagree in the modeling panel.")
    else:
        df["formation_date"] = df["date"]
    if "realized_return_date" not in df.columns:
        df["realized_return_date"] = df["formation_date"] + pd.offsets.MonthEnd(1)
    else:
        df["realized_return_date"] = pd.to_datetime(df["realized_return_date"])
    feature_numeric = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    complete_features = feature_numeric.notna().all(axis=1) & np.isfinite(feature_numeric.to_numpy(float)).all(axis=1)
    df = df.loc[complete_features].copy()
    df.loc[:, feature_cols] = feature_numeric.loc[complete_features]
    df = df.sort_values(["formation_date", "permno"]).reset_index(drop=True)
    start = pd.Timestamp(start_test) + pd.offsets.MonthEnd(0)
    end = (pd.Timestamp(end_test) + pd.offsets.MonthEnd(0)) if end_test is not None else df["formation_date"].max()
    test_dates = [pd.Timestamp(d) for d in sorted(df["formation_date"].dropna().unique()) if start <= pd.Timestamp(d) <= end]
    outputs: list[pd.DataFrame] = []
    fitted: dict[str, tuple[FitResult, pd.Timestamp, pd.Timestamp]] = {}
    for test_index, test_date in enumerate(test_dates):
        observable_target = df["realized_return_date"] <= test_date
        train = df[(df["formation_date"] < test_date) & observable_target & df["next_month_return"].notna()].copy()
        test = df[(df["formation_date"] == test_date) & df["next_month_return"].notna()].copy()
        if test.empty or train["formation_date"].nunique() < min_train_months:
            continue
        current_train_formation_end = pd.Timestamp(train["formation_date"].max())
        current_train_realized_end = pd.Timestamp(train["realized_return_date"].max())
        if not current_train_formation_end < test_date:
            raise AssertionError("Leakage guard failed: training formation_date is not strictly before prediction formation_date.")
        if not current_train_realized_end <= test_date:
            raise AssertionError("Leakage guard failed: training realized_return_date is later than prediction formation_date.")
        stock_mean = train.groupby("permno")["next_month_return"].mean()
        global_mean = float(train["next_month_return"].mean())
        benchmark = test["permno"].map(stock_mean).fillna(global_mean).to_numpy(float)
        x_test = test[feature_cols].to_numpy(dtype=float)
        for name, model in models.items():
            should_refit = name not in fitted or test_index % retrain_every_months == 0
            if should_refit:
                x_train = train[feature_cols].to_numpy(dtype=float)
                y_train = train["next_month_return"].to_numpy(dtype=float)
                fit = model.fit(x_train, y_train)
                fitted[name] = (fit, current_train_formation_end, current_train_realized_end)
            fit, fit_formation_end, fit_realized_end = fitted[name]
            if not fit_formation_end < test_date:
                raise AssertionError("Stored model uses a formation date from the prediction month or later.")
            if not fit_realized_end <= test_date:
                raise AssertionError("Stored model uses a target not observable by prediction formation_date.")
            pred = model.predict(fit, x_test)
            carry = ["date", "formation_date", "permno", "regime", "vix", "next_month_return", "realized_return_date"]
            for optional in ("ticker", "me_lag", "me_lag_date", "market_equity"):
                if optional in test.columns:
                    carry.append(optional)
            out = test[list(dict.fromkeys(carry))].copy()
            out = out.rename(columns={"next_month_return": "actual"})
            out["actual_next_month_return"] = out["actual"]
            if "me_lag_date" in out.columns:
                out["value_weight_date"] = out["me_lag_date"]
            out["benchmark_prediction"] = benchmark
            out["prediction"] = pred
            out["model"] = name
            out["train_formation_end_date"] = fit_formation_end
            out["train_realized_return_end_date"] = fit_realized_end
            out["train_feature_end_date"] = fit_formation_end
            out["train_target_end_date"] = fit_realized_end
            out["train_end"] = fit_formation_end
            outputs.append(out)
    columns = [
        "date", "formation_date", "realized_return_date", "permno", "ticker", "regime", "vix",
        "actual", "actual_next_month_return", "benchmark_prediction", "prediction", "model",
        "me_lag", "me_lag_date", "value_weight_date", "train_formation_end_date",
        "train_realized_return_end_date", "train_feature_end_date", "train_target_end_date", "train_end",
    ]
    if not outputs:
        return pd.DataFrame(columns=columns)
    result = pd.concat(outputs, ignore_index=True)
    return result.sort_values(["formation_date", "model", "permno"]).reset_index(drop=True)
