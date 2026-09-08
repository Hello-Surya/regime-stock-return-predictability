"""Strict expanding-window out-of-sample prediction engine."""
from __future__ import annotations

from collections.abc import Mapping
import pandas as pd
from rdsrp.models.base import FitResult, ModelSpec


def run_expanding_oos(panel: pd.DataFrame, feature_cols: list[str], models: Mapping[str, ModelSpec], start_test: str | pd.Timestamp, end_test: str | pd.Timestamp | None = None, min_train_months: int = 12, retrain_every_months: int = 1) -> pd.DataFrame:
    if retrain_every_months < 1:
        raise ValueError("retrain_every_months must be >= 1.")
    required = {"date", "permno", "next_month_return", "regime", "vix", *feature_cols}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Modeling panel missing required columns: {sorted(missing)}")
    df = panel.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "permno"]).reset_index(drop=True)
    start = pd.Timestamp(start_test)
    end = pd.Timestamp(end_test) if end_test is not None else df["date"].max()
    test_dates = [pd.Timestamp(d) for d in sorted(df["date"].dropna().unique()) if start <= d <= end]
    outputs: list[pd.DataFrame] = []
    fitted: dict[str, tuple[FitResult, pd.Timestamp]] = {}
    for test_index, test_date in enumerate(test_dates):
        train = df[(df["date"] < test_date) & df["next_month_return"].notna()].copy()
        test = df[(df["date"] == test_date) & df["next_month_return"].notna()].copy()
        if test.empty or train["date"].nunique() < min_train_months:
            continue
        current_train_end = pd.Timestamp(train["date"].max())
        if not current_train_end < test_date:
            raise AssertionError("Leakage guard failed: training date is not strictly before test date.")
        stock_mean = train.groupby("permno")["next_month_return"].mean()
        global_mean = float(train["next_month_return"].mean())
        benchmark = test["permno"].map(stock_mean).fillna(global_mean).to_numpy(float)
        X_test = test[feature_cols].to_numpy(dtype=float)
        for name, model in models.items():
            should_refit = name not in fitted or test_index % retrain_every_months == 0
            if should_refit:
                X_train = train[feature_cols].to_numpy(dtype=float)
                y_train = train["next_month_return"].to_numpy(dtype=float)
                fit = model.fit(X_train, y_train)
                fitted[name] = (fit, current_train_end)
            fit, fit_train_end = fitted[name]
            if not fit_train_end < test_date:
                raise AssertionError("Stored model uses information from the prediction month or later.")
            pred = model.predict(fit, X_test)
            out = test[["date", "permno", "regime", "vix", "next_month_return"]].copy()
            if "ticker" in test.columns:
                out["ticker"] = test["ticker"].to_numpy()
            out = out.rename(columns={"next_month_return": "actual"})
            out["benchmark_prediction"] = benchmark
            out["prediction"] = pred
            out["model"] = name
            out["train_end"] = fit_train_end
            outputs.append(out)
    columns = ["date", "permno", "ticker", "regime", "vix", "actual", "benchmark_prediction", "prediction", "model", "train_end"]
    if not outputs:
        return pd.DataFrame(columns=columns)
    result = pd.concat(outputs, ignore_index=True)
    return result.sort_values(["date", "model", "permno"]).reset_index(drop=True)
