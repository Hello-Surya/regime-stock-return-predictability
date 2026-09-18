"""Finalization and validation helpers for the canonical baseline modeling panel."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from rdsrp.model_spec import (
    BASELINE_FEATURES,
    BASELINE_TARGET,
    FORMATION_DATE,
    REALIZED_RETURN_DATE,
)

QUANTILES = (
    (0.001, "p0.1"),
    (0.01, "p1"),
    (0.05, "p5"),
    (0.25, "p25"),
    (0.50, "median"),
    (0.75, "p75"),
    (0.95, "p95"),
    (0.99, "p99"),
    (0.999, "p99.9"),
)
EXTREME_VARIABLES = ("book_to_market", "mom_12_2", "me", BASELINE_TARGET)


def _month_end(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce") + pd.offsets.MonthEnd(0)


def _num(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def finalize_baseline_modeling_panel(
    panel: pd.DataFrame,
    *,
    min_price: float = 5.0,
) -> pd.DataFrame:
    """Add explicit date semantics, lineage aliases, and auditable sample flags."""
    out = panel.copy()
    core = {"permno", BASELINE_TARGET, *BASELINE_FEATURES}
    missing = sorted(core.difference(out.columns))
    if missing:
        raise ValueError(f"Modeling panel is missing core model fields: {missing}")
    if "date" not in out and FORMATION_DATE not in out:
        raise ValueError("Modeling panel must contain date or formation_date.")

    if FORMATION_DATE not in out:
        out[FORMATION_DATE] = _month_end(out["date"])
    else:
        out[FORMATION_DATE] = _month_end(out[FORMATION_DATE])
    if "date" not in out:
        out["date"] = out[FORMATION_DATE]
    else:
        out["date"] = _month_end(out["date"])
        mismatch = out["date"].notna() & out[FORMATION_DATE].notna() & out["date"].ne(out[FORMATION_DATE])
        if mismatch.any():
            raise AssertionError("date and formation_date disagree in the modeling panel.")

    if REALIZED_RETURN_DATE not in out:
        raise ValueError(
            "Modeling panel must retain realized_return_date from calendar-safe target construction."
        )
    out[REALIZED_RETURN_DATE] = _month_end(out[REALIZED_RETURN_DATE])

    if "me" not in out and "market_equity" in out:
        out["me"] = _num(out, "market_equity")
    if "market_equity" not in out and "me" in out:
        out["market_equity"] = _num(out, "me")

    if "price_eligible" not in out:
        if "prc" not in out:
            raise ValueError("Modeling panel must retain prc or price_eligible.")
        out["price_eligible"] = _num(out, "prc").abs().ge(float(min_price))
    else:
        out["price_eligible"] = out["price_eligible"].fillna(False).astype(bool)

    if "accounting_available_from" in out and "accounting_eligibility_date" not in out:
        out["accounting_eligibility_date"] = pd.to_datetime(
            out["accounting_available_from"], errors="coerce"
        )
    if "characteristic_year" in out and "accounting_eligibility_year" not in out:
        out["accounting_eligibility_year"] = pd.to_numeric(
            out["characteristic_year"], errors="coerce"
        ).astype("Int64")
    if "characteristic_year" in out and "accounting_active_through" not in out:
        year = pd.to_numeric(out["characteristic_year"], errors="coerce").astype("Int64")
        out["accounting_active_through"] = pd.to_datetime(
            (year + 1).astype("string") + "-05-31", errors="coerce"
        )

    out["target_available"] = out[BASELINE_TARGET].notna()
    out["target_available_date"] = out[REALIZED_RETURN_DATE]
    out["baseline_predictors_complete"] = out[list(BASELINE_FEATURES)].notna().all(axis=1)
    identifiers = (
        out["permno"].notna()
        & out[FORMATION_DATE].notna()
        & out[REALIZED_RETURN_DATE].notna()
        & out.get("gvkey", pd.Series(pd.NA, index=out.index)).notna()
    )
    out["baseline_complete_case"] = (
        out["baseline_predictors_complete"]
        & out[BASELINE_TARGET].notna()
        & out.get("regime", pd.Series(pd.NA, index=out.index)).notna()
        & out["price_eligible"]
        & identifiers
    )
    return out.sort_values([FORMATION_DATE, "permno"], kind="mergesort").reset_index(drop=True)


def training_rows_observable_as_of(panel: pd.DataFrame, as_of: pd.Timestamp | str) -> pd.Series:
    """Return historical rows whose targets are observable at a formation date."""
    cutoff = pd.Timestamp(as_of) + pd.offsets.MonthEnd(0)
    return _month_end(panel[FORMATION_DATE]).lt(cutoff) & _month_end(
        panel[REALIZED_RETURN_DATE]
    ).le(cutoff)


def _require_columns(panel: pd.DataFrame) -> None:
    required = {
        "permno",
        "ticker",
        "gvkey",
        FORMATION_DATE,
        REALIZED_RETURN_DATE,
        BASELINE_TARGET,
        "prc",
        "shrout",
        "me",
        "log_me",
        "me_lag",
        "accounting_datadate",
        "fyear",
        "accounting_eligibility_date",
        "book_equity",
        "me_dec",
        "book_to_market",
        "mom_12_2",
        "vix",
        "vix_expanding_median",
        "regime",
        "price_eligible",
        "baseline_complete_case",
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"Canonical baseline panel is missing required columns: {missing}")


def _reject_standardized_features(panel: pd.DataFrame) -> None:
    names = {c.lower(): c for c in panel.columns}
    suspicious: list[str] = []
    for feature in BASELINE_FEATURES:
        for name in (
            f"{feature}_z",
            f"z_{feature}",
            f"{feature}_scaled",
            f"scaled_{feature}",
            f"{feature}_standardized",
            f"standardized_{feature}",
        ):
            if name in names:
                suspicious.append(names[name])
    if suspicious:
        raise RuntimeError(
            "Saved modeling panel contains standardized baseline-feature shadows: "
            f"{sorted(set(suspicious))}"
        )


def _validate_dates(panel: pd.DataFrame) -> None:
    formation = _month_end(panel[FORMATION_DATE])
    realized = _month_end(panel[REALIZED_RETURN_DATE])
    if formation.isna().any() or realized.isna().any():
        raise AssertionError("Formation and realized-return dates must be valid for every row.")
    if not realized.gt(formation).all():
        raise AssertionError("realized_return_date must be after formation_date.")
    expected = formation + pd.offsets.MonthEnd(1)
    if not realized.eq(expected).all():
        raise AssertionError(
            f"Target-date alignment failed for {int(realized.ne(expected).sum())} rows: "
            "next-month return must be calendar t+1."
        )


def _validate_size(panel: pd.DataFrame) -> dict[str, int]:
    me = _num(panel, "me")
    log_me = _num(panel, "log_me")
    expected = _num(panel, "prc").abs() * _num(panel, "shrout")
    valid = me.notna() & expected.notna()
    if valid.any():
        error = (me[valid] - expected[valid]).abs()
        tolerance = np.maximum(1e-8, expected[valid].abs() * 1e-10)
        if (error > tolerance).any():
            raise AssertionError("Market equity is inconsistent with abs(prc) * shrout.")
    if (me.le(0).fillna(False) & log_me.notna()).any():
        raise AssertionError("Nonpositive market equity produced a valid log_me value.")
    positive = me.gt(0).fillna(False) & log_me.notna()
    if positive.any() and not np.allclose(
        log_me[positive].to_numpy(float),
        np.log(me[positive].to_numpy(float)),
        rtol=1e-10,
        atol=1e-10,
    ):
        raise AssertionError("log_me is not the natural logarithm of positive market equity.")
    return {
        "size_valid": int(log_me.notna().sum()),
        "size_missing": int(log_me.isna().sum()),
        "market_equity_nonpositive": int(me.le(0).fillna(False).sum()),
    }


def _validate_book_to_market(panel: pd.DataFrame) -> dict[str, Any]:
    formation = _month_end(panel[FORMATION_DATE])
    available = _month_end(panel["accounting_eligibility_date"])
    if (available.notna() & available.gt(formation)).any():
        raise AssertionError("Future accounting information enters the formation panel.")

    if "characteristic_year" in panel:
        observed = pd.to_numeric(panel["characteristic_year"], errors="coerce").astype("Int64")
        expected = pd.Series(
            np.where(formation.dt.month.ge(6), formation.dt.year, formation.dt.year - 1),
            index=panel.index,
            dtype="Int64",
        )
        has_accounting = panel["accounting_datadate"].notna()
        if (has_accounting & observed.ne(expected).fillna(True)).any():
            raise AssertionError("Book-to-market characteristic-year assignment violates June/May timing.")
        datadate = pd.to_datetime(panel["accounting_datadate"], errors="coerce")
        if (
            has_accounting
            & datadate.dt.year.astype("Int64").ne(observed - 1).fillna(True)
        ).any():
            raise AssertionError("Accounting datadate year conflicts with the June assignment.")

    bm = _num(panel, "book_to_market")
    be = _num(panel, "book_equity")
    me_dec = _num(panel, "me_dec")
    valid = bm.notna()
    if valid.any():
        if (be[valid] <= 0).any() or (me_dec[valid] <= 0).any():
            raise AssertionError("Valid book-to-market requires positive book equity and December ME.")
        if not np.allclose(
            bm[valid].to_numpy(float),
            (be[valid] / me_dec[valid]).to_numpy(float),
            rtol=1e-9,
            atol=1e-12,
        ):
            raise AssertionError("book_to_market does not equal book_equity / me_dec.")
        median = float(bm[valid].median())
        if not 0.001 <= median <= 50.0:
            raise RuntimeError("Book-to-market scale strongly indicates a unit error.")
    else:
        median = float("nan")
    return {
        "book_to_market_valid": int(valid.sum()),
        "book_to_market_median": median,
    }


def _validate_price(panel: pd.DataFrame, min_price: float) -> None:
    expected = _num(panel, "prc").abs().ge(float(min_price)).fillna(False)
    observed = panel["price_eligible"].fillna(False).astype(bool)
    if not observed.eq(expected).all():
        raise AssertionError("price_eligible does not match the formation-month price screen.")


def _validate_regime(
    panel: pd.DataFrame,
    reference_regimes: pd.DataFrame | None,
) -> dict[str, Any]:
    monthly = panel[
        [FORMATION_DATE, "vix", "vix_expanding_median", "regime"]
    ].copy()
    monthly[FORMATION_DATE] = _month_end(monthly[FORMATION_DATE])
    consistency = monthly.groupby(FORMATION_DATE)[
        ["vix", "vix_expanding_median", "regime"]
    ].nunique(dropna=False)
    if (consistency.max(axis=1) > 1).any():
        raise AssertionError("VIX/regime values are not unique within formation month.")
    monthly = monthly.drop_duplicates(FORMATION_DATE).sort_values(FORMATION_DATE)
    vix = _num(monthly, "vix")
    median = _num(monthly, "vix_expanding_median")
    implied = pd.Series(np.where(vix > median, "HIGH", "LOW"), index=monthly.index)
    observed = monthly["regime"].astype("string").str.upper()
    valid = vix.notna() & median.notna() & observed.notna()
    if valid.any() and not observed[valid].eq(implied[valid]).all():
        raise AssertionError("Regime labels do not match VIX_t > expanding-median_t.")

    status = "classification_verified"
    if reference_regimes is not None and not reference_regimes.empty:
        ref = reference_regimes.copy()
        date_col = FORMATION_DATE if FORMATION_DATE in ref else "date"
        ref[date_col] = _month_end(ref[date_col])
        ref = ref.rename(columns={date_col: FORMATION_DATE})[
            [FORMATION_DATE, "vix", "vix_expanding_median", "regime"]
        ].drop_duplicates(FORMATION_DATE)
        check = monthly.merge(
            ref,
            on=FORMATION_DATE,
            how="left",
            suffixes=("", "_reference"),
            validate="one_to_one",
        )
        if (check["vix_reference"].isna() & check["vix"].notna()).any():
            raise AssertionError("Reference VIX history is missing formation months.")
        for column in ("vix", "vix_expanding_median"):
            left = pd.to_numeric(check[column], errors="coerce")
            right = pd.to_numeric(check[f"{column}_reference"], errors="coerce")
            if not np.all(np.isclose(left, right, rtol=1e-10, atol=1e-10, equal_nan=True)):
                raise AssertionError(
                    f"Panel {column} differs from the historical-only VIX reference series."
                )
        if not check["regime"].astype("string").str.upper().fillna("<NA>").eq(
            check["regime_reference"].astype("string").str.upper().fillna("<NA>")
        ).all():
            raise AssertionError("Panel regime differs from the historical-only VIX reference series.")
        status = "historical_reference_verified"
    return {
        "regime_validation": status,
        "high_months": int(observed.eq("HIGH").sum()),
        "low_months": int(observed.eq("LOW").sum()),
    }


def validate_baseline_panel(
    panel: pd.DataFrame,
    *,
    min_price: float = 5.0,
    reference_regimes: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Run hard leakage/integrity checks and return aggregate metadata."""
    _require_columns(panel)
    _reject_standardized_features(panel)
    duplicates = int(panel.duplicated(["permno", FORMATION_DATE]).sum())
    if duplicates:
        raise RuntimeError(
            f"Canonical modeling panel has {duplicates} duplicate security-month rows."
        )
    _validate_dates(panel)
    _validate_price(panel, min_price)
    size = _validate_size(panel)
    bm = _validate_book_to_market(panel)
    regime = _validate_regime(panel, reference_regimes)
    complete = panel["baseline_complete_case"].fillna(False).astype(bool)
    if not complete.any():
        raise RuntimeError("Complete-case baseline modeling sample is empty.")
    return {
        "rows": int(len(panel)),
        "unique_permnos": int(panel["permno"].nunique()),
        "duplicate_security_month_rows": duplicates,
        "formation_start": str(panel[FORMATION_DATE].min().date()),
        "formation_end": str(panel[FORMATION_DATE].max().date()),
        "realized_return_start": str(panel[REALIZED_RETURN_DATE].min().date()),
        "realized_return_end": str(panel[REALIZED_RETURN_DATE].max().date()),
        "complete_case_rows": int(complete.sum()),
        "complete_case_permnos": int(panel.loc[complete, "permno"].nunique()),
        **size,
        **bm,
        **regime,
    }


def sample_flow_report(panel: pd.DataFrame, *, min_price: float = 5.0) -> pd.DataFrame:
    missing = pd.Series(np.nan, index=panel.index)
    conditions = [
        ("CRSP eligible universe", pd.Series(True, index=panel.index)),
        ("Valid next-month target", panel[BASELINE_TARGET].notna()),
        (f"Price >= ${min_price:g}", panel["price_eligible"].fillna(False).astype(bool)),
        ("Valid size", panel["log_me"].notna()),
        ("Valid momentum", panel["mom_12_2"].notna()),
        ("Valid CCM match", panel.get("gvkey", missing).notna()),
        ("Valid book equity", _num(panel, "book_equity").gt(0)),
        ("Valid book-to-market", panel["book_to_market"].notna()),
        ("Valid VIX regime", panel["vix"].notna() & panel["regime"].notna()),
    ]
    keep = pd.Series(True, index=panel.index)
    rows: list[dict[str, Any]] = []
    for stage, condition in conditions:
        keep &= condition.fillna(False)
        rows.append(
            {
                "Stage": stage,
                "Observations": int(keep.sum()),
                "Unique PERMNOs": int(panel.loc[keep, "permno"].nunique()),
            }
        )
    complete = keep & panel["baseline_complete_case"].fillna(False).astype(bool)
    rows.append(
        {
            "Stage": "Complete baseline predictor set",
            "Observations": int(complete.sum()),
            "Unique PERMNOs": int(panel.loc[complete, "permno"].nunique()),
        }
    )
    return pd.DataFrame(rows)


def missingness_report(panel: pd.DataFrame) -> pd.DataFrame:
    fields = (
        BASELINE_TARGET,
        "me",
        "log_me",
        "me_lag",
        "book_equity",
        "me_dec",
        "book_to_market",
        "mom_12_2",
        "vix",
        "regime",
    )
    total = len(panel)
    rows = []
    for field in fields:
        series = panel[field] if field in panel else pd.Series(np.nan, index=panel.index)
        missing = int(series.isna().sum())
        rows.append(
            {
                "variable": field,
                "missing_count": missing,
                "missing_percentage": 100.0 * missing / total if total else np.nan,
                "nonmissing_count": int(total - missing),
            }
        )
    return pd.DataFrame(rows)


def _distribution_row(name: str, series: pd.Series) -> dict[str, Any]:
    x = pd.to_numeric(series, errors="coerce").dropna()
    row: dict[str, Any] = {
        "variable": name,
        "count": int(len(x)),
        "mean": float(x.mean()) if len(x) else np.nan,
        "std": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
        "min": float(x.min()) if len(x) else np.nan,
    }
    for q, label in QUANTILES:
        row[label] = float(x.quantile(q)) if len(x) else np.nan
    row["max"] = float(x.max()) if len(x) else np.nan
    return row


def predictor_summary_report(panel: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([_distribution_row(f, panel[f]) for f in BASELINE_FEATURES])


def target_summary_report(panel: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([_distribution_row(BASELINE_TARGET, panel[BASELINE_TARGET])])


def coverage_by_year_report(panel: pd.DataFrame) -> pd.DataFrame:
    x = panel.copy()
    x["year"] = _month_end(x[FORMATION_DATE]).dt.year
    regimes = x[["year", FORMATION_DATE, "regime"]].drop_duplicates(FORMATION_DATE)
    regimes["HIGH_months"] = regimes["regime"].astype("string").str.upper().eq("HIGH")
    regimes["LOW_months"] = regimes["regime"].astype("string").str.upper().eq("LOW")
    regime_counts = regimes.groupby("year", as_index=False).agg(
        HIGH_months=("HIGH_months", "sum"),
        LOW_months=("LOW_months", "sum"),
    )
    grouped = x.groupby("year", as_index=False).agg(
        stock_month_observations=("permno", "size"),
        unique_securities=("permno", "nunique"),
        valid_size_count=("log_me", lambda s: int(s.notna().sum())),
        valid_book_to_market_count=("book_to_market", lambda s: int(s.notna().sum())),
        valid_momentum_count=("mom_12_2", lambda s: int(s.notna().sum())),
        complete_baseline_count=(
            "baseline_complete_case",
            lambda s: int(s.fillna(False).astype(bool).sum()),
        ),
    )
    return grouped.merge(regime_counts, on="year", how="left").sort_values("year")


def monthly_predictor_coverage_report(panel: pd.DataFrame) -> pd.DataFrame:
    x = panel.copy()
    x["all_three"] = x[list(BASELINE_FEATURES)].notna().all(axis=1)
    return (
        x.groupby(FORMATION_DATE, as_index=False)
        .agg(
            eligible_securities=("permno", "nunique"),
            valid_size=("log_me", lambda s: int(s.notna().sum())),
            valid_book_to_market=("book_to_market", lambda s: int(s.notna().sum())),
            valid_momentum=("mom_12_2", lambda s: int(s.notna().sum())),
            all_three_predictors=("all_three", lambda s: int(s.sum())),
            complete_baseline=(
                "baseline_complete_case",
                lambda s: int(s.fillna(False).astype(bool).sum()),
            ),
        )
        .sort_values(FORMATION_DATE)
        .reset_index(drop=True)
    )


def predictor_correlations_report(panel: pd.DataFrame) -> pd.DataFrame:
    complete = panel[list(BASELINE_FEATURES)].dropna().astype(float)
    pairs = [
        (BASELINE_FEATURES[i], BASELINE_FEATURES[j])
        for i in range(len(BASELINE_FEATURES))
        for j in range(i + 1, len(BASELINE_FEATURES))
    ]
    rows: list[dict[str, Any]] = []
    for method in ("pearson", "spearman"):
        corr = complete.corr(method=method)
        for a, b in pairs:
            rows.append(
                {
                    "scope": "pooled",
                    "method": method,
                    "feature_1": a,
                    "feature_2": b,
                    "correlation": float(corr.loc[a, b]),
                    "months": np.nan,
                    "observations": int(len(complete)),
                }
            )
    monthly: dict[tuple[str, str, str], list[float]] = {}
    for _, group in panel.groupby(FORMATION_DATE, sort=True):
        g = group[list(BASELINE_FEATURES)].dropna().astype(float)
        if len(g) < 3:
            continue
        for method in ("pearson", "spearman"):
            corr = g.corr(method=method)
            for a, b in pairs:
                value = float(corr.loc[a, b])
                if np.isfinite(value):
                    monthly.setdefault((method, a, b), []).append(value)
    for (method, a, b), values in monthly.items():
        rows.append(
            {
                "scope": "average_monthly_cross_section",
                "method": method,
                "feature_1": a,
                "feature_2": b,
                "correlation": float(np.mean(values)),
                "months": int(len(values)),
                "observations": np.nan,
            }
        )
    return pd.DataFrame(rows)


def extreme_observation_reports(
    panel: pd.DataFrame,
    *,
    n_each_tail: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary: list[dict[str, Any]] = []
    audits: list[pd.DataFrame] = []
    lineage = [
        c
        for c in (
            "permno",
            "ticker",
            "gvkey",
            FORMATION_DATE,
            REALIZED_RETURN_DATE,
            "accounting_datadate",
            "fyear",
            "accounting_eligibility_date",
            "book_equity",
            "me_dec",
            "prc",
            "shrout",
            "me",
            "log_me",
            "book_to_market",
            "mom_12_2",
            BASELINE_TARGET,
            "vix",
            "regime",
        )
        if c in panel
    ]
    for variable in EXTREME_VARIABLES:
        values = pd.to_numeric(panel[variable], errors="coerce").dropna()
        if values.empty:
            continue
        lo = float(values.quantile(0.001))
        hi = float(values.quantile(0.999))
        summary.append(
            {
                "variable": variable,
                "count": int(len(values)),
                "minimum": float(values.min()),
                "p0.1": lo,
                "p99.9": hi,
                "maximum": float(values.max()),
                "count_at_or_below_p0.1": int((values <= lo).sum()),
                "count_at_or_above_p99.9": int((values >= hi).sum()),
            }
        )
        for tail, index in (
            ("low", values.nsmallest(n_each_tail).index),
            ("high", values.nlargest(n_each_tail).index),
        ):
            part = panel.loc[index, lineage].copy()
            part.insert(0, "audit_variable", variable)
            part.insert(1, "tail", tail)
            audits.append(part)
    audit = pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()
    return pd.DataFrame(summary), audit


def monthly_complete_case_summary(monthly: pd.DataFrame) -> dict[str, int | float]:
    counts = pd.to_numeric(monthly["complete_baseline"], errors="coerce").dropna()
    if counts.empty:
        return {"minimum": 0, "median": 0.0, "maximum": 0}
    return {
        "minimum": int(counts.min()),
        "median": float(counts.median()),
        "maximum": int(counts.max()),
    }


def model_loader_smoke_test(panel: pd.DataFrame) -> dict[str, Any]:
    """Fit tiny in-memory models only to verify the finalized X/y interface."""
    from rdsrp.models.elastic_net import ElasticNetModel
    from rdsrp.models.xgboost_model import XGBoostModel

    sample = panel.loc[
        panel["baseline_complete_case"].fillna(False),
        [*BASELINE_FEATURES, BASELINE_TARGET],
    ].replace([np.inf, -np.inf], np.nan).dropna().head(500)
    if len(sample) < 20:
        raise RuntimeError("Too few complete rows for the model-loader smoke test.")
    x = sample[list(BASELINE_FEATURES)].to_numpy(float)
    y = sample[BASELINE_TARGET].to_numpy(float)
    train_n = max(15, int(len(sample) * 0.8))
    x_test = x[train_n : train_n + 5] if train_n < len(x) else x[:1]

    elastic = ElasticNetModel(
        {"alpha": 0.001, "l1_ratio": 0.5, "max_iter": 2_000},
        random_state=42,
    )
    elastic_fit = elastic.fit(x[:train_n], y[:train_n])
    elastic_pred = elastic.predict(elastic_fit, x_test)

    xgb = XGBoostModel(
        {"n_estimators": 5, "max_depth": 2, "n_jobs": 1, "tree_method": "hist"},
        random_state=42,
    )
    xgb_fit = xgb.fit(x[:train_n], y[:train_n])
    xgb_pred = xgb.predict(xgb_fit, x_test)
    if not np.isfinite(elastic_pred).all() or not np.isfinite(xgb_pred).all():
        raise RuntimeError("Model-loader smoke test produced nonfinite predictions.")
    return {
        "rows_used": int(len(sample)),
        "features": list(BASELINE_FEATURES),
        "target": BASELINE_TARGET,
        "elastic_net": "pass",
        "xgboost": "pass",
    }


def validation_markdown(
    metadata: dict[str, Any],
    missingness: pd.DataFrame,
    monthly_summary: dict[str, int | float],
) -> str:
    def coverage(variable: str) -> str:
        row = missingness.loc[missingness["variable"].eq(variable)]
        if row.empty:
            return "n/a"
        return f"{100.0 - float(row.iloc[0]['missing_percentage']):.2f}%"

    return "\n".join(
        [
            "# Baseline Modeling Panel Validation",
            "",
            "## Sample",
            "",
            f"- Formation period: {metadata['formation_start']} through {metadata['formation_end']}",
            f"- Realized-return period: {metadata['realized_return_start']} through {metadata['realized_return_end']}",
            f"- Observations: {metadata['rows']:,}",
            f"- Securities: {metadata['unique_permnos']:,}",
            f"- Complete baseline observations: {metadata['complete_case_rows']:,}",
            "",
            "## Baseline predictors",
            "",
            "- Size: log_me",
            "- Book-to-market: book_to_market",
            "- Momentum 12-2: mom_12_2",
            "- Target: next_month_return",
            "",
            "## Predictor coverage",
            "",
            f"- log_me: {coverage('log_me')}",
            f"- book_to_market: {coverage('book_to_market')}",
            f"- mom_12_2: {coverage('mom_12_2')}",
            "",
            "## Accounting timing",
            "",
            "Accounting lineage is retained and validation rejects accounting data used before its June eligibility date.",
            "",
            "## Target timing",
            "",
            "Every realized-return date is exactly the next calendar month-end after formation, and training eligibility is governed by target observability.",
            "",
            "## Volatility regimes",
            "",
            f"- HIGH months: {metadata['high_months']:,}",
            f"- LOW months: {metadata['low_months']:,}",
            f"- Timing check: {metadata['regime_validation']}",
            "",
            "## Duplicate-key check",
            "",
            f"Duplicate permno + formation_date rows: {metadata['duplicate_security_month_rows']}",
            "",
            "## Monthly cross-sectional coverage",
            "",
            f"- Minimum complete-case securities: {monthly_summary['minimum']}",
            f"- Median complete-case securities: {monthly_summary['median']}",
            f"- Maximum complete-case securities: {monthly_summary['maximum']}",
            "",
            "## Data-quality observations",
            "",
            "Raw economically defined predictors are retained. This stage applies neither full-sample standardization nor implicit winsorization.",
            "",
            "## Current limitations",
            "",
            "Row-level WRDS-derived panel and extreme-observation audit files remain local and are excluded from public version control.",
            "",
            "## Next Stage",
            "",
            "Full baseline out-of-sample model estimation uses this validated canonical panel with Elastic Net and XGBoost.",
            "",
        ]
    )
