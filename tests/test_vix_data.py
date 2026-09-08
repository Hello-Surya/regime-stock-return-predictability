from __future__ import annotations

import pandas as pd

from rdsrp.data.vix import (
    VixSource,
    load_cboe_vix_history,
    resolve_vix_source,
    vix_daily_to_monthly,
)


class FakeVixDb:
    def list_libraries(self):
        return ["crsp", "cboe_all"]

    def list_tables(self, library):
        return ["vix"] if library == "cboe_all" else []

    def raw_sql(self, query, **kwargs):
        assert "cboe_all.vix" in query.lower()
        return pd.DataFrame({"date": pd.to_datetime(["2020-01-02"]), "vix": [12.0], "vixo": [13.0]})


class FakeNonVixNamedDb:
    """Mimic WRDS exposing the VIX product with a non-VIX physical table name."""

    def list_libraries(self):
        return ["cboe_all"]

    def list_tables(self, library):
        return ["dailyprices"] if library == "cboe_all" else []

    def raw_sql(self, query, **kwargs):
        assert "cboe_all.dailyprices" in query.lower()
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-02"]),
                "open": [13.0],
                "high": [14.0],
                "low": [12.0],
                "close": [13.5],
            }
        )


def test_vix_resolver_finds_cboe_all_vix():
    source = resolve_vix_source(FakeVixDb())
    assert source == VixSource("cboe_all", "vix", "date", "vix")


def test_vix_resolver_does_not_require_vix_in_table_name():
    source = resolve_vix_source(FakeNonVixNamedDb())
    assert source == VixSource("cboe_all", "dailyprices", "date", "close")


def test_official_cboe_csv_loader_uses_close_and_filters_dates(monkeypatch):
    raw = pd.DataFrame(
        {
            "DATE": ["01/02/2020", "01/03/2020", "01/06/2020"],
            "OPEN": [12.0, 13.0, 14.0],
            "HIGH": [13.0, 14.0, 15.0],
            "LOW": [11.0, 12.0, 13.0],
            "CLOSE": [12.5, 13.5, 14.5],
        }
    )
    monkeypatch.setattr(pd, "read_csv", lambda *_args, **_kwargs: raw.copy())
    out = load_cboe_vix_history("2020-01-03", "2020-01-06")
    assert out["date"].tolist() == [pd.Timestamp("2020-01-03"), pd.Timestamp("2020-01-06")]
    assert out["vix"].tolist() == [13.5, 14.5]
    assert set(out["source_type"]) == {"cboe_public_csv"}


def test_daily_vix_uses_last_observation_in_calendar_month():
    daily = pd.DataFrame({
        "date": pd.to_datetime(["2020-01-02", "2020-01-31", "2020-02-03", "2020-02-28"]),
        "vix": [12.0, 18.0, 17.0, 14.0],
    })
    monthly = vix_daily_to_monthly(daily)
    assert monthly["date"].tolist() == [pd.Timestamp("2020-01-31"), pd.Timestamp("2020-02-29")]
    assert monthly["vix"].tolist() == [18.0, 14.0]
