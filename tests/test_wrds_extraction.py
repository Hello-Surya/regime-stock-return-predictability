from __future__ import annotations

import pandas as pd

from rdsrp.data.wrds import (
    CrspSource,
    build_crsp_query,
    normalize_crsp_monthly,
    resolve_crsp_monthly_source,
)


class FakeDb:
    def __init__(self, libraries, tables):
        self._libraries = libraries
        self._tables = tables

    def list_libraries(self):
        return self._libraries

    def list_tables(self, library):
        return self._tables.get(library, [])


def test_resolver_prefers_current_ciz_monthly_source():
    db = FakeDb(
        ["crsp", "crsp_m_stock"],
        {
            "crsp": ["msf", "msenames"],
            "crsp_m_stock": ["stksecurityinfohist", "msf_v2"],
        },
    )
    source = resolve_crsp_monthly_source(db)
    assert source == CrspSource("crsp_m_stock", "msf_v2", "ciz")


def test_resolver_supports_legacy_fallback():
    db = FakeDb(["crspa"], {"crspa": ["msf", "msenames"]})
    source = resolve_crsp_monthly_source(db)
    assert source.format == "legacy"
    assert source.names_table == "msenames"


def test_ciz_query_encodes_common_stock_and_exchange_filters():
    source = CrspSource("crsp_m_stock", "msf_v2", "ciz")
    query = build_crsp_query(source, "2020-01-01", "2020-12-31", min_price=5.0)
    lower = query.lower()
    assert "sharetype = 'ns'" in lower
    assert "securitytype = 'eqty'" in lower
    assert "securitysubtype = 'com'" in lower
    assert "issuertype in ('acor', 'corp')" in lower
    assert "primaryexch in ('n', 'a', 'q')" in lower
    assert "conditionaltype = 'rw'" in lower
    assert "tradingstatusflg = 'a'" in lower
    assert "abs(mthprc) >= 5.00000000" in lower


def test_normalization_computes_lagged_market_equity_when_needed():
    source = CrspSource("crsp", "msf", "legacy", "msenames")
    raw = pd.DataFrame(
        {
            "permno": [1, 1, 1],
            "date": pd.to_datetime(["2020-01-15", "2020-02-15", "2020-03-15"]),
            "ret": [0.01, 0.02, -0.01],
            "prc": [-10.0, -11.0, -10.5],
            "shrout": [100.0, 100.0, 100.0],
            "market_equity": [1000.0, 1100.0, 1050.0],
            "me_lag": [None, None, None],
        }
    )
    out = normalize_crsp_monthly(raw, source)
    assert out.loc[0, "date"] == pd.Timestamp("2020-01-31")
    assert out.loc[0, "prc"] == 10.0
    assert pd.isna(out.loc[0, "me_lag"])
    assert out.loc[1, "me_lag"] == 1000.0
    assert out.loc[2, "me_lag"] == 1100.0


def test_ciz_fixed_top_n_query_uses_start_of_sample_formation_universe():
    source = CrspSource("crsp", "msf_v2", "ciz")
    query = build_crsp_query(
        source,
        "2012-01-01",
        "2025-12-31",
        min_price=5.0,
        top_n_by_market_cap=300,
        include_ticker="AAPL",
        universe_formation_date="2012-01-31",
    )
    lower = query.lower()
    assert "with formation as" in lower
    assert "mthcaldt = '2012-01-31'" in lower
    assert "order by mthcap desc" in lower
    assert "limit 300" in lower
    assert "permno in (select permno from formation)" in lower
    assert "upper(ticker) = 'aapl'" in lower
    assert "partition by" not in lower
    assert lower.count("abs(mthprc) >= 5.00000000") == 1
