import numpy as np
import pandas as pd
from rdsrp.data.modeling_panel import (
    build_complete_predictor_panel,
    construction_summary,
    distribution_report,
    sample_flow,
)


def test_complete_panel_has_unique_keys_and_three_predictors():
    dates = pd.date_range("2019-01-31", "2021-07-31", freq="ME")
    crsp = pd.DataFrame({
        "permno": [1]*len(dates), "ticker": ["TEST"]*len(dates), "date": dates,
        "ret": [0.01]*len(dates), "prc": [10.0]*len(dates), "shrout": [10_000.0]*len(dates),
    })
    funda = pd.DataFrame({
        "gvkey":["001","001"], "datadate":pd.to_datetime(["2019-12-31","2020-12-31"]),
        "fyear":[2019,2020],"fyr":[12,12],"seq":[50.0,60.0],
        "pstkrv":[0.0,0.0],"txditc":[0.0,0.0],
    })
    ccm = pd.DataFrame({
        "gvkey":["001"],"lpermno":[1],"linktype":["LC"],"linkprim":["P"],
        "linkdt":pd.to_datetime(["2000-01-01"]),"linkenddt":[pd.NaT],
    })
    regimes = pd.DataFrame({"date":dates,"vix":20.0,"vix_expanding_median":19.0,"regime":"HIGH"})
    panel, audit = build_complete_predictor_panel(crsp, funda, ccm, regimes)
    assert panel.duplicated(["permno","date"]).sum() == 0
    june20 = panel.loc[panel.date.eq(pd.Timestamp("2020-06-30"))].iloc[0]
    may20 = panel.loc[panel.date.eq(pd.Timestamp("2020-05-31"))].iloc[0]
    june21 = panel.loc[panel.date.eq(pd.Timestamp("2021-06-30"))].iloc[0]
    assert pd.isna(may20.book_to_market)
    assert np.isclose(june20.book_to_market, 0.5)
    assert np.isclose(june21.book_to_market, 0.6)
    assert audit["duplicate_permno_date_rows"] == 0
    assert set(["log_me","book_to_market","mom_12_2"]).issubset(panel.columns)
    assert not distribution_report(panel).empty
    flow = sample_flow(panel)
    assert "Complete baseline predictor set" in set(flow.stage)


def test_construction_summary_research_sample_counts_override_buffer_audit():
    panel = pd.DataFrame({
        "permno": [1, 1, 2],
        "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-01-31"]),
        "gvkey": ["001", "001", "002"],
        "book_equity": [10.0, 10.0, np.nan],
        "book_to_market": [0.5, 0.5, np.nan],
        "log_me": [1.0, 1.1, 1.2],
        "mom_12_2": [0.1, 0.2, 0.3],
        "next_month_return": [0.01, 0.02, 0.03],
        "regime": ["LOW", "LOW", "HIGH"],
    })
    comp = pd.DataFrame({"gvkey": ["001", "002"], "datadate": pd.to_datetime(["2019-12-31", "2019-12-31"])})
    ccm = pd.DataFrame({"gvkey": ["001", "002"]})
    audit = {
        "valid_book_to_market_months": 999,
        "complete_predictor_rows": 999,
        "duplicate_permno_date_rows": 999,
        "some_buffer_diagnostic": 7,
    }
    summary = construction_summary(comp, ccm, panel, audit)
    assert summary["valid_book_to_market_months"] == 2
    assert summary["complete_predictor_rows"] == 2
    assert summary["duplicate_permno_date_rows"] == 0
    assert summary["some_buffer_diagnostic"] == 7
