import numpy as np
import pandas as pd
import pytest

from rdsrp.features.book_equity import (
    assign_book_to_market_monthly,
    construct_annual_book_to_market,
    construct_book_equity,
    filter_ccm_links,
    link_accounting_to_permno,
    prepare_annual_accounting,
)


def test_book_equity_fallback_and_preferred_hierarchy():
    f = pd.DataFrame({
        "gvkey": ["1", "2", "3"], "datadate": pd.to_datetime(["2019-12-31"]*3),
        "seq": [100.0, np.nan, np.nan], "ceq": [1.0, 80.0, np.nan], "pstk": [9.0, 20.0, 5.0],
        "at": [200.0, 200.0, 150.0], "lt": [80.0, 90.0, 50.0],
        "pstkrv": [10.0, np.nan, np.nan], "pstkl": [9.0, 15.0, np.nan],
        "txditc": [5.0, np.nan, np.nan], "txdb": [np.nan, 2.0, np.nan], "itcb": [np.nan, 3.0, np.nan],
    })
    out = construct_book_equity(f)
    assert out["stockholders_equity_source"].tolist() == ["seq", "ceq_plus_pstk", "at_minus_lt"]
    assert out["preferred_stock_source"].tolist() == ["pstkrv", "pstkl", "pstk"]
    assert np.isclose(out.loc[0, "book_equity"], 95.0)
    assert np.isclose(out.loc[1, "book_equity"], 90.0)
    assert np.isclose(out.loc[2, "book_equity"], 95.0)


def test_missing_tax_and_preferred_are_explicit_zero():
    f = pd.DataFrame({"gvkey":["1"], "datadate":pd.to_datetime(["2019-12-31"]), "seq":[100.0]})
    out = construct_book_equity(f)
    assert out.loc[0, "deferred_taxes_source"] == "assumed_zero_missing"
    assert out.loc[0, "preferred_stock_source"] == "assumed_zero_missing"
    assert out.loc[0, "book_equity"] == 100.0


def test_negative_be_cannot_make_valid_bm():
    f = pd.DataFrame({"gvkey":["1"], "datadate":pd.to_datetime(["2019-12-31"]), "seq":[-10.0]})
    annual, _ = prepare_annual_accounting(f)
    ccm = pd.DataFrame({"gvkey":["1"],"lpermno":[1],"linktype":["LC"],"linkprim":["P"],"linkdt":pd.to_datetime(["2000-01-01"]),"linkenddt":[pd.NaT]})
    linked, _ = link_accounting_to_permno(annual, ccm)
    crsp = pd.DataFrame({"permno":[1],"date":pd.to_datetime(["2019-12-31"]),"market_equity":[100_000.0]})
    bm, _ = construct_annual_book_to_market(linked, crsp)
    assert pd.isna(bm.loc[0,"book_to_market"])


def test_ccm_effective_dates_and_invalid_link_types():
    annual = pd.DataFrame({"gvkey":["1"],"datadate":pd.to_datetime(["2019-12-31"]),"characteristic_year":pd.Series([2020],dtype="Int64")})
    ccm = pd.DataFrame({
        "gvkey":["1","1","1","1"],"lpermno":[1,2,3,4],"linktype":["LC","LC","LD","LC"],"linkprim":["P","P","P","P"],
        "linkdt":pd.to_datetime(["2000-01-01","2020-01-01","2000-01-01","2000-01-01"]),"linkenddt":pd.to_datetime([None,None,None,"2018-12-31"]),
    })
    linked, _ = link_accounting_to_permno(annual, ccm)
    assert linked["permno"].tolist() == [1]
    filtered, _ = filter_ccm_links(ccm)
    assert 3 not in filtered["lpermno"].tolist()


def test_multiple_duplicate_links_resolve_deterministically():
    annual = pd.DataFrame({"gvkey":["1"],"datadate":pd.to_datetime(["2019-12-31"]),"characteristic_year":pd.Series([2020],dtype="Int64")})
    ccm = pd.DataFrame({
        "gvkey":["1","1"],"lpermno":[1,1],"linktype":["LU","LC"],"linkprim":["C","P"],
        "linkdt":pd.to_datetime(["2000-01-01","2000-01-01"]),"linkenddt":[pd.NaT,pd.NaT],
    })
    linked, stats = link_accounting_to_permno(annual, ccm)
    assert len(linked) == 1 and linked.iloc[0]["linktype"] == "LC" and linked.iloc[0]["linkprim"] == "P"
    assert stats["duplicate_ccm_rows_resolved"] == 1


def test_june_assignment_and_no_future_accounting():
    annual = pd.DataFrame({
        "permno":pd.Series([1,1],dtype="Int64"), "gvkey":["1","1"],
        "characteristic_year":pd.Series([2020,2021],dtype="Int64"),
        "datadate":pd.to_datetime(["2019-12-31","2020-12-31"]),
        "accounting_available_from":pd.to_datetime(["2020-06-30","2021-06-30"]),
        "book_equity":[100.0,200.0],"me_dec":[100.0,100.0],"book_to_market":[1.0,2.0],
    })
    months = pd.DataFrame({"permno":[1]*4,"date":pd.to_datetime(["2020-05-31","2020-06-30","2021-05-31","2021-06-30"])})
    out, _ = assign_book_to_market_monthly(months, annual)
    assert pd.isna(out.loc[0,"book_to_market"])
    assert out.loc[1,"book_to_market"] == 1.0
    assert out.loc[2,"book_to_market"] == 1.0
    assert out.loc[3,"book_to_market"] == 2.0


def test_december_me_units_and_bm_arithmetic():
    linked = pd.DataFrame({
        "gvkey":["1"],"permno":pd.Series([1],dtype="Int64"),"characteristic_year":pd.Series([2020],dtype="Int64"),
        "book_equity":[50.0],"datadate":pd.to_datetime(["2019-12-31"]),
    })
    crsp = pd.DataFrame({"permno":[1],"date":pd.to_datetime(["2019-12-31"]),"market_equity":[100_000.0]})
    out, _ = construct_annual_book_to_market(linked, crsp)
    assert out.loc[0,"me_dec"] == 100.0
    assert out.loc[0,"book_to_market"] == 0.5


def test_share_classes_use_firm_level_december_me():
    linked = pd.DataFrame({
        "gvkey":["1","1"],"permno":pd.Series([1,2],dtype="Int64"),"characteristic_year":pd.Series([2020,2020],dtype="Int64"),
        "book_equity":[90.0,90.0],"datadate":pd.to_datetime(["2019-12-31","2019-12-31"]),
    })
    crsp = pd.DataFrame({"permno":[1,2],"date":pd.to_datetime(["2019-12-31","2019-12-31"]),"market_equity":[60_000.0,30_000.0]})
    out, _ = construct_annual_book_to_market(linked, crsp)
    assert set(out["me_dec"]) == {90.0}
    assert set(out["book_to_market"]) == {1.0}


def test_explicit_future_accounting_availability_is_rejected():
    annual = pd.DataFrame({
        "permno": pd.Series([1], dtype="Int64"),
        "gvkey": ["1"],
        "characteristic_year": pd.Series([2020], dtype="Int64"),
        "datadate": pd.to_datetime(["2019-12-31"]),
        "accounting_available_from": pd.to_datetime(["2021-06-30"]),
        "book_equity": [100.0],
        "me_dec": [100.0],
        "book_to_market": [1.0],
    })
    months = pd.DataFrame({"permno": [1], "date": pd.to_datetime(["2020-06-30"])})
    with pytest.raises(RuntimeError, match="before its allowed June date"):
        assign_book_to_market_monthly(months, annual)


def test_incomplete_linked_share_class_december_me_leaves_firm_bm_missing():
    linked = pd.DataFrame({
        "gvkey": ["1", "1"],
        "permno": pd.Series([1, 2], dtype="Int64"),
        "characteristic_year": pd.Series([2020, 2020], dtype="Int64"),
        "book_equity": [90.0, 90.0],
        "datadate": pd.to_datetime(["2019-12-31", "2019-12-31"]),
    })
    crsp = pd.DataFrame({
        "permno": [1, 2],
        "date": pd.to_datetime(["2019-12-31", "2019-11-30"]),
        "market_equity": [60_000.0, 30_000.0],
    })
    out, stats = construct_annual_book_to_market(linked, crsp)
    assert out["book_to_market"].isna().all()
    assert stats["gvkey_years_incomplete_december_me"] == 1


def test_nullable_accounting_values_produce_missing_bm_without_boolean_error():
    linked = pd.DataFrame({
        "gvkey": ["1", "2"],
        "permno": pd.Series([1, 2], dtype="Int64"),
        "characteristic_year": pd.Series([2020, 2020], dtype="Int64"),
        "book_equity": pd.Series([pd.NA, 50.0], dtype="Float64"),
        "datadate": pd.to_datetime(["2019-12-31", "2019-12-31"]),
    })
    crsp = pd.DataFrame({
        "permno": [1, 2],
        "date": pd.to_datetime(["2019-12-31", "2019-12-31"]),
        "market_equity": [100_000.0, pd.NA],
    })
    out, _ = construct_annual_book_to_market(linked, crsp)
    assert out["book_to_market"].isna().all()
