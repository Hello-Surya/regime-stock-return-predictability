import pandas as pd

from rdsrp.data.accounting import (
    CcmSource,
    CompustatSource,
    build_ccm_query,
    build_compustat_query,
    resolve_ccm_source,
    resolve_compustat_source,
)


class FakeDb:
    def list_libraries(self):
        return ["other", "comp", "crsp"]

    def list_tables(self, library):
        if library == "comp":
            return ["funda"]
        if library == "crsp":
            return ["ccmxpf_lnkhist"]
        return []

    def describe_table(self, library, table):
        if (library, table) == ("comp", "funda"):
            return pd.DataFrame({"name": [
                "gvkey", "datadate", "fyear", "fyr", "seq", "ceq", "at", "lt",
                "txditc", "txdb", "itcb", "pstkrv", "pstkl", "pstk",
                "indfmt", "datafmt", "popsrc", "consol", "curcd",
            ]})
        if (library, table) == ("crsp", "ccmxpf_lnkhist"):
            return pd.DataFrame({"name": [
                "gvkey", "lpermno", "linktype", "linkprim", "linkdt", "linkenddt",
            ]})
        raise KeyError((library, table))


def test_dynamic_source_discovery_prefers_active_tables():
    db = FakeDb()
    comp = resolve_compustat_source(db)
    ccm = resolve_ccm_source(db)
    assert (comp.library, comp.table) == ("comp", "funda")
    assert (ccm.library, ccm.table) == ("crsp", "ccmxpf_lnkhist")


def test_compustat_query_has_server_side_standard_filters_and_gvkeys():
    source = CompustatSource(
        "comp", "funda",
        (
            "gvkey", "datadate", "seq", "ceq", "at", "lt", "indfmt", "datafmt",
            "popsrc", "consol", "curcd",
        ),
    )
    query = build_compustat_query(
        source,
        "1987-01-01",
        "2025-12-31",
        gvkeys=["001234", "005678"],
    )
    assert "datadate BETWEEN '1987-01-01' AND '2025-12-31'" in query
    assert "indfmt = 'INDL'" in query
    assert "datafmt = 'STD'" in query
    assert "popsrc = 'D'" in query
    assert "consol = 'C'" in query
    assert "curcd = 'USD'" in query
    assert "gvkey IN ('001234', '005678')" in query


def test_ccm_query_uses_link_history_fields_and_lpermno_alias():
    source = CcmSource(
        "crsp", "ccmxpf_lnkhist",
        ("gvkey", "lpermno", "linktype", "linkprim", "linkdt", "linkenddt"),
    )
    query = build_ccm_query(source)
    assert "FROM crsp.ccmxpf_lnkhist" in query
    assert "lpermno AS lpermno" in query
    assert "linkdt" in query and "linkenddt" in query
