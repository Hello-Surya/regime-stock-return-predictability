"""M10 scientific-invariance and accounting tests; all data here are synthetic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rdsrp.inference.hac import BASELINE_HAC_LAG
from rdsrp.transaction_costs.pipeline import (
    apply_cost_scenarios, bps_to_decimal, calculate_turnover, drift_weights,
    reconstruct_portfolios, target_weights, validate_assignments,
    validate_gross_replication, validate_authoritative_long_short,
    _transaction_summaries, _turnover_summary,
)


def _decile(p: list[int], me: list[float] | None = None, ret: list[float] | None = None) -> pd.DataFrame:
    return pd.DataFrame({"permno":p, "weighting_variable":me or [1.0]*len(p),
                         "realized_next_month_return":ret or [.01]*len(p)})


def _synthetic_assignments(nmonths: int = 15) -> pd.DataFrame:
    rows=[]
    dates=pd.date_range("2020-01-31",periods=nmonths,freq="ME")
    for j,date in enumerate(dates):
        for model in ("Elastic Net","XGBoost"):
            for d in range(1,11):
                for k in range(2):
                    # Changing some members assures both old-only exits and new entrants.
                    permno=1000+d*50+k+(j % 2)*10
                    rows.append({"formation_date":date,"realized_return_date":date+pd.offsets.MonthEnd(1),
                        "permno":permno,"ticker":f"T{permno}","model":model,"decile":d,
                        "realized_next_month_return":0.001*d+0.002*(j%3)+0.0001*k+0.0002*(d-5)*np.sin(j*.7),
                        "weighting_variable":float(d*10+k+1),
                        "regime_binary":"LOW" if j%2==0 else "HIGH",
                        "regime_3state":("LOW","MIDDLE","HIGH")[j%3]})
    return pd.DataFrame(rows)


def test_basis_points() -> None:
    assert bps_to_decimal(50)==.005
    assert bps_to_decimal(1)==.0001
    with pytest.raises(ValueError):
        bps_to_decimal(-1)


def test_ew_target_weights() -> None:
    assert target_weights(_decile([1,2,3]),"EW")=={1:1/3,2:1/3,3:1/3}


def test_vw_target_weights_and_normalization() -> None:
    weights=target_weights(_decile([1,2,3],[2,3,5]),"VW")
    assert weights=={1:.2,2:.3,3:.5}
    assert sum(weights.values())==1


def test_vw_invalid_me_excluded_no_ew_fallback() -> None:
    result=target_weights(_decile([1,2,3],[-1,np.nan,9]),"VW")
    assert result=={3:1.0}
    with pytest.raises(AssertionError):
        target_weights(_decile([1,2],[-1,0]),"VW")


def test_drift_weights() -> None:
    shifted=drift_weights({1:.5,2:.5},{1:.10,2:-.10})
    assert shifted[1]==pytest.approx(.55)
    assert shifted[2]==pytest.approx(.45)


def test_simple_turnover_and_cost() -> None:
    turnover=calculate_turnover({1:.25,2:.75},{1:.5,2:.5})
    assert turnover==pytest.approx(.25)
    assert turnover*bps_to_decimal(50)==pytest.approx(.00125)


def test_new_position_and_exit() -> None:
    weights={1:.5,2:.5}
    new={2:.5,3:.5}
    assert calculate_turnover(new,weights)==pytest.approx(.5)


def test_initial_formation_cost() -> None:
    assert calculate_turnover({1:.25,2:.75},None)==1.0


def test_turnover_no_zeros_if_full_reallocation() -> None:
    assert calculate_turnover({3:.5,4:.5},{1:.5,2:.5})==pytest.approx(1.0)


def test_invalid_weights_and_returns_fail() -> None:
    with pytest.raises(AssertionError):
        calculate_turnover({1:.9}, {1:1.0})
    with pytest.raises(AssertionError):
        drift_weights({1:.5,2:.5},{1:.1})
    with pytest.raises(AssertionError):
        drift_weights({1:.5,2:.5},{1:-1.,2:-1.})
    with pytest.raises(AssertionError):
        drift_weights({1:1.},{1:-1.01})


def test_m8_membership_and_regimes_preserved() -> None:
    assign=_synthetic_assignments()
    original=assign.copy(deep=True)
    rebuilt, turn, weights=reconstruct_portfolios(assign)
    pd.testing.assert_frame_equal(assign,original)
    assert len(rebuilt)==len(assign.formation_date.unique())*4
    assert len(turn)==len(rebuilt)*10
    assert set(rebuilt.regime_3state)=={"LOW","MIDDLE","HIGH"}
    assert set(rebuilt.regime_binary)=={"LOW","HIGH"}
    assert set(weights.decile)=={"D1","D10"}
    assert turn.groupby(["model","weighting","formation_date"]).size().eq(10).all()
    assert (turn.turnover>=0).all() and (turn.turnover<=1+1e-12).all()
    assert (turn.loc[turn.initial_formation,"turnover"]==1).all()
    # No price, membership, or forecast input is modified.


def test_zero_cost_reproduction_and_cost_both_legs() -> None:
    rebuilt, _, _=reconstruct_portfolios(_synthetic_assignments())
    net=apply_cost_scenarios(rebuilt)
    z=net[net.transaction_cost_bps==0]
    assert np.array_equal(z.D10_minus_D1_gross,z.D10_minus_D1_net)
    assert (z.long_short_cost==0).all()
    f=net[net.transaction_cost_bps==50]
    assert np.allclose(f.long_short_cost,(f.D10_turnover+f.D1_turnover)*.005)
    assert np.allclose(f.D10_minus_D1_net,f.D10_minus_D1_gross-f.long_short_cost)
    assert np.allclose(f.D10_minus_D1_net,f.D10_net+f.D1_short_leg_net_contribution)
    # D1_net is long-only; it is not the contribution of a short.
    assert np.allclose(f.D1_net,f.D1_gross-f.D1_cost)
    first=f[f.formation_date==f.formation_date.min()]
    assert np.allclose(first.long_short_cost,.01)


def test_future_return_leakage() -> None:
    base=_synthetic_assignments(nmonths=9)
    changed=base.copy()
    first=base.formation_date.min()
    # Changing returns realized at first_date+1 does not affect turnover at first_date.
    changed.loc[changed.formation_date==first,"realized_next_month_return"]+=.4
    a,_,_=reconstruct_portfolios(base)
    b,_,_=reconstruct_portfolios(changed)
    for c in ("D1_turnover","D10_turnover"):
        assert np.allclose(a.loc[a.formation_date==first,c],b.loc[b.formation_date==first,c])
    # Changed previous returns are observable at the next rebalance.
    # Common constant shift across a leg preserves drift proportions only if returns differ.
    assert a.formation_date.equals(b.formation_date)


def test_future_assignment_leakage() -> None:
    a=_synthetic_assignments(nmonths=9)
    last=a.formation_date.max()
    b=a.copy()
    b.loc[b.formation_date==last,"weighting_variable"]*=2
    older=a.formation_date.unique()[-2]
    ra,_,_=reconstruct_portfolios(a)
    rb,_,_=reconstruct_portfolios(b)
    assert np.allclose(ra.loc[ra.formation_date==older,"D1_turnover"],rb.loc[rb.formation_date==older,"D1_turnover"])


def test_no_future_weight_used() -> None:
    a=_synthetic_assignments(nmonths=9)
    date=a.formation_date.min()
    b=a.copy()
    later=b.formation_date.sort_values().unique()[1]
    b.loc[b.formation_date==later,"weighting_variable"]*=4
    ra,_,_=reconstruct_portfolios(a)
    rb,_,_=reconstruct_portfolios(b)
    assert np.allclose(ra.loc[ra.formation_date==date,"D10"],rb.loc[rb.formation_date==date,"D10"])


def test_gross_m8_replication_fail_closed() -> None:
    frame,_,_=reconstruct_portfolios(_synthetic_assignments())
    validate_gross_replication(frame,frame.copy())
    corrupt=frame.copy()
    corrupt.loc[0,"D10"]+=0.01
    with pytest.raises(AssertionError,match="replication"):
        validate_gross_replication(frame,corrupt)


def test_long_short_gross_replication_fail_closed() -> None:
    monthly,_,_=reconstruct_portfolios(_synthetic_assignments())
    gross=monthly[["formation_date","realized_return_date","model","regime_binary",
                  "regime_3state","weighting","D10","D1","D10_minus_D1","n_stocks"]].rename(
                  columns={"D10":"D10_return","D1":"D1_return"})
    validate_authoritative_long_short(monthly,gross)
    gross.loc[0,"D10_minus_D1"]+=.1
    with pytest.raises(AssertionError,match="gross"):
        validate_authoritative_long_short(monthly,gross)


def test_regime_counts_and_break_even() -> None:
    rebuilt,turn,_=reconstruct_portfolios(_synthetic_assignments())
    net=apply_cost_scenarios(rebuilt)
    stats,break_even=_transaction_summaries(net)
    turnover=_turnover_summary(rebuilt,turn)
    assert len(stats)==4*2*(1+2+3)
    assert len(break_even)==4*(1+2+3)
    assert set(turnover.regime_definition)=={"overall","binary","three_state"}
    first=net[net.transaction_cost_bps==50]
    assert (first.long_short_cost>=0).all()
    assert (stats.loc[stats.transaction_cost_bps==50,"cost_attrition"]>=-1e-12).all()
    assert BASELINE_HAC_LAG==6


def test_does_not_redefine_deciles() -> None:
    source=_synthetic_assignments(9)
    source.loc[source.decile==1,"decile"]=10
    with pytest.raises(AssertionError,match="Missing decile"):
        validate_assignments(source)


def test_missing_calendar_month_rejected() -> None:
    source=_synthetic_assignments()
    missing_month=sorted(source.formation_date.unique())[3]
    source=source[source.formation_date!=missing_month]
    with pytest.raises(AssertionError,match="Nonconsecutive"):
        validate_assignments(source)
