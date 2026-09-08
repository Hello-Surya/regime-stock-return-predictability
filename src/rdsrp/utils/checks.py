"""Data-quality and leakage checks."""
from __future__ import annotations
import pandas as pd

def validate_oos_predictions(predictions: pd.DataFrame) -> None:
    if predictions.empty: raise ValueError("No OOS predictions were produced.")
    p = predictions.copy(); p["date"] = pd.to_datetime(p["date"]); p["train_end"] = pd.to_datetime(p["train_end"])
    if not (p["train_end"] < p["date"]).all(): raise AssertionError("Leakage detected: at least one training end date is not before its test date.")
    if p[["actual", "prediction"]].isna().any().any(): raise AssertionError("Predictions contain missing realized or predicted returns.")

def data_validation_report(panel: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows=[{"check":"sample_start","value":str(pd.to_datetime(panel["date"]).min().date())},{"check":"sample_end","value":str(pd.to_datetime(panel["date"]).max().date())},{"check":"n_rows","value":int(len(panel))},{"check":"n_stocks","value":int(panel["permno"].nunique())},{"check":"duplicate_permno_date","value":int(panel.duplicated(["permno","date"]).sum())},{"check":"target_missing_rate","value":float(panel["next_month_return"].isna().mean())}]
    for regime,count in panel["regime"].value_counts(dropna=False).items(): rows.append({"check":f"regime_count_{regime}","value":int(count)})
    for col in feature_cols:
        rows += [{"check":f"missing_rate_{col}","value":float(panel[col].isna().mean())},{"check":f"p01_{col}","value":float(panel[col].quantile(0.01))},{"check":f"p50_{col}","value":float(panel[col].quantile(0.50))},{"check":f"p99_{col}","value":float(panel[col].quantile(0.99))}]
    for q in (0.01,0.50,0.99): rows.append({"check":f"return_q{int(q*100):02d}","value":float(panel["ret"].quantile(q))})
    return pd.DataFrame(rows)

def crsp_monthly_validation_report(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty: return pd.DataFrame([{"check":"n_rows","value":0}])
    frame=data.copy(); frame["date"]=pd.to_datetime(frame["date"])
    rows=[{"check":"sample_start","value":str(frame["date"].min().date())},{"check":"sample_end","value":str(frame["date"].max().date())},{"check":"n_rows","value":int(len(frame))},{"check":"n_stocks","value":int(frame["permno"].nunique())},{"check":"n_months","value":int(frame["date"].nunique())},{"check":"duplicate_permno_date","value":int(frame.duplicated(["permno","date"]).sum())}]
    for col in ["ret","prc","shrout","market_equity","me_lag"]:
        if col in frame:
            values=pd.to_numeric(frame[col],errors="coerce"); rows.append({"check":f"missing_rate_{col}","value":float(values.isna().mean())})
            for q,label in [(0.01,"p01"),(0.50,"p50"),(0.99,"p99")]: rows.append({"check":f"{label}_{col}","value":float(values.quantile(q))})
    return pd.DataFrame(rows)
