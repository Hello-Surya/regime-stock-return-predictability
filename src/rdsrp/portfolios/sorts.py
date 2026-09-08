import pandas as pd

def assign_deciles(df: pd.DataFrame, prediction_col: str = "prediction", date_col: str = "date") -> pd.DataFrame:
    out=df.copy()
    def _rank(values: pd.Series) -> pd.Series:
        valid=values.notna(); result=pd.Series(pd.NA,index=values.index,dtype="Int64")
        if valid.sum()<10: return result
        ranks=values.loc[valid].rank(method="first"); result.loc[valid]=(pd.qcut(ranks,10,labels=False).astype(int)+1).astype("Int64"); return result
    out["decile"]=out.groupby(date_col,group_keys=False)[prediction_col].transform(_rank).astype("Int64")
    return out
