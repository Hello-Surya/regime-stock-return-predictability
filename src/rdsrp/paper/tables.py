import pandas as pd

def model_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    cols=["model","regime","n_obs","mse","rmse","mae","r2","oos_r2","oos_r2_vs_zero","correlation"]
    return metrics[cols].sort_values(["regime","model"]).reset_index(drop=True)
