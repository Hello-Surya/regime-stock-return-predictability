def apply_one_way_cost(gross_return: float, turnover: float, cost_bps: float) -> float:
    return float(gross_return - turnover * cost_bps / 10000.0)
