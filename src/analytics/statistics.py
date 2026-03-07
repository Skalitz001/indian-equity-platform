import pandas as pd
import numpy as np


def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
    return returns.mean() * periods_per_year


def annualized_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    excess = returns - risk_free_rate / periods_per_year
    vol = annualized_volatility(excess, periods_per_year)
    if vol == 0:
        return 0.0
    return annualized_return(excess, periods_per_year) / vol
