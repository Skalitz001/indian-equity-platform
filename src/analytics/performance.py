import pandas as pd
import numpy as np


def total_return(equity_curve: pd.Series) -> float:
    return equity_curve.iloc[-1] - 1.0


def annualized_return(
    returns: pd.Series, periods_per_year: int = 252
) -> float:
    return returns.mean() * periods_per_year


def annualized_volatility(
    returns: pd.Series, periods_per_year: int = 252
) -> float:
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    excess = returns - risk_free_rate / periods_per_year
    vol = annualized_volatility(excess, periods_per_year)
    return 0.0 if vol == 0 else excess.mean() / vol * np.sqrt(periods_per_year)


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    return drawdown.min()


def win_rate(returns: pd.Series) -> float:
    trades = returns[returns != 0]
    if len(trades) == 0:
        return 0.0
    return (trades > 0).mean()
