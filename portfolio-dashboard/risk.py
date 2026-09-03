import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression



TRADING_DAYS_PER_YEAR = 252


def historical_var_cvar(daily_returns: pd.DataFrame, weights: np.ndarray, confidence: float=0.95) -> tuple[float, float]:
    portfolio_daily_returns = np.dot(daily_returns.values, weights)
    var_threshold = np.percentile(portfolio_daily_returns, (1 - confidence)*100)
    cvar = portfolio_daily_returns[portfolio_daily_returns <= var_threshold].mean()
    return -var_threshold, -cvar


def compute_capm(daily_returns: pd.DataFrame, ticker: str, benchmark: str = "^GSPC") -> dict:
    """Fit CAPM regression (asset return ~ benchmark return), return beta/alpha."""
    model = LinearRegression()
    model.fit(daily_returns[[benchmark]], daily_returns[ticker])

    beta = model.coef_[0]
    alpha = model.intercept_

    return {"beta": beta, "alpha": alpha, "model": model}


def rolling_beta(daily_returns: pd.DataFrame, ticker: str, benchmark: str, window: int = 60) -> pd.Series:
    cov = daily_returns[ticker].rolling(window=window).cov(daily_returns[benchmark])
    var = daily_returns[benchmark].rolling(window=window).var()
    return cov / var

def build_risk_profile(daily_returns: pd.DataFrame, tickers: list[str], benchmark: str = "^GSPC") -> pd.DataFrame:
    """Build the return/risk/beta/alpha/sharpe table used for clustering."""
    beta_results = {}
    alpha_results = {}
    for ticker in tickers:
        capm = compute_capm(daily_returns, ticker, benchmark)
        beta_results[ticker] = capm["beta"]
        alpha_results[ticker] = capm["alpha"]

    profile = pd.DataFrame()
    profile["returns"] = daily_returns[tickers].mean() * TRADING_DAYS_PER_YEAR
    profile["risk"] = daily_returns[tickers].std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    profile["beta"] = pd.Series(beta_results)
    profile["alpha"] = pd.Series(alpha_results)
    profile["sharpe"] = profile["returns"] / profile["risk"]

    return profile


def portfolio_sortino_ratio(daily_returns: pd.DataFrame, weights: np.ndarray, risk_free_rate: float = 0.0) -> float:
    portfolio_daily_returns = np.dot(daily_returns.values, weights)
    avg_returns = daily_returns.mean()
    annual_return = np.dot(weights, avg_returns) * TRADING_DAYS_PER_YEAR

    downside_returns = np.minimum(0, portfolio_daily_returns)
    downside_deviation = np.sqrt(np.mean(downside_returns**2) * TRADING_DAYS_PER_YEAR)
    sortino = (annual_return - risk_free_rate) / downside_deviation
    return sortino


def portfolio_max_drawdown(daily_returns: pd.DataFrame, weights: np.ndarray) -> float:
    portfolio_daily_returns = np.dot(daily_returns.values, weights)
    
    cumulative_value = pd.Series(1 + portfolio_daily_returns).cumprod()
    running_max = cumulative_value.cummax()
    drawdown = (cumulative_value - running_max) / running_max
    
    max_drawdown = drawdown.min()
    return max_drawdown


def portfolio_calmar_ratio(daily_returns: pd.DataFrame, weights: np.ndarray) -> float:
    avg_returns = daily_returns.mean()
    annual_return = np.dot(weights, avg_returns) * TRADING_DAYS_PER_YEAR

    max_dd = portfolio_max_drawdown(daily_returns, weights)

    return annual_return / abs(max_dd)


def compute_cumulative_value(daily_returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """Cumulative portfolio value over time (starting at 1.0) for fixed weights."""
    portfolio_daily_returns = np.dot(daily_returns.values, weights)
    return pd.Series(1 + portfolio_daily_returns, index=daily_returns.index).cumprod()
