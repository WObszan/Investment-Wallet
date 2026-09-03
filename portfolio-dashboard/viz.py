import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_correlation_heatmap(daily_returns: pd.DataFrame) -> go.Figure:
    """Heatmap of pairwise correlations between assets' daily returns."""
    corr = daily_returns.corr()
    return px.imshow(
        corr, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto",
    )


def plot_efficient_frontier(mc_result: pd.DataFrame, max_sharpe_idx, min_risk_idx) -> go.Figure:
    """Monte Carlo cloud with the Max Sharpe and Min Risk portfolios highlighted."""
    fig = px.scatter(
        mc_result, x="Expected Annual Risk", y="Annual Expected Return",
        color="Expected Sharpe Ratio", color_continuous_scale="Viridis",
        opacity=0.6, title="Efficient Frontier (Monte Carlo)",
    )
    fig.add_trace(go.Scatter(
        x=[mc_result.loc[max_sharpe_idx, "Expected Annual Risk"]],
        y=[mc_result.loc[max_sharpe_idx, "Annual Expected Return"]],
        mode="markers", marker=dict(size=18, symbol="star", color="red"),
        name="Max Sharpe",
    ))
    fig.add_trace(go.Scatter(
        x=[mc_result.loc[min_risk_idx, "Expected Annual Risk"]],
        y=[mc_result.loc[min_risk_idx, "Annual Expected Return"]],
        mode="markers", marker=dict(size=18, symbol="star", color="blue"),
        name="Min Risk",
    ))
    return fig


def add_exact_frontier(fig: go.Figure, frontier: pd.DataFrame, exact_max_sharpe: dict, exact_min_var: dict) -> go.Figure:
    """Overlay the exact scipy efficient frontier + exact portfolios onto an existing fig."""
    fig.add_trace(go.Scatter(
        x=frontier["Risk"], y=frontier["Target Return"],
        mode="lines", line=dict(color="green", width=2, dash="dash"),
        name="Efficient Frontier (scipy)",
    ))
    fig.add_trace(go.Scatter(
        x=[exact_max_sharpe["Annual Risk"]], y=[exact_max_sharpe["Annual Return"]],
        mode="markers", marker=dict(size=14, symbol="diamond", color="red"),
        name="Max Sharpe (scipy)",
    ))
    fig.add_trace(go.Scatter(
        x=[exact_min_var["Annual Risk"]], y=[exact_min_var["Annual Return"]],
        mode="markers", marker=dict(size=14, symbol="diamond", color="blue"),
        name="Min Variance (scipy)",
    ))
    fig.update_layout(
        legend={"x": 0.02, "y": 0.98, "xanchor": "left", "yanchor": "top", "bgcolor": "rgba(0,0,0,0.5)"}
    )
    return fig


def plot_capm_regression(combined_returns: pd.DataFrame, ticker: str, benchmark: str, pred: np.ndarray) -> go.Figure:
    """Scatter of asset vs benchmark daily returns with the fitted CAPM regression line."""
    fig = px.scatter(
        combined_returns, x=benchmark, y=ticker, opacity=0.5,
        title=f"CAPM: {ticker} vs {benchmark}",
    )
    fig.add_trace(go.Scatter(
        x=combined_returns[benchmark], y=pred, mode="lines",
        line=dict(color="red", width=2), name="CAPM Regression",
    ))
    return fig


def plot_rolling_beta(beta_series: pd.Series, ticker: str, benchmark: str) -> go.Figure:
    """Rolling beta over time with a reference line at beta = 1."""
    fig = px.line(
        x=beta_series.index, y=beta_series.values,
        labels={"x": "Data", "y": "Beta"},
        title=f"Rolling Beta: {ticker} vs {benchmark}",
    )
    fig.add_hline(y=1, line_dash="dash", line_color="gray", annotation_text="beta = 1")
    return fig


def plot_cluster_scatter(clustered: pd.DataFrame) -> go.Figure:
    """Scatter of assets by risk vs return, colored by K-Means cluster."""
    fig = px.scatter(
        clustered, x="risk", y="returns", color=clustered["Cluster"].astype(str),
        text=clustered.index, size_max=20,
        labels={"risk": "Annual Risk", "returns": "Annual Return", "color": "Cluster"},
        title="Companies grouping (Risk vs Return)",
    )
    fig.update_traces(textposition="top center", marker=dict(size=16))
    return fig


def plot_backtest_cumulative(
    frozen_cum: pd.Series, equal_cum: pd.Series, bench_cum: pd.Series, benchmark: str
) -> go.Figure:
    """Cumulative value over the test period for the frozen, equal-weight, and benchmark strategies."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=frozen_cum.index, y=frozen_cum.values, mode="lines", name="Max Sharpe (frozen)"))
    fig.add_trace(go.Scatter(x=equal_cum.index, y=equal_cum.values, mode="lines", name="Equal Weight"))
    fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum.values, mode="lines", name=f"Benchmark ({benchmark})"))
    fig.update_layout(title="Cumulative portfolio value over the test period (start = 1.0)")
    return fig

def plot_portfolio_growth(growth_df: pd.DataFrame, strategy_choice: str) -> go.Figure:
    """Fan chart showing expected, optimistic, and pessimistic portfolio growth over time."""
    growth_df = growth_df.copy()
    growth_df["Years"] = growth_df["Period"] / 12
    fig = px.line(
        growth_df, x="Years", 
        y=["Expected (Mean)", "Optimistic (+1σ)", "Pessimistic (-1σ)"],
        title=f"Portfolio Value Projection for Strategy: {strategy_choice}",
        labels={"Years": "Investment Horizon (Years)", "value": "Portfolio Value ($)"}
    )
    return fig


def plot_backtest_results(backtest_results: pd.DataFrame):
    """Creates a Plotly figure comparing ML strategy equity curve against Buy & Hold."""
    fig = px.line(
        backtest_results, 
        y=['Strategy_Cumulative', 'Market_Cumulative'],
        labels={'value': 'Cumulative Return', 'index': 'Date', 'variable': 'Strategy'},
        title="ML Strategy vs. Buy & Hold (Out-of-Sample)"
    )
    fig.data[0].name = 'ML Strategy'
    fig.data[1].name = 'Buy & Hold (Market)'
    
    # Clean up layout styling
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        legend_title="Strategy Type",
        template="plotly_dark"
    )
    return fig