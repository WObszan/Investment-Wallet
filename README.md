# Portfolio Optimization & Risk Analysis using Monte Carlo Simulation

## Project Overview

This project is a comprehensive **Python-based** implementation of **Modern Portfolio Theory (MPT)**. The primary goal is to analyze a portfolio of financial assets (stocks, ETFs) and identify optimal asset allocation.

The tool fetches historical market data, performs exploratory data analysis (EDA), calculates risk metrics (including **Value at Risk - VaR**), and then runs **10,000+ Monte Carlo simulations** to map the **Efficient Frontier**. Finally, it identifies the portfolio with the **maximum Sharpe Ratio** (best risk-adjusted return) and the **minimum volatility** portfolio.


##  Key Features

* **Dynamic Data Ingestion:** Fetches historical price data for any list of tickers using the `yfinance` library.
* **Data Cleaning:** Intelligently handles `NaN` values resulting from different exchange holidays (e.g., WSE vs. NYSE) by forward-filling (`ffill`) prices and filling `0` for volume.
* **Exploratory Data Analysis (EDA):** Calculates daily returns, analyzes return distributions (histograms), and visualizes the **correlation matrix** (`seaborn.heatmap`) to understand diversification benefits.
* **Risk Modeling:** Implements historical **Value at Risk (VaR)** to quantify potential losses.
* **Monte Carlo Simulation:** Runs over 10,000 simulations with random portfolio weights to generate the **Efficient Frontier**.
* **Optimization:** Identifies and highlights two key portfolios:
    1.  **Minimum Volatility Portfolio** (The "safest" option).
    2.  **Maximum Sharpe Ratio Portfolio** (The "optimal" risk-adjusted option).



##  Core Insights

* The correlation analysis confirmed (e.g., *a low correlation between Polish WIG20 assets and US tech stocks*), demonstrating the clear benefits of geographic diversification.
* The generated Efficient Frontier clearly illustrates that achieving higher expected returns requires taking on exponentially higher levels of risk (volatility).
* The Max Sharpe Ratio (optimal) portfolio is not the same as the Minimum Volatility (safest) portfolio, presenting a clear trade-off for different investment strategies.


## Tech Stack & Libraries

* **Python 3.12**
* **Pandas:** For data manipulation, time-series analysis, and `DataFrame` management.
* **NumPy:** For high-performance numerical computing and matrix operations (dot products, covariance).
* **Matplotlib & Seaborn:** For static data visualization (histograms, heatmaps, density plots).
* **yfinance:** For fetching historical market data from the Yahoo Finance API.
* **Tableau Public:** For the final interactive dashboard visualization.
* **Git & GitHub:** For version control.
