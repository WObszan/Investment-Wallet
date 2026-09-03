#cos tu bedzie
import numpy as np
import pandas as pd
import pandas_ta as ta
import xgboost as xgb
import shap
from sklearn.metrics import accuracy_score
import optuna

TRADING_DAYS_PER_YEAR = 252

def prepare_futures(df: pd.DataFrame, benchmark_df: pd.DataFrame) -> pd.DataFrame:
    """Creates technical features for ML model"""
    data = df.copy()
    benchmark_data = benchmark_df.copy()
    
    #logarithmic returns
    data["log_returns_1d"] = np.log(data['Close'] / data['Close'].shift(1))
    data["log_returns_5d"] = np.log(data['Close'] / data['Close'].shift(5))
    data["log_returns_20d"] = np.log(data['Close'] / data['Close'].shift(20))
    data["log_returns_60d"] = np.log(data['Close'] / data['Close'].shift(60))

    # Rolling Volatility
    data['Volatility_14d'] = data['log_returns_1d'].rolling(14).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    data['Volatility_30d'] = data['log_returns_1d'].rolling(30).std() * np.sqrt(TRADING_DAYS_PER_YEAR)

    # RSI, MACD, SMA
    data['rsi'] = ta.rsi(data['Close'], length=14)

    macd_df = ta.macd(data['Close'], fast=12, slow=26, signal=9)
    data['MACD_line'] = macd_df['MACD_12_26_9']
    data['MACD_signal'] = macd_df['MACDs_12_26_9']
    data['MACD_hist'] = macd_df['MACDh_12_26_9']

    data['SMA_50'] = ta.sma(data['Close'], length=50)
    data['SMA_200'] = ta.sma(data['Close'], length=200)

    # Price distance from the average (percentage or logarithmic)
    data['Dist_SMA50'] = (data['Close'] - data['SMA_50']) / data['SMA_50']

    data['Relative_Return_20d'] = data['log_returns_20d'] - np.log(benchmark_data['Close'] / benchmark_data['Close'].shift(20))

    data['Volume_SMA_20'] = data['Volume'].rolling(20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA_20']

    # On-Balance Volume by pandas-ta
    data['OBV'] = ta.obv(data['Close'], data['Volume'])

    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)

    data['Month'] = data.index.month
    data['Quarter'] = data.index.quarter

    #definicion of target
    future_close = df['Close'].shift(-5)
    data['Target'] = (future_close > df['Close']).astype(float)
    data.loc[future_close.isna(), 'Target'] = np.nan
    data = data.dropna()
    data['Target'] = data['Target'].astype(int)
    return data



def objective(trial, X_train, y_train, X_test, y_test):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'eval_metric': 'logloss',
        'random_state': 42
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)

def train_xgboost(data: pd.DataFrame):
    features = [
        "log_returns_1d", "log_returns_5d", "log_returns_20d", "log_returns_60d",
        "Volatility_14d", "Volatility_30d", 'rsi', 
        'MACD_line', 'MACD_signal', 'MACD_hist',
        'Dist_SMA50',
        'Relative_Return_20d',
        'Volume_Ratio', 'OBV',
        'Month', 'Quarter'
    ]
    X = data[features]
    y = data['Target']

    n = len(data)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)
 
    X_train, X_val, X_test = X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:]
    y_train, y_val, y_test = y.iloc[:train_end], y.iloc[train_end:val_end], y.iloc[val_end:]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction='maximize')
    
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=30)
    
    best_params = study.best_params
    best_params['eval_metric'] = 'logloss'
    best_params['random_state'] = 42

    print(f"Best params from Optuna: {best_params}")

    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])

    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train_full, y_train_full)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    return model, X_test, y_test, shap_values


def run_backtest(model, X_test, y_test, original_data, horizon=5):
    """Runs a simple backtest of the ML-driven strategy vs Buy and Hold."""

    forward_return = np.log(original_data['Close'].shift(-horizon) / original_data['Close'])
    
    test_data = original_data.loc[X_test.index].copy()
    test_data['Forward_Return'] = forward_return.loc[X_test.index]

    #Generate model predictions for the test set
    preds = model.predict(X_test)
    test_data['Model_Signal'] = preds

    test_data = test_data.iloc[::horizon].copy()

    test_data['Strategy_Return'] = test_data['Forward_Return'] * test_data['Model_Signal']
    test_data['Strategy_Return'] = test_data['Strategy_Return'].fillna(0)
    test_data['Forward_Return'] = test_data['Forward_Return'].fillna(0)
    
    # Calculate cumulative returns (Equity Curve)
    test_data['Market_Cumulative'] = np.exp(test_data['Forward_Return'].cumsum()) - 1
    test_data['Strategy_Cumulative'] = np.exp(test_data['Strategy_Return'].cumsum()) - 1
    
    return test_data[['Close', 'Market_Cumulative', 'Strategy_Cumulative', 'Model_Signal']]