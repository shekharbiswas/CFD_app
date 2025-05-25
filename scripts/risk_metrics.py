import pandas as pd
import numpy as np

def calculate_metrics_summary(portfolio_name: str,
                              portfolio_values: pd.Series,
                              portfolio_returns_original: pd.Series,
                              rfr_annual: float,
                              initial_capital: float,
                              trading_days_per_year: int) -> dict:
    """Calculates various performance metrics for a given portfolio."""
    metrics = {"Portfolio": portfolio_name}

    if portfolio_values.empty or portfolio_returns_original.empty or initial_capital is None:
        nan_metrics_keys = [
            "Total Return", "Annualized Return", "Annualized Volatility", "Sharpe Ratio",
            "Max Drawdown", "Calmar Ratio", "Sortino Ratio", "Daily VaR 95%",
            "Daily CVaR 95%", "Omega Ratio", "Skewness", "Kurtosis", "Best Day",
            "Worst Day", "Win Rate %", "Average Win %", "Average Loss %",
            "Profit Factor", "Recovery Factor"
        ]
        for key in nan_metrics_keys:
            metrics[key] = np.nan
        return metrics

    # Construct the series of actual daily returns experienced by the portfolio
    if initial_capital == 0:
        first_day_actual_return = np.nan
    else:
        first_day_actual_return = (portfolio_values.iloc[0] / initial_capital) - 1 if len(portfolio_values) > 0 else np.nan

    if len(portfolio_values) == 1:
        portfolio_returns_for_stats = pd.Series([first_day_actual_return]).dropna()
    elif len(portfolio_values) > 1 :
        subsequent_returns = portfolio_returns_original.iloc[1:] # from .pct_change()
        portfolio_returns_for_stats = pd.concat([
            pd.Series([first_day_actual_return]),
            subsequent_returns
        ]).reset_index(drop=True).dropna()
    else: # portfolio_values is empty, handled by initial check
        portfolio_returns_for_stats = pd.Series(dtype=float)


    if portfolio_returns_for_stats.empty and not pd.isna(first_day_actual_return) and len(portfolio_values) == 1:
         portfolio_returns_for_stats = pd.Series([first_day_actual_return])
    
    final_value = portfolio_values.iloc[-1] if not portfolio_values.empty else initial_capital
    num_actual_periods = len(portfolio_returns_for_stats)

    # 1. Total Return
    metrics["Total Return"] = (final_value / initial_capital) - 1 if initial_capital != 0 else np.nan

    # 2. Annualized Return
    if pd.isna(metrics["Total Return"]) or num_actual_periods == 0:
        metrics["Annualized Return"] = np.nan
    else:
        base_for_annualization = (1 + metrics["Total Return"])
        metrics["Annualized Return"] = (base_for_annualization ** (trading_days_per_year / num_actual_periods)) - 1 if num_actual_periods > 0 else metrics["Total Return"]
    annualized_return = metrics["Annualized Return"]

    # 3. Annualized Volatility
    if num_actual_periods > 1:
        metrics["Annualized Volatility"] = portfolio_returns_for_stats.std(ddof=1) * np.sqrt(trading_days_per_year)
    elif num_actual_periods == 1 and not pd.isna(portfolio_returns_for_stats.iloc[0]):
        metrics["Annualized Volatility"] = 0.0 # Volatility of a single point is 0
    else:
        metrics["Annualized Volatility"] = np.nan
    annualized_volatility = metrics["Annualized Volatility"]
    
    # 4. Sharpe Ratio
    if pd.isna(annualized_return) or pd.isna(annualized_volatility) or pd.isna(rfr_annual):
        metrics["Sharpe Ratio"] = np.nan
    elif annualized_volatility == 0:
        excess_return = annualized_return - rfr_annual
        if excess_return > 0: metrics["Sharpe Ratio"] = np.inf
        elif excess_return < 0: metrics["Sharpe Ratio"] = -np.inf
        else: metrics["Sharpe Ratio"] = 0.0
    else:
        metrics["Sharpe Ratio"] = (annualized_return - rfr_annual) / annualized_volatility

    # 5. Max Drawdown
    values_for_drawdown = pd.concat([pd.Series([initial_capital]), portfolio_values], ignore_index=True)
    rolling_max = values_for_drawdown.expanding().max()
    daily_drawdown = (values_for_drawdown / rolling_max) - 1.0
    metrics["Max Drawdown"] = daily_drawdown.min()
    max_drawdown_value = metrics["Max Drawdown"]

    # 6. Calmar Ratio
    if pd.isna(annualized_return) or pd.isna(max_drawdown_value) or max_drawdown_value == 0:
        metrics["Calmar Ratio"] = np.inf if annualized_return > 0 and max_drawdown_value == 0 else (0.0 if annualized_return == 0 and max_drawdown_value == 0 else np.nan)
    else:
        metrics["Calmar Ratio"] = annualized_return / abs(max_drawdown_value)

    # 7. Sortino Ratio
    rfr_daily = rfr_annual / trading_days_per_year
    if not portfolio_returns_for_stats.empty:
        downside_diff = portfolio_returns_for_stats - rfr_daily
        downside_returns_sq = np.square(np.minimum(0, downside_diff))
        mean_sq_downside = np.mean(downside_returns_sq) if len(downside_returns_sq) > 0 else 0
        downside_dev_annualized = np.sqrt(mean_sq_downside) * np.sqrt(trading_days_per_year)
    else:
        downside_dev_annualized = np.nan

    if pd.isna(annualized_return) or pd.isna(downside_dev_annualized) or pd.isna(rfr_annual):
        metrics["Sortino Ratio"] = np.nan
    elif downside_dev_annualized == 0:
        excess_return = annualized_return - rfr_annual
        if excess_return > 0: metrics["Sortino Ratio"] = np.inf
        elif excess_return < 0: metrics["Sortino Ratio"] = -np.inf
        else: metrics["Sortino Ratio"] = 0.0
    else:
        metrics["Sortino Ratio"] = (annualized_return - rfr_annual) / downside_dev_annualized

    # 8. Daily Value at Risk (VaR) 95%
    metrics["Daily VaR 95%"] = portfolio_returns_for_stats.quantile(0.05) if num_actual_periods > 0 else np.nan
    var_95 = metrics["Daily VaR 95%"]

    # 9. Conditional Value at Risk (CVaR) 95%
    if not portfolio_returns_for_stats.empty and not pd.isna(var_95):
        tail_returns = portfolio_returns_for_stats[portfolio_returns_for_stats <= var_95]
        metrics["Daily CVaR 95%"] = tail_returns.mean() if not tail_returns.empty else var_95
    else:
        metrics["Daily CVaR 95%"] = np.nan

    # 10. Omega Ratio
    if not portfolio_returns_for_stats.empty and num_actual_periods > 0:
        gains_over_rfr = (portfolio_returns_for_stats[portfolio_returns_for_stats > rfr_daily] - rfr_daily).sum()
        losses_under_rfr = (rfr_daily - portfolio_returns_for_stats[portfolio_returns_for_stats < rfr_daily]).sum() # Losses are positive
        metrics["Omega Ratio"] = gains_over_rfr / losses_under_rfr if losses_under_rfr != 0 else (np.inf if gains_over_rfr > 0 else np.nan)
    else:
        metrics["Omega Ratio"] = np.nan
        
    metrics["Skewness"] = portfolio_returns_for_stats.skew() if num_actual_periods > 0 else np.nan
    metrics["Kurtosis"] = portfolio_returns_for_stats.kurtosis() if num_actual_periods > 0 else np.nan # Excess kurtosis
    metrics["Best Day"] = portfolio_returns_for_stats.max() if num_actual_periods > 0 else np.nan
    metrics["Worst Day"] = portfolio_returns_for_stats.min() if num_actual_periods > 0 else np.nan

    if num_actual_periods > 0:
        positive_returns = portfolio_returns_for_stats[portfolio_returns_for_stats > 0]
        negative_returns = portfolio_returns_for_stats[portfolio_returns_for_stats < 0]
        metrics["Win Rate %"] = (len(positive_returns) / num_actual_periods) * 100
        metrics["Average Win %"] = positive_returns.mean() * 100 if not positive_returns.empty else 0.0
        metrics["Average Loss %"] = negative_returns.mean() * 100 if not negative_returns.empty else 0.0
        gross_profit = positive_returns.sum()
        gross_loss_abs = abs(negative_returns.sum())
        metrics["Profit Factor"] = gross_profit / gross_loss_abs if gross_loss_abs != 0 else (np.inf if gross_profit > 0 else np.nan)
    else:
        metrics["Win Rate %"] = metrics["Average Win %"] = metrics["Average Loss %"] = metrics["Profit Factor"] = np.nan

    # 19. Recovery Factor
    if pd.isna(metrics["Total Return"]) or pd.isna(max_drawdown_value) or max_drawdown_value == 0:
        metrics["Recovery Factor"] = np.inf if not pd.isna(metrics["Total Return"]) and metrics["Total Return"] > 0 and max_drawdown_value == 0 else (0.0 if not pd.isna(metrics["Total Return"]) and metrics["Total Return"] == 0 and max_drawdown_value == 0 else np.nan)
    else:
        metrics["Recovery Factor"] = metrics["Total Return"] / abs(max_drawdown_value)
        
    return metrics