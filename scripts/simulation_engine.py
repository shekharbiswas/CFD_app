import pandas as pd
import numpy as np

def simulate_portfolio_A(df_data, initial_capital, equity_alloc_A):
    """Simulates Model A: Static allocation."""
    # print(f"Simulating Model A with initial capital: {initial_capital}, equity_alloc: {equity_alloc_A}")
    if df_data.empty or 'SP500_Return' not in df_data.columns:
        # print("Model A: Empty data or missing SP500_Return column.")
        return pd.Series(dtype=float)

    portfolio_values_list = []
    current_value = initial_capital
    
    # The first day's return applies to the initial capital.
    # The loop should start from the first row of df_data.
    # portfolio_values_list will store end-of-day values.

    for i in range(len(df_data)):
        sp500_return = df_data['SP500_Return'].iloc[i]
        if pd.isna(sp500_return): # Handle potential NaNs in returns
            # print(f"Model A: NaN SP500_Return at index {i}, date {df_data['date'].iloc[i]}. Holding value.")
            portfolio_return = 0
        else:
            portfolio_return = equity_alloc_A * sp500_return
        
        current_value *= (1 + portfolio_return)
        portfolio_values_list.append(current_value)
        
    if not portfolio_values_list:
        return pd.Series(dtype=float)
        
    return pd.Series(portfolio_values_list, index=df_data.index)


def simulate_portfolio_B_momentum(df_data_with_signals, initial_capital, cfg):
    """Simulates Model B: Dynamic VIX momentum hedge."""
    # print(f"Simulating Model B (Momentum) with initial capital: {initial_capital}")
    if df_data_with_signals.empty:
        # print("Model B: Empty data.")
        return pd.Series(dtype=float)

    essential_cols = ['SP500_Return', 'S&P500', 'SOFR_Rate', 
                      'Short_Signal_Today', 'Cover_Signal_Momentum_Today', 'Cover_Signal_Absolute_VIX_Today']
    for col in essential_cols:
        if col not in df_data_with_signals.columns:
            print(f"Model B: Missing essential column '{col}'.")
            return pd.Series(dtype=float)

    total_value = initial_capital
    equity_value = total_value * cfg.EQUITY_ALLOC_B
    cash_value = total_value * cfg.CASH_ALLOC_B
    
    portfolio_values_history = []
    cfd_active = False
    cfd_entry_sp500_price = 0.0
    cfd_notional_value = 0.0
    cfd_margin_account_deduction = 0.0

    for i in range(len(df_data_with_signals)):
        row = df_data_with_signals.iloc[i]
        sp500_return_t = row['SP500_Return']
        sp500_price_t = row['S&P500']
        # SOFR rate needs to be present and numeric. If it's NaN, simulation might break or give bad results.
        sofr_annual_t = row['SOFR_Rate'] if pd.notna(row['SOFR_Rate']) else 0.0 # Default to 0 if NaN


        trigger_short_today = row['Short_Signal_Today']
        trigger_cover_momentum = row['Cover_Signal_Momentum_Today']
        trigger_cover_absolute_vix = row['Cover_Signal_Absolute_VIX_Today']
        
        if pd.isna(sp500_return_t) or pd.isna(sp500_price_t):
            # print(f"Model B: NaN market data at index {i}, date {row['date']}. Holding value.")
            portfolio_values_history.append(total_value) # Append current total value if data is bad
            continue

        # 1. Equity component grows/shrinks
        equity_value *= (1 + sp500_return_t)

        # 2. CFD Financing & P&L (if active)
        if cfd_active:
            # Financing cost for short CFD is -(SOFR - Broker Fee) = Broker Fee - SOFR
            # If Broker Fee > SOFR, it's a cost. If SOFR > Broker Fee, it's a credit.
            # The notebook had (sofr_annual_t - BROKER_FEE_ANNUALIZED)
            # For a short CFD, you typically pay financing based on a benchmark rate + broker spread,
            # or receive financing if benchmark rate is negative or below broker's base.
            # Assuming the broker fee is what you pay *on top* of the interbank rate for financing.
            # So, cost is (Interbank Rate + Broker Spread).
            # The notebook seems to model it as (SOFR - Broker Fee), implying SOFR is received and Broker Fee is paid.
            # Let's stick to the notebook's (SOFR_annual - BROKER_FEE_ANNUALIZED) / 365
            # financing_rate_daily = (sofr_annual_t - cfg.BROKER_FEE_ANNUALIZED) / 365.0 # Correct as per notebook
            # A positive financing_rate_daily means a gain for the short CFD holder (cash increases)
            # A negative financing_rate_daily means a cost for the short CFD holder (cash decreases)
            
            # Clarification on CFD financing for a SHORT position:
            # You are effectively borrowing the stock to sell it.
            # If you short, you receive cash, but also have to pay dividends out.
            # Financing costs are typically (benchmark rate + broker's borrow fee) * notional.
            # The notebook formula (sofr_annual_t - BROKER_FEE_ANNUALIZED) suggests:
            # sofr_annual_t is a rate received (e.g. on cash from short sale proceeds if held by broker)
            # BROKER_FEE_ANNUALIZED is a cost (e.g. stock borrowing fee or general CFD fee).
            # If `financing_rate_daily` is positive, it's a credit to cash. If negative, a debit.
            # This seems like a net financing effect.
            financing_rate_daily = (sofr_annual_t - cfg.BROKER_FEE_ANNUALIZED) / 365.0
            cfd_financing_impact = cfd_notional_value * financing_rate_daily # This is gain/loss on the cash from short position
            cash_value += cfd_financing_impact # Add to cash
            
        # 3. Hedging Logic
        if cfd_active and (trigger_cover_momentum or trigger_cover_absolute_vix):
            # P&L from short CFD: -(Current Price - Entry Price) / Entry Price * Notional
            # = (Entry Price - Current Price) / Entry Price * Notional
            # = (1 - Current Price / Entry Price) * Notional
            cfd_pnl = cfd_notional_value * (1 - (sp500_price_t / cfd_entry_sp500_price))
            cash_value += cfd_pnl
            
            # Spread cost on closing
            # Notional at close for spread calculation (current market value of the hedged amount)
            notional_at_close_for_spread = cfd_notional_value * (sp500_price_t / cfd_entry_sp500_price) # This is current value of initial notional
            spread_cost = cfg.SPREAD_COST_PERCENT * notional_at_close_for_spread
            cash_value -= spread_cost
            
            # Return margin
            cash_value += cfd_margin_account_deduction
            
            cfd_active = False
            cfd_entry_sp500_price = 0.0
            cfd_notional_value = 0.0
            cfd_margin_account_deduction = 0.0
        
        elif not cfd_active and trigger_short_today:
            amount_to_hedge = cfg.MODEL_B_FIXED_HEDGE_RATIO_MOMENTUM * equity_value
            margin_required = cfg.CFD_INITIAL_MARGIN_PERCENT * amount_to_hedge
            
            if cash_value >= margin_required:
                cfd_active = True
                cfd_notional_value = amount_to_hedge
                cfd_entry_sp500_price = sp500_price_t
                cfd_margin_account_deduction = margin_required
                cash_value -= cfd_margin_account_deduction
            # else: No action if insufficient margin

        # 4. Update total portfolio value
        total_value = equity_value + cash_value
        portfolio_values_history.append(total_value)
        
        # 5. Daily Rebalance (Simplified)
        equity_value = total_value * cfg.EQUITY_ALLOC_B
        cash_value = total_value * cfg.CASH_ALLOC_B
        
    if not portfolio_values_history:
        return pd.Series(dtype=float)

    return pd.Series(portfolio_values_history, index=df_data_with_signals.index)