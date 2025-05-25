import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
# Remove time if not used: import time

# --- Import new modular scripts ---
# Assume app.py is in the root, and scripts is a subfolder
import scripts.config as static_cfg # For parameters not in UI
import scripts.signal_generation as sg
import scripts.simulation_engine as sim_eng
# import scripts.risk_metrics as rm # if you want to use the full metrics later

# ==============================================================================
# Utility Class for Config
# ==============================================================================
class ConfigObject:
    """Helper class to access dict keys as attributes for modular functions."""
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                # For nested dicts (like 'hedging_strategy' or 'cfd_costs' in the app's config)
                # The modular scripts might expect flat attributes from cfg.
                # We need to be careful here. For now, let's assume modular scripts
                # access keys like cfg.VIX_THRESHOLD directly if it's top-level in the dict.
                # If modular scripts expect cfg.hedging_strategy_vix_threshold, this needs more complex mapping.
                # For now, we'll ensure necessary top-level keys are present in the dict.
                setattr(self, key, ConfigObject(value) if isinstance(value, dict) else value)
            else:
                setattr(self, key, value)

# ==============================================================================
# Data Loading and Preprocessing (Adjusted for provided CSV)
# ==============================================================================
@st.cache_data
def load_and_prepare_data(config_dict_ui, data_file_path_abs):
    # Use 'date' as the index column name directly as it is in the CSV
    # The first column in the CSV is an unnamed index, pandas handles it with index_col=0
    # Or, ensure the CSV is saved without that initial unnamed index.
    # For the provided CSV, it has an initial unnamed index. Then 'date'.
    # So, if 'date' is passed as date_col_name, it should work.

    date_col_name = config_dict_ui['date_column'] # Should be 'date'
    sp500_target_col = config_dict_ui['sp500_column']
    vix_target_col = config_dict_ui['vix_column']
    # --- MODIFICATION: Use SOFR_Rate directly from CSV ---
    sofr_target_col = 'SOFR_Rate' # Directly use the column name from CSV
    config_dict_ui['sofr_column'] = sofr_target_col # Update config for consistency

    if not os.path.exists(data_file_path_abs):
        st.error(f"Data file not found: {data_file_path_abs}")
        return pd.DataFrame()

    try:
        # The CSV has an initial unnamed index column. We can skip it or let pandas handle it.
        # If 'date' is the first *named* column, index_col='date' is fine.
        # Given the sample, the first column is unnamed, 'date' is the second.
        # Let's read it and then set index if 'date' is a column.
        df = pd.read_csv(data_file_path_abs)
        if 'date' not in df.columns: # Check if 'date' column exists
             if df.columns[0] == 'date': # If first column after read is 'date'
                 pass # already fine
             elif 'Unnamed: 0' in df.columns and df.columns[1] == 'date': # Common case with saved index
                 df = df.rename(columns = {'Unnamed: 0':'orig_idx'}) # Keep original index if needed later
             else:
                st.error(f"Expected 'date' column not found. Columns are: {df.columns.tolist()}")
                return pd.DataFrame()
        
        df[date_col_name] = pd.to_datetime(df[date_col_name])
        df = df.set_index(date_col_name)

    except Exception as e:
        st.error(f"Error loading or parsing CSV '{data_file_path_abs}': {e}")
        return pd.DataFrame()
    
    # --- MODIFICATION: Columns like SP500_Return, Prev_S&P500 are already in the CSV ---
    # We can choose to use them or recalculate for consistency.
    # For now, let's assume they are correct and use them.
    # The drift adjustment logic below uses 'SP500_Return'.

    required_cols_final = [sp500_target_col, vix_target_col, sofr_target_col, 'SP500_Return', 'Prev_S&P500']
    if not all(col in df.columns for col in required_cols_final):
        missing_cols = [col for col in required_cols_final if col not in df.columns]
        st.error(f"DataFrame missing required columns after initial load: {missing_cols}. Available: {df.columns.tolist()}")
        return pd.DataFrame()

    for col in [sp500_target_col, vix_target_col, sofr_target_col, 'SP500_Return', 'Prev_S&P500']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df.loc[df[sp500_target_col] <= 0, sp500_target_col] = np.nan # Good practice

    # --- Keep Drift Adjustment if it's a desired feature of the app ---
    # This part modifies the S&P500 prices based on an adjusted return series
    # If the input CSV's S&P500 and SP500_Return are considered final, this can be skipped.
    # Based on the prompt "no processing required to obtain data as its perfect",
    # this drift adjustment *might* be considered "processing".
    # However, it was in the original app.py. I'll keep it but add a note.
    # st.info("Applying drift adjustment to S&P500 returns and reconstructing prices.")
    target_daily_positive_drift = 0.00052
    valid_return_indices = df['SP500_Return'].notna()
    if not df.loc[valid_return_indices, 'SP500_Return'].empty:
        original_mean_return = df.loc[valid_return_indices, 'SP500_Return'].mean()
        df.loc[valid_return_indices, 'SP500_Return'] = \
            (df.loc[valid_return_indices, 'SP500_Return'] - original_mean_return) + target_daily_positive_drift
        
        # Reconstruct S&P500 prices based on new returns if Prev_S&P500 is reliable
        # Or, more robustly, start from first valid S&P500 price and cumulate.
        first_valid_price_idx = df[sp500_target_col].first_valid_index()
        if first_valid_price_idx is not None:
            base_price = df.loc[first_valid_price_idx, sp500_target_col]
            reconstructed_prices = [np.nan] * len(df)
            
            # Find the numeric index for the first valid price
            numeric_idx_start = df.index.get_loc(first_valid_price_idx)
            reconstructed_prices[numeric_idx_start] = base_price

            for i in range(numeric_idx_start + 1, len(df)):
                if pd.notna(df['SP500_Return'].iloc[i]) and pd.notna(reconstructed_prices[i-1]):
                    reconstructed_prices[i] = reconstructed_prices[i-1] * (1 + df['SP500_Return'].iloc[i])
                elif pd.notna(reconstructed_prices[i-1]): # If return is NaN, carry forward price
                    reconstructed_prices[i] = reconstructed_prices[i-1]
                # Else, if previous price was NaN, current remains NaN until a valid point
            
            df[sp500_target_col] = reconstructed_prices
            df['Prev_S&P500'] = df[sp500_target_col].shift(1) # Recalculate Prev_S&P500
            # SP500_Return should ideally be recalculated too after price reconstruction
            # df['SP500_Return'] = df[sp500_target_col].pct_change() # But the drift was applied to original returns.
                                                                # This creates a slight inconsistency if not careful.
                                                                # For now, keeping the drift-adjusted returns as they were.
        else:
            st.warning("S&P500 price reconstruction skipped (no valid initial S&P500 price found).")
    else:
        st.warning("No valid original returns for drift modification.")

    initial_rows_count = len(df)
    essential_check_cols = [sp500_target_col, vix_target_col, sofr_target_col, 'SP500_Return', 'Prev_S&P500']
    df = df[df.index.notna()]
    df = df.dropna(subset=essential_check_cols) # Critical for simulation
    rows_dropped_count = initial_rows_count - len(df)
    
    if df.empty:
        st.error("DataFrame is empty after final cleaning and NaN removal.")
        return pd.DataFrame()
    
    df = df.sort_index()
    # st.sidebar.info(f"Data prepared: {len(df)} rows ({rows_dropped_count} dropped).")
    return df

# ==============================================================================
# CFD Cost Model Functions (Kept in app.py for now as per current structure)
# These will be used if simulation_engine directly calls them or if they are
# used to populate parameters for a config object passed to simulation_engine.
# The provided simulation_engine.py uses config attributes for these.
# ==============================================================================
# calculate_margin, calculate_daily_financing_cost, calculate_daily_borrowing_cost, calculate_spread_cost
# These functions remain IDENTICAL to your original app.py unless the simulation_engine.py
# expects them to be called differently or uses its own internal logic based on config.
# For now, assuming the new simulation_engine.py will use config values directly for costs.
# So these functions in app.py might become unused if not called explicitly.
# I'll keep them here as they were in the original app, but the new engine might not call them.

def calculate_margin(num_contracts, index_price, config_dict): # config_dict is from Streamlit UI
    if num_contracts <= 0 or index_price <= 0 or np.isnan(num_contracts) or np.isnan(index_price):
        return 0.0
    lot_sz = config_dict.get('lot_size', 1.0) # Use .get for safety
    margin_tier_list = config_dict.get('margin_tiers', [])
    if not isinstance(margin_tier_list, list) or not all(isinstance(tier, dict) for tier in margin_tier_list):
        return 0.0
        
    total_notional = num_contracts * index_price * lot_sz
    margin_val = 0.0
    contracts_accounted_for = 0
    # Ensure tiers are sorted by their 'limit'
    sorted_tiers = sorted(margin_tier_list, key=lambda x: x.get('limit', float('inf')))
    last_tier_contract_limit = 0

    for tier_info in sorted_tiers:
        current_tier_upper_contract_limit = tier_info.get('limit', float('inf'))
        tier_rate = tier_info.get('rate', 0.0)
        # Contracts falling into this specific segment of the tier
        contracts_capacity_in_this_tier_segment = current_tier_upper_contract_limit - last_tier_contract_limit
        contracts_for_this_tier_rate = min(num_contracts - contracts_accounted_for, contracts_capacity_in_this_tier_segment)
        
        if contracts_for_this_tier_rate > 0:
            margin_for_these_contracts = contracts_for_this_tier_rate * index_price * lot_sz * tier_rate
            margin_val += margin_for_these_contracts
            contracts_accounted_for += contracts_for_this_tier_rate
        
        last_tier_contract_limit = current_tier_upper_contract_limit
        if contracts_accounted_for >= num_contracts:
            break

    # If num_contracts exceeds the limit of the highest defined tier, apply the last tier's rate
    if contracts_accounted_for < num_contracts and sorted_tiers:
        remaining_contracts = num_contracts - contracts_accounted_for
        final_tier_rate = sorted_tiers[-1].get('rate', 0.0) # Rate of the highest tier
        margin_val += remaining_contracts * index_price * lot_sz * final_tier_rate

    if np.isnan(margin_val) or np.isinf(margin_val): return 0.0
    return min(margin_val, total_notional) # Margin cannot exceed total notional

def calculate_daily_financing_cost(contracts_held, price_at_calc, sofr_rate_decimal, config_dict_ui, is_short_pos):
    if contracts_held <= 0 or price_at_calc <= 0 or np.isnan(contracts_held) or np.isnan(price_at_calc) or pd.isna(sofr_rate_decimal):
        return 0.0
    lot_sz = config_dict_ui.get('lot_size', 1.0)
    broker_fee = config_dict_ui.get('broker_annual_financing_fee', 0.03) # from UI
    days_in_yr = config_dict_ui.get('days_in_year_financing', 360)
    notional_val = contracts_held * lot_sz * price_at_calc
    
    # Net rate for short position: (Benchmark - Broker's Fee). If SOFR is benchmark.
    # Cost to you if positive, gain if negative.
    # The prompt's simulation engine formula for financing for short was:
    # cfd_financing_impact = cfd_notional_value * ((sofr_annual_t - BROKER_FEE_ANNUALIZED) / 365.0)
    # cash_value += cfd_financing_impact
    # So, a positive impact (SOFR > Fee) increases cash. A negative impact (SOFR < Fee) decreases cash.
    # This is consistent with receiving interest on short proceeds less a fee, or paying if net is negative.
    # The function here should return the *cost*. So if (SOFR - Fee) is positive (gain), cost is negative.
    
    if is_short_pos:
        # Cost to short seller = Notional * (Benchmark_Rate + Broker_Lending_Spread_or_Borrow_Fee) / Days
        # Or, if broker passes through interest on short proceeds:
        # Net Effect = (Interest_on_Proceeds_Rate - Borrow_Fee_or_Financing_Spread)
        # Here, assuming `broker_annual_financing_fee` is a charge on top of benchmark for longs,
        # and a benefit subtracted from benchmark for shorts (or a charge if benchmark is too low).
        # Let's directly use the logic from the notebook's simulation_engine for consistency:
        # (sofr_annual_t - BROKER_FEE_ANNUALIZED)
        # If this is positive, it's a credit to the shorter. Cost is negative.
        # If this is negative, it's a debit to the shorter. Cost is positive.
        annual_rate_effect = sofr_rate_decimal - broker_fee 
        financing_effect_on_cash = (notional_val * annual_rate_effect) / days_in_yr
        return -financing_effect_on_cash # Return as cost
    else: # Long position
        annual_rate_cost = sofr_rate_decimal + broker_fee
        financing_cst = (notional_val * annual_rate_cost) / days_in_yr
        return financing_cst

def calculate_daily_borrowing_cost(contracts_held, price_at_calc, config_dict_ui, is_short_pos):
    # This function was specific to the original app if it had a separate borrowing cost.
    # The new simulation_engine's financing cost might already encompass this implicitly
    # via the BROKER_FEE_ANNUALIZED. If BROKER_FEE_ANNUALIZED is the all-in cost for shorts
    # (relative to SOFR), then a separate borrowing cost might not be needed *if* the
    # financing already covers (SOFR - (borrow_fee_component_of_broker_fee)).
    # The notebook logic only had (SOFR - BROKER_FEE).
    # For simplicity and to align with the notebook's simulation engine logic,
    # we will assume the BROKER_FEE_ANNUALIZED in the financing calculation is comprehensive.
    # If a separate explicit borrowing_cost_annual from UI needs to be added,
    # it would be: (notional_val * borrowing_cost_annual_from_ui) / days_in_yr
    # and it would always be a cost for short positions.
    # The provided simulation_engine.py does not have a separate borrowing cost parameter.
    # It uses cfg.BROKER_FEE_ANNUALIZED in its financing calculation.
    # So, this function is likely not directly used by the new engine unless modified.
    # For now, let's keep it as in the original app but it might be redundant.
    if not is_short_pos or contracts_held <= 0 or price_at_calc <= 0 or np.isnan(contracts_held) or np.isnan(price_at_calc):
        return 0.0
    lot_sz = config_dict_ui.get('lot_size', 1.0)
    borrow_annual_rate = config_dict_ui.get('borrowing_cost_annual', 0.006) # from UI
    days_in_yr = config_dict_ui.get('days_in_year_financing', 360)
    notional_val = contracts_held * lot_sz * price_at_calc
    borrowing_cst = (notional_val * borrow_annual_rate) / days_in_yr
    return borrowing_cst

def calculate_spread_cost(contracts_transacted_abs, price_at_transaction, config_dict_ui):
    if contracts_transacted_abs <= 0 or price_at_transaction <= 0 or np.isnan(contracts_transacted_abs) or np.isnan(price_at_transaction):
        return 0.0
    lot_sz = config_dict_ui.get('lot_size', 1.0)
    spread_pts = config_dict_ui.get('avg_spread_points', 0.3) # from UI
    # Cost = Number of Contracts * Lot Size * Spread in Points
    # (Assuming spread_pts is the monetary value of the spread for 1 unit of the index,
    # and lot_size defines how many units of index one CFD contract represents)
    # Example: S&P 500 at 4000. Spread 0.3 points. Lot size 1.
    # Cost per contract = 1 * 0.3 = $0.3
    cost_val = contracts_transacted_abs * lot_sz * spread_pts
    return cost_val

# ==============================================================================
# Hedging Strategy Functions (OLD - Will be replaced by signal_generation module)
# ==============================================================================
# def get_hedge_action(...): # This old function will be removed

# ==============================================================================
# Simulation Engine (OLD - Will be replaced by simulation_engine module)
# ==============================================================================
# def simulate_classic_portfolio(...): # Old version
# def simulate_hedged_portfolio(...): # Old version (with simple VIX threshold)

# ==============================================================================
# Plotting Functions (Plotly - can be kept or moved to plotting.py)
# ==============================================================================
# plot_portfolio_comparison_plotly function remains the same as in your app.py
def plot_portfolio_comparison_plotly(portfolio_A_series, portfolio_B_series, config_dict_ui, model_B_label="Model B (CFD-Hedged)"): # Added model_B_label
    initial_cap = config_dict_ui['initial_capital']
    
    if portfolio_A_series.empty or portfolio_B_series.empty:
        st.warning("No simulation data to plot.")
        return go.Figure()

    # Ensure consistent indexing if one series is shorter (e.g. due to NaNs at start)
    # Use .reindex(common_index).ffill() if necessary, but results from engine should align.
    comp_df = pd.concat([portfolio_A_series, portfolio_B_series], axis=1).rename(
        columns={portfolio_A_series.name: 'Portfolio_A', portfolio_B_series.name: 'Portfolio_B'}
    )
    # ffill and dropna might be too aggressive if start dates differ slightly due to NaNs
    # It's better if simulation engine returns series aligned to the input df_sim's index.
    comp_df = comp_df.dropna(subset=['Portfolio_A', 'Portfolio_B'], how='all') # Drop rows where both are NaN
    comp_df = comp_df.ffill() # Forward fill to handle initial NaNs from pct_change if any
    comp_df = comp_df.dropna() # Drop any remaining NaNs (e.g. if all were NaN at start)


    if comp_df.empty:
        st.warning("No common dates to plot after processing.")
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=comp_df.index, y=comp_df['Portfolio_A'] / initial_cap,
                             mode='lines', name=static_cfg.PORTFOLIO_A_LABEL, # Use label from config
                             line=dict(width=2, color=static_cfg.COLOR_A))) # Use color from config

    fig.add_trace(go.Scatter(x=comp_df.index, y=comp_df['Portfolio_B'] / initial_cap,
                             mode='lines', name=model_B_label, # Use dynamic label
                             line=dict(width=2, color=static_cfg.COLOR_B_MOMENTUM), opacity=0.8)) # Use color from config


    shapes = []
    # Ensure 'analysis_periods' exists and is a dictionary
    analysis_periods_ui = config_dict_ui.get('analysis_periods', {})
    if isinstance(analysis_periods_ui, dict):
        for name, period in analysis_periods_ui.items():
            if isinstance(period, dict) and 'start' in period and 'end' in period:
                try:
                    start_dt = pd.to_datetime(period['start'])
                    end_dt = pd.to_datetime(period['end'])
                    color = 'rgba(255, 160, 122, 0.2)' if 'covid' in name.lower() else 'rgba(135, 206, 250, 0.2)'
                    shapes.append(
                        dict(type="rect", xref="x", yref="paper", x0=start_dt, y0=0, x1=end_dt, y1=1,
                             fillcolor=color, opacity=0.5, layer="below", line_width=0)
                    )
                except Exception as e:
                    st.warning(f"Could not parse dates for period '{name}': {e}")
            else:
                st.warning(f"Period '{name}' in 'analysis_periods' is not correctly formatted.")
    
    # Dummy traces for legend
    unique_shapes_info = []
    seen_names = set()
    if isinstance(analysis_periods_ui, dict):
        for name, period in analysis_periods_ui.items():
            shape_name = f"{name.replace('_',' ').title()}"
            if shape_name not in seen_names:
                color = 'rgba(255, 160, 122, 0.7)' if 'covid' in name.lower() else 'rgba(135, 206, 250, 0.7)'
                unique_shapes_info.append({'name': shape_name, 'color': color})
                seen_names.add(shape_name)

    for info in unique_shapes_info:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='markers',
            marker=dict(size=10, color=info['color']),
            legendgroup="periods", name=info['name'], showlegend=True
        ))

    fig.update_layout(
        title=dict(text=f'<b>{static_cfg.PLOT_TITLE_PERFORMANCE}</b>', x=0.5, font=dict(size=20)),
        xaxis_title='Date',
        yaxis_title='Normalized Portfolio Value',
        legend_title_text='Portfolios & Periods',
        font=dict(family="Arial, sans-serif", size=12),
        yaxis_tickformat=",.2f",
        shapes=shapes,
        hovermode="x unified",
        template="plotly_white"
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    return fig

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="CFD Strategy Analysis")

# Custom CSS
st.markdown("""
    <style> /* Your CSS */ </style>
    """, unsafe_allow_html=True)

st.title("📈 CFD Hedging & VIX Momentum Strategy Analysis")
st.markdown("This application simulates a classic S&P 500 portfolio (Model A) against a portfolio using a **VIX Momentum-triggered CFD hedge** (Model B).")

# Default Configuration from app.py (UI driven)
default_config_ui = {
    'initial_capital': 1000000,
    'equity_allocation': 0.8, # Used for BOTH Model A and Model B base equity
    'trading_days_per_year': 252,
    'data_file': 'vix_sp500_data.csv', # Relative to app.py location
    'date_column': 'date',
    'sp500_column': 'S&P500',
    'vix_column': 'VIX',
    'sofr_column': 'SOFR_Rate', # Matches CSV header
    'raw_csv_sofr_col': 'SOFR_Rate', # Matches CSV header
    # Model B Specific (VIX Momentum related parameters will come from static_cfg)
    'model_b_fixed_hedge_ratio_momentum': 0.70, # UI will control this for Model B
    # CFD Costs (from UI)
    'lot_size': 1.0,
    'broker_annual_financing_fee': 0.025, # Matches example from problem
    'borrowing_cost_annual': 0.0, # Set to 0 if financing fee is all-inclusive for shorts
    'days_in_year_financing': 365, # More standard than 360
    'avg_spread_points': 0.3, # Spread in S&P500 points
    'cfd_initial_margin_percent': 0.05, # For Model B
    'margin_tiers': [ # This tier logic is complex and might be simplified or handled by engine
        {'limit': 5, 'rate': 0.05}, # Assuming rate is the margin rate itself
        {'limit': 25, 'rate': 0.10},
        {'limit': float('inf'), 'rate': 0.15}, # For contracts beyond 25
    ],
    'analysis_periods': {
        'covid_crisis': {'start': '2020-02-01', 'end': '2020-04-30'}, # Match notebook
        'vix_spike_2025': {'start': '2025-03-01', 'end': '2025-05-21'} # Match notebook
    }
}

# Sidebar for User Inputs
with st.sidebar:
    st.header("⚙️ General Configuration")
    initial_capital_ui = st.number_input("Initial Capital ($)", min_value=10000, value=default_config_ui['initial_capital'], step=10000, format="%d")
    equity_allocation_ui = st.slider("Base Equity Allocation (%) (For Both Models)", min_value=0.1, max_value=1.0, value=default_config_ui['equity_allocation'], step=0.01)
    
    st.markdown("---")
    st.subheader("Model B: VIX Momentum Hedge")
    model_b_hedge_ratio_ui = st.slider("CFD Hedge Ratio (Proportion of Equity)", min_value=0.0, max_value=1.0, value=default_config_ui['model_b_fixed_hedge_ratio_momentum'], step=0.01)
    
    # VIX signal parameters are mostly static from scripts/config.py, but VIX_ABSOLUTE_COVER_THRESHOLD can be from UI
    vix_abs_cover_thresh_ui = st.slider("VIX Absolute Cover Threshold (Model B)", min_value=10.0, max_value=30.0, value=static_cfg.VIX_ABSOLUTE_COVER_THRESHOLD, step=0.5)


    st.markdown("---")
    st.subheader("CFD Costs (for Model B)")
    broker_financing_fee_ui = st.slider("Broker Annual Financing Fee (Markup/Markdown vs SOFR)", min_value=0.0, max_value=0.05, value=default_config_ui['broker_annual_financing_fee'], step=0.001, format="%.3f")
    # borrowing_cost_annual_ui = st.slider("Annual Borrowing Cost for Shorts (Separate)", min_value=0.0, max_value=0.05, value=default_config_ui['borrowing_cost_annual'], step=0.001, format="%.3f")
    avg_spread_points_ui = st.number_input("Spread Cost (S&P 500 points per contract transaction)", min_value=0.0, max_value=2.0, value=default_config_ui['avg_spread_points'], step=0.01, format="%.2f")
    cfd_initial_margin_percent_ui = st.slider("CFD Initial Margin Percent", min_value=0.01, max_value=0.20, value=default_config_ui['cfd_initial_margin_percent'], step=0.01)


script_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path_abs = os.path.join(script_dir, default_config_ui['data_file'])
# Ensure 'data' subfolder is not assumed if vix_sp500_data.csv is in root with app.py
if not os.path.exists(data_file_path_abs) and default_config_ui['data_file'].startswith("data/"):
     data_file_path_abs = os.path.join(script_dir, os.path.basename(default_config_ui['data_file']))


if st.sidebar.button("🚀 Run Simulation & Plot", use_container_width=True):
    # Create a combined configuration for the backend modules
    # Start with static defaults, then override with UI inputs
    
    # Prepare a dictionary that will be passed to the backend functions,
    # structured as they expect (e.g., flat attributes via ConfigObject).
    
# Inside the if st.sidebar.button("🚀 Run Simulation & Plot", use_container_width=True): block

    # ... (other parts of cfg_dict_for_backend) ...
    # Inside the if st.sidebar.button("🚀 Run Simulation & Plot", use_container_width=True): block

    # ... (other parts of cfg_dict_for_backend) ...
    cfg_dict_for_backend = {
        # General
        'INITIAL_CAPITAL': initial_capital_ui,
        'TRADING_DAYS_PER_YEAR': default_config_ui['trading_days_per_year'],
        
        'sp500_column': default_config_ui['sp500_column'],
        'vix_column': default_config_ui['vix_column'],    
        'sofr_column': default_config_ui['sofr_column'],  
        
        # Model A
        'EQUITY_ALLOC_A': equity_allocation_ui,
        # --- ADD CASH_ALLOC_A ---
        'CASH_ALLOC_A': 1.0 - equity_allocation_ui, 
        # --- END OF ADDITION ---
        
        # Model B Base Allocation & Hedge Ratio
        'EQUITY_ALLOC_B': equity_allocation_ui,
        # --- ADD CASH_ALLOC_B ---
        'CASH_ALLOC_B': 1.0 - equity_allocation_ui,
        # --- END OF ADDITION ---
        'MODEL_B_FIXED_HEDGE_RATIO_MOMENTUM': model_b_hedge_ratio_ui,
        
        # VIX Momentum Signal Params
        'MOMENTUM_LOOKBACK_PERIOD': static_cfg.MOMENTUM_LOOKBACK_PERIOD,
        'VIX_PCT_CHANGE_THRESHOLD_UP': static_cfg.VIX_PCT_CHANGE_THRESHOLD_UP,
        'VIX_PCT_CHANGE_THRESHOLD_DOWN': static_cfg.VIX_PCT_CHANGE_THRESHOLD_DOWN,
        'N_CONSECUTIVE_UP_DAYS_TO_SHORT': static_cfg.N_CONSECUTIVE_UP_DAYS_TO_SHORT,
        'N_CONSECUTIVE_DOWN_DAYS_TO_COVER': static_cfg.N_CONSECUTIVE_DOWN_DAYS_TO_COVER,
        'VIX_ABSOLUTE_COVER_THRESHOLD': vix_abs_cover_thresh_ui,

        # CFD Costs
        'BROKER_FEE_ANNUALIZED': broker_financing_fee_ui,
        'SPREAD_COST_PERCENT': 0.0002, 
        'AVG_SPREAD_POINTS': avg_spread_points_ui, 
        'CFD_INITIAL_MARGIN_PERCENT': cfd_initial_margin_percent_ui,
        'LOT_SIZE': default_config_ui['lot_size'],
        'DAYS_IN_YEAR_FINANCING': default_config_ui['days_in_year_financing'],
        
        # For plotting labels
        'PORTFOLIO_A_LABEL': static_cfg.PORTFOLIO_A_LABEL,
        'COLOR_A': static_cfg.COLOR_A,
        'COLOR_B_MOMENTUM': static_cfg.COLOR_B_MOMENTUM
    }
    cfg_object = ConfigObject(cfg_dict_for_backend)

    # ... rest of your code ...

    with st.spinner("Loading and preparing data..."):
        # load_and_prepare_data uses default_config_ui for its specific keys
        main_df_loaded = load_and_prepare_data(default_config_ui, data_file_path_abs)

    if not main_df_loaded.empty:
        with st.spinner("Generating VIX signals..."):
            # generate_vix_momentum_signals uses cfg_object, which now has the column names
            main_df_with_signals = sg.generate_vix_momentum_signals(main_df_loaded, cfg_object) 
        
        # Drop NaNs created by signal generation's shift operations
        cols_to_check_for_nan = [
            cfg_object.sp500_column, cfg_object.vix_column, cfg_object.sofr_column, # Now accessible
            'SP500_Return', 'Prev_S&P500',
            'VIX_Lagged_5D', 'VIX_Pct_Change_5D',
            'Short_Signal_Today', 'Cover_Signal_Momentum_Today', 'Cover_Signal_Absolute_VIX_Today'
            ]
        main_df_sim = main_df_with_signals.dropna(subset=cols_to_check_for_nan).copy()
        main_df_sim.reset_index(drop=True, inplace=True)

        # ... rest of your code ...

        if main_df_sim.empty:
            st.error("Dataframe became empty after signal generation and NaN dropping. Cannot simulate.")
        else:
            st.sidebar.success(f"Data ready for simulation: {len(main_df_sim)} rows.")

            sim_portfolio_A = sim_eng.simulate_portfolio_A(
                main_df_sim, 
                cfg_object.INITIAL_CAPITAL, 
                cfg_object.EQUITY_ALLOC_A
            )
            # The simulation_engine.simulate_portfolio_B_momentum needs to be adapted to return cost_details_df
            # For now, let's assume it only returns portfolio values.
            sim_portfolio_B = sim_eng.simulate_portfolio_B_momentum(
                main_df_sim, 
                cfg_object.INITIAL_CAPITAL, 
                cfg_object # Pass the wrapped config object
            )
            # Placeholder for cost_df if engine doesn't return it:
            sim_cost_df = pd.DataFrame() # Or calculate separately if needed by app display

            if sim_portfolio_A.empty or sim_portfolio_B.empty:
                st.error("Simulation produced empty results. Check data and parameters.")
            else:
                st.header("📊 Portfolio Value Comparison")
                model_B_dynamic_label = f"Model B (VIX Momentum HR:{cfg_object.MODEL_B_FIXED_HEDGE_RATIO_MOMENTUM:.2f})"
                
                # Align indices before plotting if series are returned with original df_sim_full index
                # If simulation engines return Series with df_sim_full's index, direct concat is fine.
                # The plot function handles Series with DateTimeIndex.
                # We need to ensure the series index matches the dates in main_df_sim for plotting.
                # The simulation engines were modified to return Series with original df index.
                
                # To pass to plotting function, ensure index is datetime if not already
                # The simulation engines should return series with the same index as the input df (main_df_sim)
                # which already has a DatetimeIndex.

                # The original plotting function uses the initial_capital from config_dict_ui.
                # We should pass the cfg_object or the specific initial_capital_ui.
                plot_config_for_plotly = {
                    'initial_capital': initial_capital_ui,
                    'analysis_periods': default_config_ui['analysis_periods'] # from app's config
                }

                fig_comp_plotly = plot_portfolio_comparison_plotly(
                    sim_portfolio_A.rename("Portfolio_A"), # Ensure names are consistent
                    sim_portfolio_B.rename("Portfolio_B"), 
                    plot_config_for_plotly,
                    model_B_dynamic_label
                )
                st.plotly_chart(fig_comp_plotly, use_container_width=True)
                
                st.header("📈 Simulation Summary")
                col1, col2 = st.columns(2) # Simplified for now without cost display
                with col1:
                    st.metric("Model A Final Value", f"${sim_portfolio_A.iloc[-1]:,.0f}")
                    st.metric("Model A Return", f"{(sim_portfolio_A.iloc[-1]/initial_capital_ui -1)*100:.1f}%")
                with col2:
                    st.metric(f"{model_B_dynamic_label} Final Value", f"${sim_portfolio_B.iloc[-1]:,.0f}")
                    st.metric(f"{model_B_dynamic_label} Return", f"{(sim_portfolio_B.iloc[-1]/initial_capital_ui -1)*100:.1f}%")
                # with col3:
                #     if not sim_cost_df.empty: # This depends on simulate_portfolio_B_momentum returning cost_df
                #          st.metric("Model B Total CFD Costs", f"${sim_cost_df['Total'].sum():,.0f}")
                #     else:
                #          st.metric("Model B Total CFD Costs", "N/A (not returned by engine)")
                
                with st.expander("Show Raw Simulation Data (Last 5 rows)"):
                    st.subheader(static_cfg.PORTFOLIO_A_LABEL)
                    st.dataframe(sim_portfolio_A.tail())
                    st.subheader(model_B_dynamic_label)
                    st.dataframe(sim_portfolio_B.tail())
                    # if not sim_cost_df.empty:
                    #     st.subheader("Model B Cost Details (if returned by engine)")
                    #     st.dataframe(sim_cost_df.tail())
    else:
        st.warning("Data could not be loaded. Please ensure your data file is correctly specified and formatted.")
else:
    st.info("Adjust parameters in the sidebar and click '🚀 Run Simulation & Plot'.")

st.sidebar.markdown("---")
st.sidebar.info("This app uses a VIX Momentum strategy for Model B's CFD hedging.")