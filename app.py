import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # Still needed for other plots if you add them
import matplotlib.dates as mdates
import plotly.graph_objects as go # For Plotly
from plotly.subplots import make_subplots
import os
import time


# [Keep all your data loading, simulation, and cost model functions from the previous version here]
# For brevity, I'm omitting them here, but they are ESSENTIAL.
# Make sure to copy them from the previous app.py you have.
# I'll only show the modified plot_portfolio_comparison function.

# ==============================================================================
# Utility Functions (Simplified - no logging to console directly in Streamlit)
# ==============================================================================
def log_and_time_stub(func): # Stub for the decorator if not fully implemented
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return result
    return wrapper

# ==============================================================================
# Data Loading and Preprocessing
# ==============================================================================
@st.cache_data # Cache the data loading
def load_and_prepare_data(config_dict, data_file_path_abs):
    date_col_name = config_dict['date_column']
    sp500_target_col = config_dict['sp500_column']
    vix_target_col = config_dict['vix_column']
    sofr_target_col = config_dict['sofr_column']

    if not os.path.exists(data_file_path_abs):
        st.error(f"Data file not found: {data_file_path_abs}")
        return pd.DataFrame()

    df = pd.read_csv(data_file_path_abs, index_col=date_col_name, parse_dates=True)
    data_source_info = "CSV file"
    # st.info(f"Data loaded from: {data_source_info}") # Less verbose for cleaner UI

    rename_map_csv_load = {}
    if config_dict.get('raw_csv_sp500_col') and config_dict['raw_csv_sp500_col'] in df.columns and config_dict['raw_csv_sp500_col'] != sp500_target_col:
        rename_map_csv_load[config_dict['raw_csv_sp500_col']] = sp500_target_col
    if config_dict.get('raw_csv_vix_col') and config_dict['raw_csv_vix_col'] in df.columns and config_dict['raw_csv_vix_col'] != vix_target_col:
        rename_map_csv_load[config_dict['raw_csv_vix_col']] = vix_target_col
    if config_dict.get('raw_csv_sofr_col') and config_dict['raw_csv_sofr_col'] in df.columns and config_dict['raw_csv_sofr_col'] != sofr_target_col:
        rename_map_csv_load[config_dict['raw_csv_sofr_col']] = sofr_target_col
    
    if rename_map_csv_load:
        df = df.rename(columns=rename_map_csv_load)

    required_cols_final = [sp500_target_col, vix_target_col, sofr_target_col]
    if not all(col in df.columns for col in required_cols_final):
        missing_cols = [col for col in required_cols_final if col not in df.columns]
        st.error(f"DataFrame missing required columns: {missing_cols}. Available: {df.columns.tolist()}")
        return pd.DataFrame()

    for col in required_cols_final:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.loc[df[sp500_target_col] <= 0, sp500_target_col] = np.nan

    df['SP500_Return'] = df[sp500_target_col].pct_change()
    target_daily_positive_drift = 0.00052 # Approx 14% annualized (as per notebook)
    
    valid_return_indices = df['SP500_Return'].notna()
    if not df.loc[valid_return_indices, 'SP500_Return'].empty:
        original_mean_return = df.loc[valid_return_indices, 'SP500_Return'].mean()
        df.loc[valid_return_indices, 'SP500_Return'] = \
            (df.loc[valid_return_indices, 'SP500_Return'] - original_mean_return) + target_daily_positive_drift
        
        indices_to_iterate = df.index
        if len(indices_to_iterate) > 1 and pd.notna(df[sp500_target_col].iloc[0]):
            base_price = df[sp500_target_col].iloc[0]
            reconstructed_prices_list = [base_price]
            for i in range(1, len(df)):
                current_return_val = df['SP500_Return'].iloc[i]
                if pd.notna(current_return_val):
                    new_price_val = reconstructed_prices_list[-1] * (1 + current_return_val)
                    reconstructed_prices_list.append(new_price_val)
                else:
                    reconstructed_prices_list.append(reconstructed_prices_list[-1])
            df[sp500_target_col] = reconstructed_prices_list
        else:
            st.warning("S&P500 price reconstruction skipped (empty df or invalid initial price).")
    else:
        st.warning("No valid original returns for drift modification. S&P500 prices not altered from CSV.")

    df['Prev_S&P500'] = df[sp500_target_col].shift(1)
    initial_rows_count = len(df)
    essential_check_cols = [sp500_target_col, vix_target_col, sofr_target_col, 'SP500_Return', 'Prev_S&P500']
    df = df[df.index.notna()]
    df = df.dropna(subset=essential_check_cols)
    rows_dropped_count = initial_rows_count - len(df)
    if df.empty:
        st.error("DataFrame is empty after final cleaning and NaN removal.")
        return pd.DataFrame()
    
    # st.sidebar.success(f"Data prepared: {len(df)} rows ({rows_dropped_count} dropped).") # Moved to sidebar
    df.index.name = date_col_name
    df = df.sort_index()
    return df

# ==============================================================================
# CFD Cost Model Functions
# ==============================================================================
def calculate_margin(num_contracts, index_price, config_dict):
    if num_contracts <= 0 or index_price <= 0 or np.isnan(num_contracts) or np.isnan(index_price):
        return 0.0
    lot_sz = config_dict['lot_size']
    margin_tier_list = config_dict.get('margin_tiers', [])
    if not isinstance(margin_tier_list, list) or not all(isinstance(tier, dict) for tier in margin_tier_list):
        return 0.0
        
    total_notional = num_contracts * index_price * lot_sz
    margin_val = 0.0
    contracts_accounted_for = 0
    sorted_tiers = sorted(margin_tier_list, key=lambda x: x.get('limit', float('inf')))
    last_tier_contract_limit = 0
    for tier_info in sorted_tiers:
        current_tier_upper_contract_limit = tier_info.get('limit', float('inf'))
        tier_rate = tier_info.get('rate', 0.0)
        contracts_capacity_in_this_tier_segment = current_tier_upper_contract_limit - last_tier_contract_limit
        contracts_for_this_tier_rate = min(num_contracts - contracts_accounted_for, contracts_capacity_in_this_tier_segment)
        
        if contracts_for_this_tier_rate > 0:
            margin_for_these_contracts = contracts_for_this_tier_rate * index_price * lot_sz * tier_rate
            margin_val += margin_for_these_contracts
            contracts_accounted_for += contracts_for_this_tier_rate
        
        last_tier_contract_limit = current_tier_upper_contract_limit
        if contracts_accounted_for >= num_contracts:
            break

    if contracts_accounted_for < num_contracts and sorted_tiers:
        remaining_contracts = num_contracts - contracts_accounted_for
        final_tier_rate = sorted_tiers[-1].get('rate', 0.0)
        margin_val += remaining_contracts * index_price * lot_sz * final_tier_rate

    if np.isnan(margin_val) or np.isinf(margin_val):
        return 0.0
    return min(margin_val, total_notional)

def calculate_daily_financing_cost(contracts_held, price_at_calc, sofr_rate_decimal, config_dict, is_short_pos):
    if contracts_held <= 0 or price_at_calc <= 0 or np.isnan(contracts_held) or np.isnan(price_at_calc) or pd.isna(sofr_rate_decimal):
        return 0.0
    lot_sz = config_dict['lot_size']
    broker_fee = config_dict['broker_annual_financing_fee']
    days_in_yr = config_dict['days_in_year_financing']
    notional_val = contracts_held * lot_sz * price_at_calc
    if is_short_pos:
        annual_rate_eff = sofr_rate_decimal - broker_fee
        financing_adj = (notional_val * annual_rate_eff) / days_in_yr
        return -financing_adj
    else:
        annual_rate_eff = sofr_rate_decimal + broker_fee
        financing_cst = (notional_val * annual_rate_eff) / days_in_yr
        return financing_cst

def calculate_daily_borrowing_cost(contracts_held, price_at_calc, config_dict, is_short_pos):
    if not is_short_pos or contracts_held <= 0 or price_at_calc <= 0 or np.isnan(contracts_held) or np.isnan(price_at_calc):
        return 0.0
    lot_sz = config_dict['lot_size']
    borrow_annual_rate = config_dict['borrowing_cost_annual']
    days_in_yr = config_dict['days_in_year_financing']
    notional_val = contracts_held * lot_sz * price_at_calc
    borrowing_cst = (notional_val * borrow_annual_rate) / days_in_yr
    return borrowing_cst

def calculate_spread_cost(contracts_transacted_abs, price_at_transaction, config_dict):
    if contracts_transacted_abs <= 0 or price_at_transaction <= 0 or np.isnan(contracts_transacted_abs) or np.isnan(price_at_transaction):
        return 0.0
    lot_sz = config_dict['lot_size']
    spread_pts = config_dict['avg_spread_points']
    cost_val = contracts_transacted_abs * lot_sz * spread_pts
    return cost_val

# ==============================================================================
# Hedging Strategy Functions
# ==============================================================================
def get_hedge_action(current_vix_level, current_equity_val, current_idx_price, config_dict):
    vix_trig_thresh = config_dict['hedging_strategy']['vix_threshold']
    hedge_ratio_val = config_dict['hedging_strategy']['hedge_ratio']
    cfd_lot_size = config_dict['lot_size']
    target_short_contracts_num = 0.0
    if pd.isna(current_vix_level) or pd.isna(current_equity_val) or pd.isna(current_idx_price):
        return 0.0

    if current_vix_level > vix_trig_thresh:
        if current_equity_val > 0 and current_idx_price > 0 and cfd_lot_size > 0:
            notional_to_be_hedged = current_equity_val * hedge_ratio_val
            value_of_one_cfd_unit = current_idx_price * cfd_lot_size
            if value_of_one_cfd_unit > 0:
                 target_short_contracts_num = notional_to_be_hedged / value_of_one_cfd_unit
                 if np.isnan(target_short_contracts_num) or np.isinf(target_short_contracts_num): target_short_contracts_num = 0.0
    return target_short_contracts_num

# ==============================================================================
# Simulation Engine
# ==============================================================================
@st.cache_data(show_spinner="Simulating Classic Portfolio (Model A)...")
def simulate_classic_portfolio(data_df, config_dict):
    initial_cap = config_dict['initial_capital']
    equity_alloc_pct = config_dict['equity_allocation']
    equity_A_val = initial_cap * equity_alloc_pct
    cash_A_val = initial_cap * (1.0 - equity_alloc_pct)
    portfolio_A_history = []
    dates_A_history = []

    if 'SP500_Return' not in data_df.columns:
        st.error("'SP500_Return' column not found in data_df for Model A.")
        return pd.Series(name="Portfolio_A", dtype=float)
        
    sim_data_A = data_df[['SP500_Return']].copy()
    current_equity_A = equity_A_val

    if not sim_data_A.empty:
        start_date_A = sim_data_A.index[0]
        portfolio_A_history.append(initial_cap)
        dates_A_history.append(start_date_A - pd.Timedelta(days=1) if isinstance(start_date_A, pd.Timestamp) else pd.to_datetime("1900-01-01"))

    for date_idx, row_data in sim_data_A.iterrows():
        sp_ret = row_data['SP500_Return']
        if pd.notna(sp_ret):
            current_equity_A *= (1 + sp_ret)
        
        portfolio_val_A = current_equity_A + cash_A_val
        if pd.isna(portfolio_val_A):
            portfolio_val_A = portfolio_A_history[-1] if portfolio_A_history else initial_cap
        
        portfolio_A_history.append(portfolio_val_A)
        dates_A_history.append(date_idx)

    if not dates_A_history: return pd.Series(name="Portfolio_A", dtype=float)
    portfolio_A_series = pd.Series(portfolio_A_history, index=pd.DatetimeIndex(dates_A_history), name="Portfolio_A")
    return portfolio_A_series

@st.cache_data(show_spinner="Simulating Hedged Portfolio (Model B)...")
def simulate_hedged_portfolio(data_df, config_dict):
    initial_cap = config_dict['initial_capital']
    equity_alloc_pct = config_dict['equity_allocation']
    sp500_col_name = config_dict['sp500_column']
    vix_col_name = config_dict['vix_column']
    sofr_col_name = config_dict['sofr_column']

    current_equity_B = initial_cap * equity_alloc_pct
    current_cash_B = initial_cap * (1.0 - equity_alloc_pct)
    
    portfolio_B_history = []
    dates_B_history = []
    cost_details_history = []
    active_short_contracts = 0.0
    current_margin_held = 0.0

    required_sim_cols = ['SP500_Return', sp500_col_name, 'Prev_S&P500', vix_col_name, sofr_col_name]
    if not all(col in data_df.columns for col in required_sim_cols):
        missing = [col for col in required_sim_cols if col not in data_df.columns]
        st.error(f"Missing required columns for Model B simulation: {missing}")
        return pd.Series(name="Portfolio_B", dtype=float), pd.DataFrame()

    if not data_df.empty:
        start_date_B = data_df.index[0]
        portfolio_B_history.append(initial_cap)
        dates_B_history.append(start_date_B - pd.Timedelta(days=1) if isinstance(start_date_B, pd.Timestamp) else pd.to_datetime("1900-01-01"))
        cost_details_history.append({
            'date': start_date_B - pd.Timedelta(days=1) if isinstance(start_date_B, pd.Timestamp) else pd.to_datetime("1900-01-01"),
            'Financing': 0.0, 'Borrowing': 0.0, 'Spread': 0.0, 'Total': 0.0,
            'Contracts': 0.0, 'Margin': 0.0, 'CFD_PnL': 0.0
        })
    
    sim_data_B = data_df[required_sim_cols].copy()

    for date_idx, row_data in sim_data_B.iterrows():
        sp_ret = row_data['SP500_Return']
        if pd.notna(sp_ret):
            current_equity_B *= (1 + sp_ret)
        
        daily_cfd_pnl_val = 0.0
        financing_cost_val = 0.0
        borrowing_cost_val = 0.0
        spread_cost_val = 0.0

        prev_close_price = row_data['Prev_S&P500']
        current_close_price = row_data[sp500_col_name]
        sofr_rate_today = row_data[sofr_col_name]
        vix_level_today = row_data[vix_col_name]

        if active_short_contracts > 0 and \
           pd.notna(prev_close_price) and prev_close_price > 0 and \
           pd.notna(current_close_price) and current_close_price > 0 and \
           pd.notna(sofr_rate_today):
            
            daily_cfd_pnl_val = active_short_contracts * config_dict['lot_size'] * (prev_close_price - current_close_price)
            financing_cost_val = calculate_daily_financing_cost(active_short_contracts, prev_close_price, sofr_rate_today, config_dict, is_short_pos=True)
            borrowing_cost_val = calculate_daily_borrowing_cost(active_short_contracts, prev_close_price, config_dict, is_short_pos=True)
            
            current_cash_B += daily_cfd_pnl_val
            current_cash_B -= financing_cost_val
            current_cash_B -= borrowing_cost_val
        
        target_contracts_today = get_hedge_action(vix_level_today, current_equity_B, current_close_price, config_dict)
        contracts_change_today = target_contracts_today - active_short_contracts
        
        if abs(contracts_change_today) > 1e-6:
            spread_cost_val = calculate_spread_cost(abs(contracts_change_today), current_close_price, config_dict)
            current_cash_B -= spread_cost_val
            active_short_contracts = target_contracts_today
        
        new_margin_needed = 0.0
        if active_short_contracts > 0 and pd.notna(current_close_price) and current_close_price > 0:
            new_margin_needed = calculate_margin(active_short_contracts, current_close_price, config_dict)
        else:
            active_short_contracts = 0.0
        
        margin_diff = new_margin_needed - current_margin_held
        if abs(margin_diff) > 1e-6:
            if margin_diff > 0 and current_cash_B < margin_diff:
                # st.warning(f"MARGIN CALL on {date_idx}: Cash {current_cash_B:.2f} < Margin Increase {margin_diff:.2f}. Forcing close of hedge.") # Less verbose
                if active_short_contracts > 0:
                     cost_to_close_on_margin_call = calculate_spread_cost(active_short_contracts, current_close_price, config_dict)
                     spread_cost_val += cost_to_close_on_margin_call
                     current_cash_B -= cost_to_close_on_margin_call
                current_cash_B += current_margin_held
                active_short_contracts = 0.0
                new_margin_needed = 0.0
                margin_diff = new_margin_needed - current_margin_held
            
            current_cash_B -= margin_diff
            current_margin_held = new_margin_needed
            
        cost_details_history.append({
            'date': date_idx,
            'Financing': financing_cost_val,
            'Borrowing': borrowing_cost_val,
            'Spread': spread_cost_val,
            'Total': financing_cost_val + borrowing_cost_val + spread_cost_val,
            'Contracts': active_short_contracts,
            'Margin': current_margin_held,
            'CFD_PnL': daily_cfd_pnl_val
        })
        
        portfolio_val_B = current_equity_B + current_cash_B + current_margin_held
        if pd.isna(portfolio_val_B):
            portfolio_val_B = portfolio_B_history[-1] if portfolio_B_history else initial_cap
        
        portfolio_B_history.append(portfolio_val_B)
        dates_B_history.append(date_idx)

    if not dates_B_history: return pd.Series(name="Portfolio_B", dtype=float), pd.DataFrame()
    
    portfolio_B_series = pd.Series(portfolio_B_history, index=pd.DatetimeIndex(dates_B_history), name="Portfolio_B")
    cost_details_df = pd.DataFrame(cost_details_history).set_index('date')
    return portfolio_B_series, cost_details_df

# ==============================================================================
# Plotting Functions (Modified for Plotly)
# ==============================================================================
def plot_portfolio_comparison_plotly(portfolio_A_series, portfolio_B_series, config_dict):
    initial_cap = config_dict['initial_capital']
    
    if portfolio_A_series.empty or portfolio_B_series.empty:
        st.warning("No simulation data to plot.")
        return go.Figure() # Return empty figure

    comp_df = pd.concat([portfolio_A_series, portfolio_B_series], axis=1).rename(
        columns={portfolio_A_series.name: 'Portfolio_A', portfolio_B_series.name: 'Portfolio_B'}
    ).ffill().dropna()
    
    if comp_df.empty:
        st.warning("No common dates to plot after processing.")
        return go.Figure()

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=comp_df.index, y=comp_df['Portfolio_A'] / initial_cap,
                             mode='lines', name='Model A (Classic)',
                             line=dict(width=2)))

    fig.add_trace(go.Scatter(x=comp_df.index, y=comp_df['Portfolio_B'] / initial_cap,
                             mode='lines', name='Model B (CFD-Hedged)',
                             line=dict(width=2), opacity=0.8))

    shapes = []
    for name, period in config_dict.get('analysis_periods', {}).items():
        start_dt = pd.to_datetime(period['start'])
        end_dt = pd.to_datetime(period['end'])
        color = 'rgba(255, 160, 122, 0.2)' if 'covid' in name.lower() else 'rgba(135, 206, 250, 0.2)' # LightSalmon, LightSkyBlue
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=start_dt,
                y0=0,
                x1=end_dt,
                y1=1,
                fillcolor=color,
                opacity=0.5,
                layer="below",
                line_width=0,
                name=f"{name.replace('_',' ').title()}" # This name won't appear directly, but good for reference
            )
        )
    # Add dummy traces for legend entries for shapes (Plotly limitation)
    # Get unique colors and names
    unique_shapes_info = []
    seen_names = set()
    for name, period in config_dict.get('analysis_periods', {}).items():
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
        title=dict(text='<b>Portfolio Value Comparison (Normalized)</b>', x=0.5, font=dict(size=20)),
        xaxis_title='Date',
        yaxis_title='Normalized Portfolio Value',
        legend_title_text='Portfolios & Periods',
        font=dict(family="Arial, sans-serif", size=12),
        yaxis_tickformat=",.2f",
        shapes=shapes,
        hovermode="x unified",
        template="plotly_white" # Using a clean template
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    return fig

# --- Streamlit App ---

st.set_page_config(layout="wide", page_title="CFD Hedging Analysis")

# Custom CSS for fonts (optional, Streamlit's default is quite good)
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif; /* Example: Change global font */
    }
    .stApp {
        /* background-color: #f0f2f6; */ /* Light gray background */
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stMetric {
        background-color: #FFFFFF;
        border-radius: 7px;
        padding: 10px !important; /* !important may be needed */
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
    }
    </style>
""", unsafe_allow_html=True)


st.title("📈 CFD Hedging Strategy Analysis")
st.markdown("This application simulates and compares a classic S&P 500 portfolio (Model A) against a portfolio that uses VIX-triggered short S&P 500 CFDs for hedging (Model B). Adjust parameters in the sidebar to see their impact.")


# Default Configuration
default_config = {
    'initial_capital': 1000000,
    'equity_allocation': 0.8,
    'trading_days_per_year': 252,
    'data_file': 'data/vix_sp500_data.csv',
    'date_column': 'date',
    'sp500_column': 'S&P500',
    'vix_column': 'VIX',
    'sofr_column': 'SOFR',
    'raw_csv_sp500_col': 'S&P500',
    'raw_csv_vix_col': 'VIX',
    'raw_csv_sofr_col': 'SOFR',
    'hedging_strategy': {
        'vix_threshold': 20.0,
        'hedge_ratio': 0.50,
    },
    'lot_size': 1.0,
    'broker_annual_financing_fee': 0.03,
    'borrowing_cost_annual': 0.006,
    'days_in_year_financing': 360,
    'avg_spread_points': 0.3,
    'margin_tiers': [
        {'limit': 5, 'rate': 0.005},
        {'limit': 25, 'rate': 0.01},
        {'limit': 40, 'rate': 0.03},
    ],
    'analysis_periods': {
        'covid_crisis': {'start': '2020-02-15', 'end': '2020-04-15'},
        'vix_spike_2025': {'start': '2025-03-01', 'end': '2025-05-15'}
    }
}

# Sidebar for User Inputs
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("---")
    initial_capital = st.number_input("Initial Capital ($)", min_value=10000, value=default_config['initial_capital'], step=10000, format="%d")
    equity_allocation = st.slider("Equity Allocation (%)", min_value=0.0, max_value=1.0, value=default_config['equity_allocation'], step=0.01)
    st.markdown("---")
    st.subheader("Hedging Strategy")
    vix_threshold = st.slider("VIX Threshold to Activate Hedge", min_value=10.0, max_value=50.0, value=default_config['hedging_strategy']['vix_threshold'], step=0.5)
    hedge_ratio = st.slider("Hedge Ratio (Proportion of Equity)", min_value=0.0, max_value=1.0, value=default_config['hedging_strategy']['hedge_ratio'], step=0.01)
    st.markdown("---")
    st.subheader("CFD Costs")
    broker_financing_fee = st.slider("Broker Annual Financing Fee (Markup)", min_value=0.0, max_value=0.1, value=default_config['broker_annual_financing_fee'], step=0.001, format="%.3f")
    borrowing_cost_annual = st.slider("Annual Borrowing Cost for Shorts", min_value=0.0, max_value=0.05, value=default_config['borrowing_cost_annual'], step=0.001, format="%.3f")
    avg_spread_points = st.number_input("Average Spread (S&P 500 points)", min_value=0.0, max_value=5.0, value=default_config['avg_spread_points'], step=0.01, format="%.2f")

    # Data file path
    script_dir = os.path.dirname(__file__) # Gets the directory of the current script
    data_file_path_rel = default_config['data_file']
    data_file_path_abs = os.path.join(script_dir, data_file_path_rel)


# Main area
if st.sidebar.button("🚀 Run Simulation & Plot", use_container_width=True):
    current_config = default_config.copy()
    current_config['initial_capital'] = initial_capital
    current_config['equity_allocation'] = equity_allocation
    current_config['hedging_strategy']['vix_threshold'] = vix_threshold
    current_config['hedging_strategy']['hedge_ratio'] = hedge_ratio
    current_config['broker_annual_financing_fee'] = broker_financing_fee
    current_config['borrowing_cost_annual'] = borrowing_cost_annual
    current_config['avg_spread_points'] = avg_spread_points

    with st.spinner("Loading and preparing data..."):
        main_df = load_and_prepare_data(current_config, data_file_path_abs)

    if not main_df.empty:
        sim_portfolio_A = simulate_classic_portfolio(main_df, current_config)
        sim_portfolio_B, sim_cost_df = simulate_hedged_portfolio(main_df, current_config)

        if sim_portfolio_A.empty or sim_portfolio_B.empty:
            st.error("Simulation produced empty results. Check data and parameters.")
        else:
            st.header("📊 Portfolio Value Comparison")
            fig_comp_plotly = plot_portfolio_comparison_plotly(sim_portfolio_A, sim_portfolio_B, current_config)
            st.plotly_chart(fig_comp_plotly, use_container_width=True)
            
            st.header("📈 Simulation Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model A Final Value", f"${sim_portfolio_A.iloc[-1]:,.0f}")
                st.metric("Model A Return", f"{(sim_portfolio_A.iloc[-1]/initial_capital -1)*100:.1f}%")
            with col2:
                st.metric("Model B Final Value", f"${sim_portfolio_B.iloc[-1]:,.0f}")
                st.metric("Model B Return", f"{(sim_portfolio_B.iloc[-1]/initial_capital -1)*100:.1f}%")
            with col3:
                if not sim_cost_df.empty:
                     st.metric("Model B Total CFD Costs", f"${sim_cost_df['Total'].sum():,.0f}")
                else:
                     st.metric("Model B Total CFD Costs", "N/A")
            
            with st.expander("Show Raw Simulation Data"):
                st.subheader("Model A Portfolio Value")
                st.dataframe(sim_portfolio_A.tail())
                st.subheader("Model B Portfolio Value")
                st.dataframe(sim_portfolio_B.tail())
                st.subheader("Model B Cost Details")
                st.dataframe(sim_cost_df.tail())
    else:
        st.warning("Data could not be loaded. Please ensure `data/vix_sp500_data.csv` exists and is correctly formatted.")
else:
    st.info("Adjust parameters in the sidebar and click '🚀 Run Simulation & Plot'.")

st.sidebar.markdown("---")
st.sidebar.markdown("Created with ❤️")
