import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# --- Import new modular scripts ---
import scripts.config as static_cfg
import scripts.signal_generation as sg
import scripts.simulation_engine as sim_eng

# ==============================================================================
# Utility Class for Config
# ==============================================================================
class ConfigObject:
    """Helper class to access dict keys as attributes for modular functions."""
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigObject(value) if isinstance(value, dict) else value)
            else:
                setattr(self, key, value)

# ==============================================================================
# Data Loading and Preprocessing (Adjusted for provided CSV)
# ==============================================================================
@st.cache_data
def load_and_prepare_data(config_dict_ui, data_file_path_abs):
    date_col_name = config_dict_ui['date_column']
    sp500_target_col = config_dict_ui['sp500_column']
    vix_target_col = config_dict_ui['vix_column']
    sofr_target_col = 'SOFR_Rate'
    config_dict_ui['sofr_column'] = sofr_target_col

    if not os.path.exists(data_file_path_abs):
        st.error(f"Data file not found: {data_file_path_abs}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(data_file_path_abs)
        if 'date' not in df.columns:
             if df.columns[0] == 'date': pass
             elif 'Unnamed: 0' in df.columns and df.columns[1] == 'date':
                 df = df.rename(columns = {'Unnamed: 0':'orig_idx'})
             else:
                st.error(f"Expected 'date' column not found. Columns are: {df.columns.tolist()}")
                return pd.DataFrame()
        
        df[date_col_name] = pd.to_datetime(df[date_col_name])
        df = df.set_index(date_col_name)
    except Exception as e:
        st.error(f"Error loading or parsing CSV '{data_file_path_abs}': {e}")
        return pd.DataFrame()
    
    required_cols_final = [sp500_target_col, vix_target_col, sofr_target_col, 'SP500_Return', 'Prev_S&P500']
    if not all(col in df.columns for col in required_cols_final):
        missing_cols = [col for col in required_cols_final if col not in df.columns]
        st.error(f"DataFrame missing required columns after initial load: {missing_cols}. Available: {df.columns.tolist()}")
        return pd.DataFrame()

    for col in [sp500_target_col, vix_target_col, sofr_target_col, 'SP500_Return', 'Prev_S&P500']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.loc[df[sp500_target_col] <= 0, sp500_target_col] = np.nan

    target_daily_positive_drift = 0.00052
    valid_return_indices = df['SP500_Return'].notna()
    if not df.loc[valid_return_indices, 'SP500_Return'].empty:
        original_mean_return = df.loc[valid_return_indices, 'SP500_Return'].mean()
        df.loc[valid_return_indices, 'SP500_Return'] = \
            (df.loc[valid_return_indices, 'SP500_Return'] - original_mean_return) + target_daily_positive_drift
        
        first_valid_price_idx = df[sp500_target_col].first_valid_index()
        if first_valid_price_idx is not None:
            base_price = df.loc[first_valid_price_idx, sp500_target_col]
            reconstructed_prices = [np.nan] * len(df)
            numeric_idx_start = df.index.get_loc(first_valid_price_idx)
            reconstructed_prices[numeric_idx_start] = base_price
            for i in range(numeric_idx_start + 1, len(df)):
                if pd.notna(df['SP500_Return'].iloc[i]) and pd.notna(reconstructed_prices[i-1]):
                    reconstructed_prices[i] = reconstructed_prices[i-1] * (1 + df['SP500_Return'].iloc[i])
                elif pd.notna(reconstructed_prices[i-1]):
                    reconstructed_prices[i] = reconstructed_prices[i-1]
            df[sp500_target_col] = reconstructed_prices
            df['Prev_S&P500'] = df[sp500_target_col].shift(1)
        else:
            st.warning("S&P500 price reconstruction skipped (no valid initial S&P500 price found).")
    else:
        st.warning("No valid original returns for drift modification.")

    initial_rows_count = len(df)
    essential_check_cols = [sp500_target_col, vix_target_col, sofr_target_col, 'SP500_Return', 'Prev_S&P500']
    df = df[df.index.notna()]
    df = df.dropna(subset=essential_check_cols)
    rows_dropped_count = initial_rows_count - len(df)
    
    if df.empty:
        st.error("DataFrame is empty after final cleaning and NaN removal.")
        return pd.DataFrame()
    
    df = df.sort_index()
    return df

# ==============================================================================
# Plotting Functions
# ==============================================================================
def plot_portfolio_comparison_plotly(portfolio_A_series, portfolio_B_series, config_dict_ui, model_B_label="Model B (CFD-Hedged)"):
    initial_cap = config_dict_ui['initial_capital']
    
    if portfolio_A_series.empty or portfolio_B_series.empty:
        st.warning("No simulation data to plot.")
        return go.Figure()

    comp_df = pd.concat([portfolio_A_series, portfolio_B_series], axis=1).rename(
        columns={portfolio_A_series.name: 'Portfolio_A', portfolio_B_series.name: 'Portfolio_B'}
    )
    comp_df = comp_df.dropna(subset=['Portfolio_A', 'Portfolio_B'], how='all')
    comp_df = comp_df.ffill()
    comp_df = comp_df.dropna()

    if comp_df.empty:
        st.warning("No common dates to plot after processing.")
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=comp_df.index, y=comp_df['Portfolio_A'] / initial_cap,
                             mode='lines', name=static_cfg.PORTFOLIO_A_LABEL,
                             line=dict(width=2, color=static_cfg.COLOR_A)))
    fig.add_trace(go.Scatter(x=comp_df.index, y=comp_df['Portfolio_B'] / initial_cap,
                             mode='lines', name=model_B_label,
                             line=dict(width=2, color=static_cfg.COLOR_B_MOMENTUM), opacity=0.8))

    shapes = []
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
st.markdown("""<style> /* Your CSS */ </style>""", unsafe_allow_html=True) # Keep CSS concise

st.title("📈 CFD Hedging & VIX Momentum Strategy Analysis")
st.markdown("Compares a **Model A (80% Equity, 20% Cash)** against **Model B (80/20 base with VIX Momentum-triggered CFD hedge)**.")

# --- MODIFICATION: Fixed Equity Allocation ---
FIXED_EQUITY_ALLOCATION = 0.80

default_config_ui = {
    'initial_capital': 1000000,
    'equity_allocation': FIXED_EQUITY_ALLOCATION, # Fixed
    'trading_days_per_year': 252,
    'data_file': 'vix_sp500_data.csv',
    'date_column': 'date',
    'sp500_column': 'S&P500',
    'vix_column': 'VIX',
    'sofr_column': 'SOFR_Rate',
    'raw_csv_sofr_col': 'SOFR_Rate',
    'model_b_fixed_hedge_ratio_momentum': 0.70,
    'lot_size': 1.0,
    'broker_annual_financing_fee': 0.025,
    'borrowing_cost_annual': 0.0,
    'days_in_year_financing': 365,
    'avg_spread_points': 0.3,
    'cfd_initial_margin_percent': 0.05,
    'margin_tiers': [
        {'limit': 5, 'rate': 0.05},
        {'limit': 25, 'rate': 0.10},
        {'limit': float('inf'), 'rate': 0.15},
    ],
    'analysis_periods': {
        'covid_crisis': {'start': '2020-02-01', 'end': '2020-04-30'},
        'vix_spike_2025': {'start': '2025-03-01', 'end': '2025-05-21'}
    }
}

# Sidebar for User Inputs
with st.sidebar:
    st.header("⚙️ General Configuration")
    initial_capital_ui = st.number_input("Initial Capital ($)", min_value=10000, value=default_config_ui['initial_capital'], step=10000, format="%d")
    # --- MODIFICATION: Removed equity_allocation_ui slider ---
    st.markdown(f"**Fixed Equity Allocation: {FIXED_EQUITY_ALLOCATION*100:.0f}%** (for both models' base)")
    
    st.markdown("---")
    st.subheader("Model B: VIX Momentum Hedge")
    model_b_hedge_ratio_ui = st.slider("CFD Hedge Ratio (Proportion of Equity)", min_value=0.0, max_value=1.0, value=default_config_ui['model_b_fixed_hedge_ratio_momentum'], step=0.01)
    vix_abs_cover_thresh_ui = st.slider("VIX Absolute Cover Threshold (Model B)", min_value=10.0, max_value=30.0, value=static_cfg.VIX_ABSOLUTE_COVER_THRESHOLD, step=0.5)

    st.markdown("---")
    st.subheader("CFD Costs (for Model B)")
    broker_financing_fee_ui = st.slider("Broker Annual Financing Fee (Markup/Markdown vs SOFR)", min_value=0.0, max_value=0.05, value=default_config_ui['broker_annual_financing_fee'], step=0.001, format="%.3f")
    avg_spread_points_ui = st.number_input("Spread Cost (S&P 500 points per contract transaction)", min_value=0.0, max_value=2.0, value=default_config_ui['avg_spread_points'], step=0.01, format="%.2f")
    cfd_initial_margin_percent_ui = st.slider("CFD Initial Margin Percent", min_value=0.01, max_value=0.20, value=default_config_ui['cfd_initial_margin_percent'], step=0.01)

script_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path_abs = os.path.join(script_dir, default_config_ui['data_file'])
if not os.path.exists(data_file_path_abs) and default_config_ui['data_file'].startswith("data/"):
     data_file_path_abs = os.path.join(script_dir, os.path.basename(default_config_ui['data_file']))

if st.sidebar.button("🚀 Run Simulation & Plot", use_container_width=True):
    cfg_dict_for_backend = {
        'INITIAL_CAPITAL': initial_capital_ui,
        'TRADING_DAYS_PER_YEAR': default_config_ui['trading_days_per_year'],
        'sp500_column': default_config_ui['sp500_column'],
        'vix_column': default_config_ui['vix_column'],    
        'sofr_column': default_config_ui['sofr_column'],  
        
        # --- MODIFICATION: Use FIXED_EQUITY_ALLOCATION ---
        'EQUITY_ALLOC_A': FIXED_EQUITY_ALLOCATION,
        'CASH_ALLOC_A': 1.0 - FIXED_EQUITY_ALLOCATION, 
        'EQUITY_ALLOC_B': FIXED_EQUITY_ALLOCATION,
        'CASH_ALLOC_B': 1.0 - FIXED_EQUITY_ALLOCATION,
        # --- END OF MODIFICATION ---
        
        'MODEL_B_FIXED_HEDGE_RATIO_MOMENTUM': model_b_hedge_ratio_ui,
        'MOMENTUM_LOOKBACK_PERIOD': static_cfg.MOMENTUM_LOOKBACK_PERIOD,
        'VIX_PCT_CHANGE_THRESHOLD_UP': static_cfg.VIX_PCT_CHANGE_THRESHOLD_UP,
        'VIX_PCT_CHANGE_THRESHOLD_DOWN': static_cfg.VIX_PCT_CHANGE_THRESHOLD_DOWN,
        'N_CONSECUTIVE_UP_DAYS_TO_SHORT': static_cfg.N_CONSECUTIVE_UP_DAYS_TO_SHORT,
        'N_CONSECUTIVE_DOWN_DAYS_TO_COVER': static_cfg.N_CONSECUTIVE_DOWN_DAYS_TO_COVER,
        'VIX_ABSOLUTE_COVER_THRESHOLD': vix_abs_cover_thresh_ui,
        'BROKER_FEE_ANNUALIZED': broker_financing_fee_ui,
        'SPREAD_COST_PERCENT': 0.0002, 
        'AVG_SPREAD_POINTS': avg_spread_points_ui, 
        'CFD_INITIAL_MARGIN_PERCENT': cfd_initial_margin_percent_ui,
        'LOT_SIZE': default_config_ui['lot_size'],
        'DAYS_IN_YEAR_FINANCING': default_config_ui['days_in_year_financing'],
        'PORTFOLIO_A_LABEL': static_cfg.PORTFOLIO_A_LABEL,
        'COLOR_A': static_cfg.COLOR_A,
        'COLOR_B_MOMENTUM': static_cfg.COLOR_B_MOMENTUM
    }
    cfg_object = ConfigObject(cfg_dict_for_backend)

    with st.spinner("Loading and preparing data..."):
        main_df_loaded = load_and_prepare_data(default_config_ui, data_file_path_abs) # default_config_ui is fine here

    if not main_df_loaded.empty:
        with st.spinner("Generating VIX signals..."):
            main_df_with_signals = sg.generate_vix_momentum_signals(main_df_loaded, cfg_object) 
        
        cols_to_check_for_nan = [
            cfg_object.sp500_column, cfg_object.vix_column, cfg_object.sofr_column,
            'SP500_Return', 'Prev_S&P500',
            'VIX_Lagged_5D', 'VIX_Pct_Change_5D',
            'Short_Signal_Today', 'Cover_Signal_Momentum_Today', 'Cover_Signal_Absolute_VIX_Today'
            ]
        main_df_sim = main_df_with_signals.dropna(subset=cols_to_check_for_nan).copy()
        main_df_sim.reset_index(drop=True, inplace=True)

        if main_df_sim.empty:
            st.error("Dataframe became empty after signal generation and NaN dropping. Cannot simulate.")
        else:
            st.sidebar.success(f"Data ready for simulation: {len(main_df_sim)} rows.")

            sim_portfolio_A = sim_eng.simulate_portfolio_A(
                main_df_sim, 
                cfg_object.INITIAL_CAPITAL, 
                cfg_object.EQUITY_ALLOC_A # This will now be the fixed 0.8
            )
            sim_portfolio_B = sim_eng.simulate_portfolio_B_momentum(
                main_df_sim, 
                cfg_object.INITIAL_CAPITAL, 
                cfg_object
            )
            sim_cost_df = pd.DataFrame() 

            if sim_portfolio_A.empty or sim_portfolio_B.empty:
                st.error("Simulation produced empty results. Check data and parameters.")
            else:
                st.header("📊 Portfolio Value Comparison")
                model_B_dynamic_label = f"Model B (VIX Momentum HR:{cfg_object.MODEL_B_FIXED_HEDGE_RATIO_MOMENTUM:.2f})"
                
                plot_config_for_plotly = {
                    'initial_capital': initial_capital_ui, # Use the UI value for initial cap in plot
                    'analysis_periods': default_config_ui['analysis_periods']
                }

                fig_comp_plotly = plot_portfolio_comparison_plotly(
                    sim_portfolio_A.rename("Portfolio_A"),
                    sim_portfolio_B.rename("Portfolio_B"), 
                    plot_config_for_plotly,
                    model_B_dynamic_label
                )
                st.plotly_chart(fig_comp_plotly, use_container_width=True)
                
                st.header("📈 Simulation Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Model A Final Value", f"${sim_portfolio_A.iloc[-1]:,.0f}")
                    st.metric("Model A Return", f"{(sim_portfolio_A.iloc[-1]/initial_capital_ui -1)*100:.1f}%")
                with col2:
                    st.metric(f"{model_B_dynamic_label} Final Value", f"${sim_portfolio_B.iloc[-1]:,.0f}")
                    st.metric(f"{model_B_dynamic_label} Return", f"{(sim_portfolio_B.iloc[-1]/initial_capital_ui -1)*100:.1f}%")
                
                with st.expander("Show Raw Simulation Data (Last 5 rows)"):
                    st.subheader(static_cfg.PORTFOLIO_A_LABEL)
                    st.dataframe(sim_portfolio_A.tail())
                    st.subheader(model_B_dynamic_label)
                    st.dataframe(sim_portfolio_B.tail())
    else:
        st.warning("Data could not be loaded. Please ensure your data file is correctly specified and formatted.")
else:
    st.info("Adjust parameters in the sidebar and click '🚀 Run Simulation & Plot'.")

st.sidebar.markdown("---")
st.sidebar.info("This app uses a VIX Momentum strategy for Model B's CFD hedging. Base allocation for both models is fixed at 80% Equity / 20% Cash.")