# scripts/config.py

# --- Portfolio Configuration (static parts or defaults if not in UI) ---
# Model A
EQUITY_ALLOC_A = 0.80 # Default, can be overridden by UI if UI provides this for A specifically
CASH_ALLOC_A = 1.0 - EQUITY_ALLOC_A

# Model B
EQUITY_ALLOC_B = 0.80 # Default, can be overridden by UI (Streamlit uses 'equity_allocation' for both)
CASH_ALLOC_B = 1.0 - EQUITY_ALLOC_B
MODEL_B_FIXED_HEDGE_RATIO_MOMENTUM = 0.70 # This IS the hedge ratio used for Model B

# --- VIX Momentum Signal Generation Parameters ---
MOMENTUM_LOOKBACK_PERIOD = 5
VIX_PCT_CHANGE_THRESHOLD_UP = 0.35  # > +35%
VIX_PCT_CHANGE_THRESHOLD_DOWN = -0.20 # < -20%
N_CONSECUTIVE_UP_DAYS_TO_SHORT = 3
N_CONSECUTIVE_DOWN_DAYS_TO_COVER = 1
VIX_ABSOLUTE_COVER_THRESHOLD = 20.0 # Matches UI vix_threshold for simplicity of cover

# --- CFD Cost Parameters (static parts or defaults if not in UI) ---
# These are mostly covered by Streamlit UI, but listed here for completeness if engine expects them via cfg
# BROKER_FEE_ANNUALIZED = 0.03 # From Streamlit default
# SPREAD_COST_PERCENT = calculated from avg_spread_points
# CFD_INITIAL_MARGIN_PERCENT - not in UI explicitly for B, using margin_tiers logic in app's cost functions
# LOT_SIZE = 1.0 # From Streamlit default
# DAYS_IN_YEAR_FINANCING = 360 # From Streamlit default

# --- Metrics Calculation ---
TRADING_DAYS_PER_YEAR = 252 # Matches Streamlit config

# --- Plotting ---
PORTFOLIO_A_LABEL = "Model A (Classic)"
# PORTFOLIO_B_LABEL is dynamic based on hedge ratio
PLOT_TITLE_PERFORMANCE = "Portfolio Value Comparison (Normalized)"
COLOR_A = "orange"
COLOR_B_MOMENTUM = "#007BBF" # Specific for Model B VIX Momentum