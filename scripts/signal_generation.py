import pandas as pd
import numpy as np

def generate_vix_momentum_signals(df, cfg):
    """
    Generates VIX momentum-based trading signals.
    df must contain 'date' and 'VIX' columns.
    cfg is the config module/object with VIX signal parameters.
    """
    print("--- Generating VIX Momentum Signals ---")
    df_signal = df.copy()

    # 1. Calculate VIX lagged value and 5-day percentage change
    df_signal['VIX_Lagged_5D'] = df_signal['VIX'].shift(cfg.MOMENTUM_LOOKBACK_PERIOD)
    df_signal['VIX_Pct_Change_5D'] = (df_signal['VIX'] - df_signal['VIX_Lagged_5D']) / df_signal['VIX_Lagged_5D']

    # 2. Identify days meeting the raw up/down momentum conditions
    df_signal['Is_VIX_Momentum_Up_Condition_Met'] = df_signal['VIX_Pct_Change_5D'] > cfg.VIX_PCT_CHANGE_THRESHOLD_UP
    df_signal['Is_VIX_Momentum_Down_Condition_Met'] = df_signal['VIX_Pct_Change_5D'] < cfg.VIX_PCT_CHANGE_THRESHOLD_DOWN

    # 3. Calculate consecutive days for these conditions
    # For UP streaks (potential short signal)
    up_blocks = (df_signal['Is_VIX_Momentum_Up_Condition_Met'] != df_signal['Is_VIX_Momentum_Up_Condition_Met'].shift()).cumsum()
    df_signal['Consecutive_VIX_Momentum_Up_Days'] = df_signal.groupby(up_blocks).cumcount() + 1
    df_signal.loc[~df_signal['Is_VIX_Momentum_Up_Condition_Met'], 'Consecutive_VIX_Momentum_Up_Days'] = 0

    # For DOWN streaks (potential cover signal component)
    down_blocks = (df_signal['Is_VIX_Momentum_Down_Condition_Met'] != df_signal['Is_VIX_Momentum_Down_Condition_Met'].shift()).cumsum()
    df_signal['Consecutive_VIX_Momentum_Down_Days'] = df_signal.groupby(down_blocks).cumcount() + 1
    df_signal.loc[~df_signal['Is_VIX_Momentum_Down_Condition_Met'], 'Consecutive_VIX_Momentum_Down_Days'] = 0

    # 4. Generate final signals
    df_signal['Short_Signal_Today'] = df_signal['Consecutive_VIX_Momentum_Up_Days'] >= cfg.N_CONSECUTIVE_UP_DAYS_TO_SHORT
    df_signal['Cover_Signal_Momentum_Today'] = df_signal['Consecutive_VIX_Momentum_Down_Days'] >= cfg.N_CONSECUTIVE_DOWN_DAYS_TO_COVER
    df_signal['Cover_Signal_Absolute_VIX_Today'] = df_signal['VIX'] < cfg.VIX_ABSOLUTE_COVER_THRESHOLD
    
    print("VIX Momentum signals calculated.")
    return df_signal