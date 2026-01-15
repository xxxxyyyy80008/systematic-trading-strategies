"""SMI Strategy implementation."""
from typing import List
import pandas as pd
from backtester.strategy_base import (
    Strategy, 
    create_signal, 
    detect_threshold_cross_above,
    detect_crossover_below
)
from backtester.types_core import Signal, StrategyConfig
from backtester.indicators import vpn, rsi, sma, ema, atr, smi


class SMIStrategy(Strategy):
    """
    Stochastic Momentum Index (SMI) Strategy.
    
    Logic:
    1. Entry: SMI crosses ABOVE Signal Line AND SMI is below threshold (-50).
    2. Exit: SMI crosses BELOW Signal Line.
    """

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        # Default to user snippet values if not in params
        self.k_period = self.params.get('k_period', 8)
        self.d_period = self.params.get('d_period', 3)
        self.oversold_threshold = self.params.get('oversold_threshold', -50)
        self.overbought_threshold = self.params.get('overbought_threshold', 50)

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate SMI indicators. 
        Note: We do NOT dropna() here to preserve index alignment for WFA.
        """
        result = data.copy()
        
        # Calculate SMI
        smi_rlt = smi(
            ohlc=data,
            k_period=self.k_period,
            d_period=self.d_period
        )
        
        result['SMI'] = smi_rlt['SMI']
        result['SMI_SIGNAL'] = smi_rlt['SMI_SIGNAL']
        
        return result

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        signals = []
        
        # 1. Vectorized Condition Logic
        # -----------------------------
        
        # Previous values for crossover detection
        prev_smi = data['SMI'].shift(1)
        prev_signal = data['SMI_SIGNAL'].shift(1)
        
        # Entry: Cross Over AND Deep Oversold (< -50)
        # Note: Checking both SMI and SIGNAL < -50 ensures the whole cross happens in the zone
        entry_mask = (
            (data['SMI'] >= data['SMI_SIGNAL']) &      # Current: SMI > Sig
            (prev_smi < prev_signal) &                 # Previous: SMI < Sig
            (data['SMI'] < self.oversold_threshold) &  # Filter: Deep Oversold
            (data['SMI_SIGNAL'] < self.oversold_threshold)
        )

        # Exit: Cross Under
        exit_mask = (
            (data['SMI'] < data['SMI_SIGNAL']) &       # Current: SMI < Sig
            (prev_smi >= prev_signal) &                # Previous: SMI > Sig
            (data['SMI_SIGNAL'] > self.overbought_threshold)
        )

        # 2. Iterate Valid Indices to Create Signal Objects
        # -------------------------------------------------
        
        # Get indices where Entry is True
        entry_indices = data.index[entry_mask]
        for idx in entry_indices:
            row = data.loc[idx]
            signals.append(create_signal(
                ticker=row.get('ticker', 'UNKNOWN'),
                date=idx,
                signal_type=1, # LONG
                price=row['Close'],
                signal_name='SMI_OVERSOLD_ENTRY'
            ))

        # Get indices where Exit is True
        exit_indices = data.index[exit_mask]
        for idx in exit_indices:
            row = data.loc[idx]
            signals.append(create_signal(
                ticker=row.get('ticker', 'UNKNOWN'),
                date=idx,
                signal_type=-1, # EXIT
                price=row['Close'],
                signal_name='SMI_CROSS_EXIT'
            ))
            
        return signals

