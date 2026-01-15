from typing import List, Tuple
import pandas as pd
import numpy as np
from backtester.strategy_base import Strategy, create_signal
from backtester.types_core import Signal, StrategyConfig
from backtester.indicators import cama, kama, rsi

class CamaKamaStrategy(Strategy):
    """
    Adaptive Moving Average Crossover Strategy (CAMA vs KAMA).
    
    Logic:
    1. Entry: CAMA crosses ABOVE KAMA (Bullish) AND RSI is oversold (< 30).
    2. Exit: RSI becomes overbought (> 70).
    """

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        # Parameters
        self.cama_period = self.params.get('cama_period', 10)
        self.kama_period = self.params.get('kama_period', 60) # Main Lookback for KAMA
        self.kama_fast = self.params.get('kama_fast', 2)
        self.kama_slow = self.params.get('kama_slow', 30)
        self.er_period = self.params.get('er_period', 10)
        
        self.rsi_period = self.params.get('rsi_period', 9)
        self.rsi_oversold = self.params.get('rsi_oversold', 30)
        self.rsi_overbought = self.params.get('rsi_overbought', 70)

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        # Calculate Indicators
        result = data.copy()
#         data = data.copy()
#         data.columns = [c.lower() for c in data.columns]

        result['CAMA'] = cama(data, 
            period=self.cama_period
        )
        
        result['KAMA'] = kama(
            price=data['Close'], 
            er_period=self.er_period, 
            ema_fast=self.kama_fast, 
            ema_slow=self.kama_slow, 
            period=self.kama_period
        )
        
        result['RSI'] = rsi(
            prices=data['Close'], 
            period=self.rsi_period
        )
        
        return result

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        signals = []
        
        # 1. Vectorized Conditions
        # ------------------------
        
        # Crossover Logic
        # Entry: CAMA Crosses Over KAMA
        crossover_bullish = (data['CAMA'] >= data['KAMA']) & (data['CAMA'].shift(1) < data['KAMA'].shift(1))
        
        # Filter: RSI < Oversold
        rsi_condition_entry = data['RSI'] < self.rsi_oversold
        
        # Combined Entry
        entry_mask = crossover_bullish & rsi_condition_entry
        
        # Exit: RSI > Overbought (Simple Threshold Cross)
        exit_mask = data['RSI'] > self.rsi_overbought
        
        # 2. Signal Generation Loop
        # -------------------------
        
        # Entries
        entry_indices = data.index[entry_mask]
        for idx in entry_indices:
            row = data.loc[idx]
            signals.append(create_signal(
                ticker=row.get('ticker', 'UNKNOWN'),
                date=idx,
                signal_type=1, # LONG
                price=row['Close'],
                signal_name='CAMA_KAMA_ENTRY'
            ))
            
        # Exits
        exit_indices = data.index[exit_mask]
        for idx in exit_indices:
            row = data.loc[idx]
            signals.append(create_signal(
                ticker=row.get('ticker', 'UNKNOWN'),
                date=idx,
                signal_type=-1, # EXIT
                price=row['Close'],
                signal_name='RSI_OB_EXIT'
            ))
            
        return signals
