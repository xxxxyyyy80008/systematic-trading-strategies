"""
CAMA-KAMA Strategy Signal Visualization.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Any

def plot_cama_kama_signals(data: pd.DataFrame, 
                           ticker: str,
                           results: Any,
                           start_date: str = None, 
                           end_date: str = None,
                           strategy_config: Any = None):
    """
    Plot CAMA-KAMA strategy signals for a selected ticker.
    
    Subplot 1: Price, CAMA (Fast), KAMA (Slow), Entry/Exit Markers.
    Subplot 2: RSI with Overbought/Oversold thresholds.
    Subplot 3: Discrete Signal State (1=Entry, -1=Exit).
    """
    
    # 1. DATA PREPARATION & FILTERING
    # -------------------------------
    
    # Filter data for the selected ticker
    if 'ticker' in data.columns:
        ticker_data = data[data['ticker'] == ticker].copy()
    else:
        ticker_data = data.copy()
    
    if ticker_data.empty:
        print(f" No data found for ticker: {ticker}")
        return
    
    # Apply date filter
    if start_date:
        ticker_data = ticker_data[ticker_data.index >= start_date]
    if end_date:
        ticker_data = ticker_data[ticker_data.index <= end_date]
        
    if ticker_data.empty:
        print(f" No data found for the selected period")
        return

    # Extract Trades
    if hasattr(results, 'trades'):
        all_trades = results.trades
    elif isinstance(results, dict) and 'trades' in results:
        all_trades = results['trades']
    else:
        all_trades = pd.DataFrame()

    entries = pd.DataFrame()
    exits = pd.DataFrame()

    if not all_trades.empty:
        ticker_trades = all_trades[all_trades['ticker'] == ticker].copy()
        ticker_trades['execution_date'] = pd.to_datetime(ticker_trades['execution_date'])
        
        # Split Actions
        entries = ticker_trades[ticker_trades['action'] == 'OPEN'].copy()
        exits = ticker_trades[ticker_trades['action'] == 'CLOSE'].copy()
        
        # Filter Trades by Date
        if start_date:
            entries = entries[entries['execution_date'] >= start_date]
            exits = exits[exits['execution_date'] >= start_date]
        if end_date:
            entries = entries[entries['execution_date'] <= end_date]
            exits = exits[exits['execution_date'] <= end_date]

    # Extract Config Params
    if strategy_config is not None:
        if hasattr(strategy_config, 'parameters'):
            params = strategy_config.parameters
        elif isinstance(strategy_config, dict):
            params = strategy_config
        else:
            params = {}
    else:
        params = {}

    rsi_ob = params.get('rsi_overbought', 70)
    rsi_os = params.get('rsi_oversold', 30)
    cama_period = params.get('cama_period', 10)
    kama_period = params.get('kama_period', 60)

    # 2. PRINT HEADER
    # ---------------
    print(f"\n" + "="*80)
    print(f"CAMA-KAMA STRATEGY SIGNALS FOR {ticker}")
    if start_date or end_date:
        s_str = start_date if start_date else ticker_data.index[0].strftime('%Y-%m-%d')
        e_str = end_date if end_date else ticker_data.index[-1].strftime('%Y-%m-%d')
        print(f"Period: {s_str} to {e_str}")
    print("="*80)

    # 3. PLOTTING
    # -----------
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), 
                             sharex=True,
                             gridspec_kw={'height_ratios': [3, 1.5, 1]})
    
    plt.subplots_adjust(hspace=0.05)
    
    # === SUBPLOT 1: Price & Moving Averages ===
    ax1 = axes[0]
    
    # Price
    ax1.plot(ticker_data.index, ticker_data['Close'], 
             label='Close Price', color='black', alpha=0.5, linewidth=1)
    
    # Indicators
    if 'CAMA' in ticker_data.columns:
        ax1.plot(ticker_data.index, ticker_data['CAMA'], 
                 label=f'CAMA ({cama_period})', color='#00BFFF', linewidth=1.5) # Deep Sky Blue
    
    if 'KAMA' in ticker_data.columns:
        ax1.plot(ticker_data.index, ticker_data['KAMA'], 
                 label=f'KAMA ({kama_period})', color='#FF4500', linewidth=1.5) # Orange Red
    
    # Fill between crossovers
    if 'CAMA' in ticker_data.columns and 'KAMA' in ticker_data.columns:
        ax1.fill_between(ticker_data.index, ticker_data['CAMA'], ticker_data['KAMA'], 
                         where=(ticker_data['CAMA'] >= ticker_data['KAMA']), 
                         color='green', alpha=0.05, label='Bullish Trend')
        ax1.fill_between(ticker_data.index, ticker_data['CAMA'], ticker_data['KAMA'], 
                         where=(ticker_data['CAMA'] < ticker_data['KAMA']), 
                         color='red', alpha=0.05, label='Bearish Trend')

    # Helper for Trade Markers
    def plot_markers(ax, trade_df, marker, color, label):
        if len(trade_df) > 0:
            for _, trade in trade_df.iterrows():
                norm_date = trade['execution_date'].normalize()
                if norm_date in ticker_data.index:
                    price = trade['price']
                    ax.scatter(norm_date, price, color=color, s=120, marker=marker, 
                               edgecolors=f'dark{color}', linewidths=1.5, zorder=5)

    plot_markers(ax1, entries, '^', 'green', 'Entry')
    plot_markers(ax1, exits, 'v', 'red', 'Exit')

    # Legend Handlers
    ax1.scatter([], [], color='green', s=100, marker='^', label='Entry')
    ax1.scatter([], [], color='red', s=100, marker='v', label='Exit')
    
    ax1.set_title(f'{ticker} - Price vs CAMA/KAMA', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # === SUBPLOT 2: RSI ===
    ax2 = axes[1]
    
    if 'RSI' in ticker_data.columns:
        ax2.plot(ticker_data.index, ticker_data['RSI'], color='purple', linewidth=1.5, label='RSI')
        
        # Thresholds
        ax2.axhline(y=rsi_ob, color='red', linestyle='--', alpha=0.6, label=f'Overbought ({rsi_ob})')
        ax2.axhline(y=rsi_os, color='green', linestyle='--', alpha=0.6, label=f'Oversold ({rsi_os})')
        
        # Fill Zones
        ax2.fill_between(ticker_data.index, ticker_data['RSI'], 100, 
                         where=(ticker_data['RSI'] >= rsi_ob), color='red', alpha=0.2)
        ax2.fill_between(ticker_data.index, ticker_data['RSI'], 0, 
                         where=(ticker_data['RSI'] <= rsi_os), color='green', alpha=0.2)
        
        # Mark Entries on RSI (to verify condition < 30)
        if len(entries) > 0:
            for _, trade in entries.iterrows():
                norm_date = trade['execution_date'].normalize()
                if norm_date in ticker_data.index:
                    rsi_val = ticker_data.loc[norm_date, 'RSI']
                    ax2.scatter(norm_date, rsi_val, marker='^', color='green', s=60, zorder=5)

    ax2.set_ylabel('RSI', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # === SUBPLOT 3: Signals ===
    ax3 = axes[2]
    
    signal_series = pd.Series(0, index=ticker_data.index)
    for _, t in entries.iterrows():
        signal_series.loc[t['execution_date'].normalize()] = 1
    for _, t in exits.iterrows():
        signal_series.loc[t['execution_date'].normalize()] = -1
        
    active = signal_series[signal_series != 0]
    
    if not active.empty:
        colors = ['green' if x == 1 else 'red' for x in active.values]
        ax3.scatter(active.index, active.values, c=colors, s=50)
        
    ax3.set_yticks([-1, 0, 1])
    ax3.set_yticklabels(['Exit', 'Neutral', 'Entry'])
    ax3.grid(True, axis='x', alpha=0.3)
    ax3.set_ylabel('Action', fontsize=11, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax3.set_ylim(-1.5, 1.5)

    fig.autofmt_xdate()
    plt.show()

    # 4. STATS OUTPUT
    # ---------------
    print(f"\nStrategy Statistics for {ticker}:")
    print("-" * 60)
    print(f"  Entries: {len(entries)}")
    print(f"  Exits:   {len(exits)}")
    
    if len(exits) > 0 and 'net_pnl' in exits.columns:
        wins = exits[exits['net_pnl'] > 0]
        losses = exits[exits['net_pnl'] <= 0]
        win_rate = len(wins) / len(exits) * 100
        
        print(f"\n  Performance:")
        print(f"    Net P&L:   ${exits['net_pnl'].sum():,.2f}")
        print(f"    Win Rate:  {win_rate:.1f}% ({len(wins)}/{len(exits)})")
        if len(wins) > 0:
            print(f"    Avg Win:   ${wins['net_pnl'].mean():,.2f}")
        if len(losses) > 0:
            print(f"    Avg Loss:  ${losses['net_pnl'].mean():,.2f}")
    print("="*80)
