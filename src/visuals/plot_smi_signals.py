"""
SMI Strategy Signal Visualization.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Optional, Any

def plot_smi_signals(data: pd.DataFrame, 
                     ticker: str,
                     results: Any,
                     start_date: str = None, 
                     end_date: str = None,
                     strategy_config: Any = None):
    """
    Plot SMI strategy signals for a selected ticker.
    
    Args:
        data: DataFrame with OHLCV and SMI indicators (from strategy.prepare_data())
        ticker: Ticker symbol to plot
        results: BacktestResults object from engine.run()
        start_date: Start date (format: 'YYYY-MM-DD')
        end_date: End date (format: 'YYYY-MM-DD')
        strategy_config: StrategyConfig object or dict with strategy parameters
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
    
    # Apply date filter if provided
    if start_date:
        ticker_data = ticker_data[ticker_data.index >= start_date]
    if end_date:
        ticker_data = ticker_data[ticker_data.index <= end_date]
    
    if ticker_data.empty:
        print(f" No data found for the selected period")
        return
    
    # Extract trades for this ticker from results
    # Assumes results has a .trades DataFrame attribute
    all_trades = results.trades
    ticker_trades = all_trades[all_trades['ticker'] == ticker].copy()
    
    # Separate entry and exit trades
    entries = ticker_trades[ticker_trades['action'] == 'OPEN'].copy()
    exits = ticker_trades[ticker_trades['action'] == 'CLOSE'].copy()
    
    # Apply date filter to trades
    if start_date:
        entries = entries[pd.to_datetime(entries['execution_date']) >= start_date]
        exits = exits[pd.to_datetime(exits['execution_date']) >= start_date]
    if end_date:
        entries = entries[pd.to_datetime(entries['execution_date']) <= end_date]
        exits = exits[pd.to_datetime(exits['execution_date']) <= end_date]
    
    # Extract strategy config parameters
    if strategy_config is not None:
        if hasattr(strategy_config, 'parameters'):
            params = strategy_config.parameters
        else:
            params = strategy_config
    else:
        params = {}
    
    # SMI specific parameters
    oversold_thresh = params.get('oversold_threshold', -50)
    k_period = params.get('k_period', 14)
    d_period = params.get('d_period', 3)

    # 2. PRINT HEADER
    # ---------------
    print(f"\n" + "="*80)
    print(f"SMI STRATEGY SIGNALS FOR {ticker}")
    if start_date or end_date:
        s_str = start_date if start_date else ticker_data.index[0].strftime('%Y-%m-%d')
        e_str = end_date if end_date else ticker_data.index[-1].strftime('%Y-%m-%d')
        print(f"Period: {s_str} to {e_str}")
    print("="*80)
    
    # 3. PLOTTING
    # -----------
    # Create figure with 4 subplots
    fig, axes = plt.subplots(4, 1, figsize=(16, 16), 
                             sharex=True,
                             gridspec_kw={'height_ratios': [3, 2, 1.5, 1]})
    
    plt.subplots_adjust(hspace=0.05)
    
    # === SUBPLOT 1: Price Action & Trades ===
    ax1 = axes[0]
    
    ax1.plot(ticker_data.index, ticker_data['Close'], 
             label='Close Price', linewidth=1.5, color='black', alpha=0.7, zorder=2)
    
    # Plot Entry signals
    if len(entries) > 0:
        for idx, trade in entries.iterrows():
            exec_date = pd.to_datetime(trade['execution_date'])
            if exec_date in ticker_data.index:
                price = trade['price']
                signal_name = trade.get('signal_name', '')
                ax1.scatter(exec_date, price, color='green', s=120, marker='^', 
                           edgecolors='darkgreen', linewidths=1.5, zorder=5)
    
    # Legend for entries
    ax1.scatter([], [], color='green', s=100, marker='^', 
               edgecolors='darkgreen', linewidths=1.5, label='Entry')
    
    # Plot Exit signals
    if len(exits) > 0:
        for idx, trade in exits.iterrows():
            exec_date = pd.to_datetime(trade['execution_date'])
            if exec_date in ticker_data.index:
                price = trade['price']
                signal_name = trade.get('signal_name', '')
                ax1.scatter(exec_date, price, color='red', s=120, marker='v', 
                           edgecolors='darkred', linewidths=1.5, zorder=5)
    
    # Legend for exits
    ax1.scatter([], [], color='red', s=100, marker='v', 
               edgecolors='darkred', linewidths=1.5, label='Exit')
    
    ax1.set_title(f'{ticker} - SMI Strategy Signals', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price ($)', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(alpha=0.3, linestyle='--')
    
    # === SUBPLOT 2: SMI Oscillator ===
    ax2 = axes[1]
    
    if 'SMI' in ticker_data.columns and 'SMI_SIGNAL' in ticker_data.columns:
        ax2.plot(ticker_data.index, ticker_data['SMI'], 
                 label=f'SMI ({k_period},{d_period})', color='#1f77b4', linewidth=1.5)
        ax2.plot(ticker_data.index, ticker_data['SMI_SIGNAL'], 
                 label='Signal Line', color='#ff7f0e', linestyle='--', linewidth=1.5)
        
        # Plot Thresholds
        ax2.axhline(y=oversold_thresh, color='red', linestyle=':', linewidth=1.5, 
                   label=f'Oversold ({oversold_thresh})')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        # Fill Zones
        ax2.fill_between(ticker_data.index, ticker_data['SMI'], oversold_thresh, 
                         where=(ticker_data['SMI'] < oversold_thresh), 
                         color='red', alpha=0.15)
        
        # Mark Entries on Oscillator
        if len(entries) > 0:
            for idx, trade in entries.iterrows():
                exec_date = pd.to_datetime(trade['execution_date'])
                if exec_date in ticker_data.index:
                    smi_val = ticker_data.loc[exec_date, 'SMI']
                    ax2.scatter(exec_date, smi_val, marker='^', s=80, 
                               c='lime', edgecolors='darkgreen', zorder=5)

    ax2.set_ylabel('SMI Value', fontsize=11, fontweight='bold')
    ax2.set_ylim(-100, 100)
    ax2.legend(loc='lower right', fontsize=9)
    ax2.grid(alpha=0.3, linestyle='--')
    
    # === SUBPLOT 3: Momentum Histogram (SMI - Signal) ===
    ax3 = axes[2]
    
    if 'SMI' in ticker_data.columns and 'SMI_SIGNAL' in ticker_data.columns:
        histogram = ticker_data['SMI'] - ticker_data['SMI_SIGNAL']
        
        # Color bars based on positive/negative
        colors = np.where(histogram >= 0, 'green', 'red')
        
        ax3.bar(ticker_data.index, histogram, color=colors, alpha=0.6, width=1.0)
        ax3.axhline(0, color='black', linewidth=0.5)
        
    ax3.set_ylabel('Histogram', fontsize=11, fontweight='bold')
    ax3.set_title('Momentum Strength (SMI - Signal)', fontsize=10)
    ax3.grid(alpha=0.3, linestyle='--')

    # === SUBPLOT 4: Discrete Signal States ===
    ax4 = axes[3]
    
    # Create signal series from trades
    signal_series = pd.Series(0, index=ticker_data.index)
    
    # Mark entries as 1
    for idx, trade in entries.iterrows():
        exec_date = pd.to_datetime(trade['execution_date'])
        if exec_date in signal_series.index:
            signal_series.loc[exec_date] = 1
    
    # Mark exits as -1
    for idx, trade in exits.iterrows():
        exec_date = pd.to_datetime(trade['execution_date'])
        if exec_date in signal_series.index:
            signal_series.loc[exec_date] = -1
    
    # Plot signals
    colors = signal_series.map({1: 'green', -1: 'red', 0: 'gray'})
    # Filter out 0s for cleaner plot
    active_signals = signal_series[signal_series != 0]
    
    if not active_signals.empty:
        ax4.scatter(active_signals.index, active_signals, c=colors[active_signals.index], s=40, alpha=0.8)
    
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.2)
    ax4.set_yticks([-1, 0, 1])
    ax4.set_yticklabels(['Exit', 'Neutral', 'Entry'])
    ax4.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax4.set_ylim([-1.5, 1.5])
    ax4.grid(alpha=0.3, axis='x', linestyle='--')
    
    # Format dates
    fig.autofmt_xdate()
    plt.show()
    
    # 4. STATS OUTPUT
    # ---------------
    print(f"\nSMI Strategy Statistics for {ticker}:")
    print("-" * 80)
    print(f"  Total Entry Signals: {len(entries)}")
    print(f"  Total Exit Signals:  {len(exits)}")
    
    # Count signals by type if available
    if 'signal_name' in entries.columns:
        entry_counts = entries['signal_name'].value_counts()
        if len(entry_counts) > 0:
            print("\n  Entry Signal Types:")
            for signal_type, count in entry_counts.items():
                print(f"    {signal_type:<20}: {count}")
    
    # Trade performance for this ticker
    if len(exits) > 0:
        total_pnl = exits['net_pnl'].sum()
        wins = exits[exits['net_pnl'] > 0]
        losses = exits[exits['net_pnl'] <= 0]
        win_rate = len(wins) / len(exits) * 100 if len(exits) > 0 else 0
        
        print(f"\n  Trade Performance:")
        print(f"    Total P&L: ${total_pnl:,.2f}")
        print(f"    Wins: {len(wins)} | Losses: {len(losses)}")
        print(f"    Win Rate: {win_rate:.1f}%")
        
        if len(wins) > 0:
            print(f"    Avg Win:  ${wins['net_pnl'].mean():,.2f}")
        if len(losses) > 0:
            print(f"    Avg Loss: ${losses['net_pnl'].mean():,.2f}")
    
    print("="*80)
