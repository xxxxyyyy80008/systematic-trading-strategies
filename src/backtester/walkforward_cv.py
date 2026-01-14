"""
Global Walk-Forward Analysis (Method 2: Static/Robust Optimization)
------------------------------------------------------------------
Optimizes a single set of parameters across all training windows (Cross-Validation)
and tests the robustness of these static parameters on Out-of-Sample data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from datetime import datetime, timedelta
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import warnings
import random
from scipy import stats 

warnings.filterwarnings('ignore')

# Assumed internal modules
try:
    from .backtest_engine import BacktestEngine
    from .types_core import StrategyConfig, TradeConfig
except ImportError:
    pass 


# ============================================================================
# CONFIGURATION
# ============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any

def calculate_cv(values: List[float]) -> float:
    """Calculate coefficient of variation."""
    if not values: return 0.0
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    return std / abs(mean) if mean != 0 else np.inf

def calculate_range_ratio(values: List[float]) -> float:
    """Calculate parameter range relative to median."""
    if not values: return 0.0
    median = np.median(values)
    return (max(values) - min(values)) / abs(median) if median != 0 else np.inf

def classify_stability(cv: float, range_ratio: float) -> str:
    """Classify parameter stability based on CV and Range."""
    if cv < 0.10 and range_ratio < 0.30:
        return "Excellent"
    elif cv < 0.15 and range_ratio < 0.60:
        return "Good"
    elif cv < 0.25 and range_ratio < 1.00:
        return "Moderate"
    else:
        return "Poor"

def classify_cv(cv: float) -> str:
    if cv < 0.10: return "Excellent"
    elif cv < 0.15: return "Good"
    elif cv < 0.25: return "Moderate"
    else: return "Poor"

def set_random_seed(seed: int = 2570):
    random.seed(seed)
    np.random.seed(seed)
  

def create_wf_config(training_days: int = 252,
                     testing_days: int = 63,
                     holdout_days: int = 21,
                     min_trades: int = 10,
                     n_trials: int = 100,
                     n_jobs: int = 1,
                     timeout: int = 3600,
                     random_seed: int = 42) -> Dict:
    return {
        'training_days': training_days,
        'testing_days': testing_days,
        'holdout_days': holdout_days,
        'min_trades': min_trades,
        'n_trials': n_trials,
        'n_jobs': n_jobs,
        'timeout': timeout,
        'random_seed': random_seed
    }


# ============================================================================
# DATA FILTERING
# ============================================================================

def filter_data_by_date(data_dict: Dict[str, pd.DataFrame],
                        start_date: datetime,
                        end_date: datetime) -> Dict[str, pd.DataFrame]:
    filtered = {}
    ts_start = pd.Timestamp(start_date)
    ts_end = pd.Timestamp(end_date)
    
    for ticker, df in data_dict.items():
        if df.empty:
            continue
        mask = (df.index >= ts_start) & (df.index <= ts_end)
        filtered_df = df[mask].copy()
        if not filtered_df.empty:
            filtered[ticker] = filtered_df
    return filtered


def get_date_range(data_dict: Dict[str, pd.DataFrame]) -> Tuple[datetime, datetime]:
    all_dates = set()
    for df in data_dict.values():
        if not df.empty:
            all_dates.update(df.index)
            
    if not all_dates:
        return datetime.now(), datetime.now()
        
    dates = sorted(all_dates)
    return dates[0], dates[-1]


# ============================================================================
# WINDOW GENERATION
# ============================================================================

def generate_windows(start_date: datetime,
                     end_date: datetime,
                     training_days: int,
                     testing_days: int,
                     holdout_days: int) -> List[Dict]:
    windows = []
    current_start = start_date
    
    while True:
        train_start = current_start
        train_end = train_start + timedelta(days=training_days)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=testing_days)
        holdout_start = test_end + timedelta(days=1)
        holdout_end = holdout_start + timedelta(days=holdout_days)
        
        if holdout_end > end_date:
            break
        
        windows.append({
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'holdout_start': holdout_start,
            'holdout_end': holdout_end
        })
        
        current_start = test_start
    
    return windows


# ============================================================================
# METRICS & BACKTESTING
# ============================================================================

def calculate_trade_metrics(trades_df: pd.DataFrame) -> Dict:
    # (Same implementation as fixed version)
    empty_metrics = {
        'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
        'win_rate': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0,
        'profit_factor': 0.0, 'expectancy': 0.0, 'total_pnl': 0.0
    }

    if trades_df is None or trades_df.empty:
        return empty_metrics
    
    # Logic to find PnL column
    pnl_col = None
    for col in ['net_pnl', 'pnl', 'profit', 'return', 'profit_loss']:
        if col in trades_df.columns:
            pnl_col = col
            break
    
    if pnl_col is None:
        return {**empty_metrics, 'total_trades': len(trades_df)}

    # Extract closed trades
    if 'action' in trades_df.columns:
        closed_trades = trades_df[trades_df['action'] == 'CLOSE'].copy()
    else:
        closed_trades = trades_df[trades_df[pnl_col].notna()].copy()

    if closed_trades.empty:
        return {**empty_metrics, 'total_trades': 0}
    
    pnl_values = closed_trades[pnl_col].replace([np.inf, -np.inf], np.nan).dropna()
    
    total_trades = len(pnl_values)
    if total_trades == 0: return empty_metrics

    winning_trades = len(pnl_values[pnl_values > 0])
    losing_trades = len(pnl_values[pnl_values < 0])
    win_rate = winning_trades / total_trades
    
    wins = pnl_values[pnl_values > 0]
    losses = pnl_values[pnl_values < 0]
    
    avg_win = wins.mean() if len(wins) > 0 else 0.0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
    
    total_wins = wins.sum() if len(wins) > 0 else 0.0
    total_losses = abs(losses.sum()) if len(losses) > 0 else 0.0
    
    profit_factor = total_wins / total_losses if total_losses > 0 else 999.0
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': min(profit_factor, 999.99),
        'expectancy': pnl_values.mean(),
        'total_pnl': pnl_values.sum()
    }


def calculate_returns_metrics(daily_values: pd.DataFrame, initial_capital: float) -> Dict:
    empty_metrics = {
        'total_return': 0.0, 'sharpe_ratio': 0.0, 'sortino_ratio': 0.0,
        'max_drawdown': 0.0, 'calmar_ratio': 0.0
    }

    if daily_values is None or daily_values.empty or 'total_value' not in daily_values.columns:
        return empty_metrics
    
    values = daily_values['total_value'].replace([np.inf, -np.inf], np.nan).dropna().values
    if len(values) < 2: return empty_metrics
    
    denom = values[:-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        returns = np.diff(values) / denom
        returns = returns[np.isfinite(returns)]
    
    if len(returns) == 0: return empty_metrics
    
    total_return = ((values[-1] - initial_capital) / initial_capital) * 100
    avg_return = np.mean(returns)
    std_return = np.std(returns, ddof=1) if len(returns) > 1 else 0.0
    sharpe_ratio = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0.0
    
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    with np.errstate(divide='ignore', invalid='ignore'):
        drawdown = (cumulative - running_max) / running_max
    drawdown = drawdown[np.isfinite(drawdown)]
    max_drawdown = abs(np.min(drawdown)) * 100 if len(drawdown) > 0 else 0.0
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': 0.0, # Simplified
        'max_drawdown': max_drawdown,
        'calmar_ratio': (total_return / max_drawdown) if max_drawdown > 0 else 0.0
    }


def run_backtest(data_dict: Dict[str, pd.DataFrame],
                params: Dict,
                strategy_class: type,
                strategy_name: str,
                trade_config: TradeConfig) -> Dict:
    if not data_dict:
        return {'error': 'No data'}

    try:
        strategy_config = StrategyConfig(name=strategy_name, parameters=params)
        strategy = strategy_class(strategy_config)
        engine = BacktestEngine(strategy, trade_config)
        results = engine.run(data_dict)
        
        trade_metrics = calculate_trade_metrics(results.trades)
        returns_metrics = calculate_returns_metrics(results.daily_values, trade_config.initial_capital)
        return {**trade_metrics, **returns_metrics}
    except Exception as e:
        return {'error': str(e), 'total_trades': 0, 'win_rate': 0.0, 'sharpe_ratio': 0.0}


# ============================================================================
# GLOBAL OPTIMIZATION (METHOD 2 CORE)
# ============================================================================

def calculate_objective(win_rate: float,
                       sharpe_ratio: float,
                       win_rate_weight: float = 0.35,
                       sharpe_weight: float = 0.65) -> float:
    normalized_sharpe = max(0, min(1, (sharpe_ratio + 3) / 6))
    return win_rate_weight * win_rate + sharpe_weight * normalized_sharpe


def create_global_objective_function(windows: List[Dict],
                                     data_dict: Dict[str, pd.DataFrame],
                                     param_space: Dict[str, Tuple],
                                     strategy_class: type,
                                     strategy_name: str,
                                     trade_config: TradeConfig,
                                     min_trades: int,
                                     objective_weights: Tuple[float, float]) -> Callable:
    
    def objective(trial: optuna.Trial) -> float:
        # 1. Suggest Parameters
        params = {}
        for param_name, (param_type, *args) in param_space.items():
            if param_type == 'int':
                params[param_name] = trial.suggest_int(param_name, args[0], args[1])
            elif param_type == 'float':
                params[param_name] = trial.suggest_float(param_name, args[0], args[1])
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(param_name, args[0])
        
        # 2. Cross-Validation Loop (Evaluate on ALL training windows)
        window_scores = []
        total_trades_all_windows = 0
        
        for i, window in enumerate(windows):
            # A. Filter Data for this specific window
            train_data = filter_data_by_date(data_dict, window['train_start'], window['train_end'])
            
            # Skip empty windows safely
            if not train_data:
                continue
            
            # B. Run Backtest
            results = run_backtest(train_data, params, strategy_class, strategy_name, trade_config)
            
            # Handle Errors (return terrible score if simulation fails)
            if 'error' in results:
                # Penalty for crashing
                window_scores.append(-1.0) 
                continue
                
            # C. Calculate Score (Weighted Sharpe + Win Rate)
            score = calculate_objective(
                results.get('win_rate', 0),
                results.get('sharpe_ratio', 0),
                objective_weights[0],
                objective_weights[1]
            )
            
            window_scores.append(score)
            total_trades_all_windows += results.get('total_trades', 0)

            # --- PRUNING LOGIC STARTS HERE ---
            # We report the current AVERAGE score across windows processed so far.
            # Step 'i' corresponds to the window index.
            current_avg_score = np.mean(window_scores)
            
            trial.report(current_avg_score, i)
            
            # Check if this trial should be pruned based on intermediate results
            if trial.should_prune():
                # Optional: Add custom tag to know it was pruned
                trial.set_user_attr("pruned_at_window", i)
                raise optuna.TrialPruned()
            # --- PRUNING LOGIC ENDS HERE ---

        # 3. Final Aggregation & Constraints
        if not window_scores:
            return -999.0
            
        avg_score = np.mean(window_scores)
        # Calculate average Max Drawdown across windows for this trial
        avg_max_dd = np.mean([res.get('max_drawdown', 0.0) for res in results_list])
        
        # Constraint: Penalty if total trades across ALL history is too low (inactive strategy)
        # We perform this check at the end because trade count varies per window.
        if total_trades_all_windows < (min_trades * len(window_scores) * 0.5):
            return -999.0

        # Store detailed stats for the successful trial
        trial.set_user_attr('avg_score', avg_score)
        trial.set_user_attr('min_score', np.min(window_scores))
        trial.set_user_attr('std_score', np.std(window_scores))
        trial.set_user_attr('max_drawdown', avg_max_dd)
        trial.set_user_attr('avg_score', avg_score)
        
        return avg_score

    return objective


def optimize_global_parameters(windows: List[Dict],
                               data_dict: Dict[str, pd.DataFrame],
                               param_space: Dict[str, Tuple],
                               strategy_class: type,
                               strategy_name: str,
                               trade_config: TradeConfig,
                               wf_config: Dict,
                               objective_weights: Tuple[float, float]) -> Tuple[Dict, optuna.Study]:
    
    print(f"\nRunning Global Optimization (Cross-Validation on {len(windows)} windows) with Pruning...")
    
    set_random_seed(wf_config['random_seed'])
    
    objective_fn = create_global_objective_function(
        windows, data_dict, param_space, strategy_class, strategy_name,
        trade_config, wf_config['min_trades'], objective_weights
    )
    
    # --- PRUNER CONFIGURATION ---
    # MedianPruner: Prunes if the trial's intermediate value is worse than the 
    # median of intermediate values of previous trials at the same step.
    # n_startup_trials: Run 5 trials blindly before starting to prune (to build a baseline).
    # n_warmup_steps: Don't prune a trial until it has completed at least 3 windows 
    # (prevents pruning just because the first window was a bad market period).
    pruner = MedianPruner(
        n_startup_trials=5, 
        n_warmup_steps=3, 
        interval_steps=1
    )

    sampler = TPESampler(seed=wf_config['random_seed'])

    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner, # <--- Inject Pruner Here
        study_name=f'{strategy_name}_global_cv'
    )
    
    study.optimize(
        objective_fn,
        n_trials=wf_config['n_trials'],
        timeout=wf_config['timeout'],
        n_jobs=wf_config['n_jobs'],
        show_progress_bar=True
    )
    
    print(f"Global Optimization Complete. Best Score: {study.best_value:.4f}")
    
    # Optional: Print pruning statistics
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"  Trials completed: {len(complete_trials)}")
    print(f"  Trials pruned:    {len(pruned_trials)}")
    
    return study.best_params, study



# ============================================================================
# MAIN ANALYSIS (METHOD 2)
# ============================================================================

def print_result_row(prefix, res):
    print(f"  {prefix:10s} | Trades: {res.get('total_trades',0):4d} | "
          f"Win: {res.get('win_rate',0):6.2%} | "
          f"Sharpe: {res.get('sharpe_ratio',0):6.4f} | "
          f"Ret: {res.get('total_return',0):6.2f}%")


def walk_forward_analysis(data_dict: Dict[str, pd.DataFrame],
                         param_space: Dict[str, Tuple],
                         strategy_class: type,
                         strategy_name: str,
                         trade_config: TradeConfig,
                         wf_config: Dict,
                         objective_weights: Tuple[float, float] = (0.35, 0.65),
                         verbose: bool = False) -> Tuple[pd.DataFrame, optuna.Study]:
    
    set_random_seed(wf_config.get('random_seed', 42))
    
    start_date, end_date = get_date_range(data_dict)
    
    # 1. Generate Windows
    windows = generate_windows(
        start_date, end_date,
        wf_config['training_days'],
        wf_config['testing_days'],
        wf_config['holdout_days']
    )
    
    if verbose:
        print(f"\n{'='*80}\nGLOBAL WALK-FORWARD ANALYSIS (Method 2)\n{'='*80}")
        print(f"Strategy:      {strategy_name}")
        print(f"Total Windows: {len(windows)}")
        print(f"Approach:      Global Optimization (One param set for all windows)")

    # 2. Global Optimization (The key difference in Method 2)
    #    We find ONE set of parameters that works best on average across ALL training windows.
    best_params, study = optimize_global_parameters(
        windows, data_dict, param_space, strategy_class, strategy_name,
        trade_config, wf_config, objective_weights
    )
    
    if verbose:
        print(f"\nSelected Global Parameters:")
        print(best_params)
       
    # 3. Apply Global Params to Out-of-Sample Windows
    #    Now we simulate how this "Robust" strategy would have performed.
    results = []
    
    for i, window in enumerate(windows, 1):
        if verbose:
            print(f"\nWindow {i}: {window['test_start'].date()} -> {window['test_end'].date()}")

        # A. Training Performance (In-Sample Check)
        train_data = filter_data_by_date(data_dict, window['train_start'], window['train_end'])
        train_res = run_backtest(train_data, best_params, strategy_class, strategy_name, trade_config)
        
        # B. Test Performance (Out-of-Sample)
        test_data = filter_data_by_date(data_dict, window['test_start'], window['test_end'])
        test_res = run_backtest(test_data, best_params, strategy_class, strategy_name, trade_config)
        
        # C. Holdout Performance (Out-of-Sample)
        holdout_data = filter_data_by_date(data_dict, window['holdout_start'], window['holdout_end'])
        holdout_res = run_backtest(holdout_data, best_params, strategy_class, strategy_name, trade_config)

        if verbose:
            if 'error' not in test_res: print_result_row("TEST", test_res)
            if 'error' not in holdout_res: print_result_row("HOLDOUT", holdout_res)

        # Build Result Dict
        row = {
            'window': i,
            'train_start': window['train_start'],
            'test_start': window['test_start'],
            'global_optimization_score': study.best_value
        }
        
        # Add params (same for every row, but useful for CSV export)
        row.update({f'best_{k}': v for k, v in best_params.items()})
        
        # Add Metrics
        row.update({f'train_{k}': v for k, v in train_res.items() if k != 'error'})
        row.update({f'test_{k}': v for k, v in test_res.items() if k != 'error'})
        row.update({f'holdout_{k}': v for k, v in holdout_res.items() if k != 'error'})
        
        results.append(row)

    results_df = pd.DataFrame(results)
    
    if verbose and not results_df.empty:
        print_summary(results_df)

    return results_df, study


# ============================================================================
# REPORTING UTILITIES
# ============================================================================

def print_summary(results_df: pd.DataFrame):
    print(f"\n{'='*80}")
    print("GLOBAL ROBUSTNESS SUMMARY")
    print(f"{'='*80}")
    
    # Calculate averages ignoring NaNs
    print("\nOUT-OF-SAMPLE (TEST) AVERAGE:")
    cols = ['test_win_rate', 'test_sharpe_ratio', 'test_total_return', 'test_max_drawdown']
    if all(c in results_df.columns for c in cols):
        print(f"  Win Rate:         {results_df['test_win_rate'].mean():.2%}")
        print(f"  Sharpe Ratio:     {results_df['test_sharpe_ratio'].mean():.4f}")
        print(f"  Avg Return:       {results_df['test_total_return'].mean():.2f}%")
        print(f"  Avg Drawdown:     {results_df['test_max_drawdown'].mean():.2f}%")
        print(f"  Positive Windows: {(results_df['test_total_return'] > 0).sum()}/{len(results_df)}")

    print("\nHOLDOUT VALIDATION AVERAGE:")
    cols_h = ['holdout_win_rate', 'holdout_sharpe_ratio', 'holdout_total_return', 'holdout_max_drawdown']
    if all(c in results_df.columns for c in cols_h):
        print(f"  Win Rate:         {results_df['holdout_win_rate'].mean():.2%}")
        print(f"  Sharpe Ratio:     {results_df['holdout_sharpe_ratio'].mean():.4f}")
        print(f"  Avg Return:       {results_df['holdout_total_return'].mean():.2f}%")
        print(f"  Avg Drawdown:     {results_df['holdout_max_drawdown'].mean():.2f}%")


def analyze_cluster_stability(study: optuna.Study, top_n: int = 20) -> pd.DataFrame:
    """
    Adapted for Method 2: Checks if the top performing trials converge 
    on similar parameter values (Cluster Stability).
    """
    # 1. Get Top N Trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    # Sort by value (descending for maximize)
    sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:top_n]
    
    if len(sorted_trials) < 2:
        return pd.DataFrame()

    # 2. Extract Data
    param_names = sorted_trials[0].params.keys()
    results = []

    for param in param_names:
        # Collect values for this specific parameter across top trials
        values = [t.params[param] for t in sorted_trials]
        
        # Skip stability check for categorical parameters (strings/booleans)
        if isinstance(values[0], str):
            results.append({
                'parameter': param,
                'type': 'categorical',
                'mode': max(set(values), key=values.count),
                'stability_score': np.nan,
                'assessment': 'N/A (Categorical)'
            })
            continue

        # Basic Stats
        mean = np.mean(values)
        median = np.median(values)
        std = np.std(values, ddof=1)
        
        # Your Stability Metrics
        cv = calculate_cv(values)
        range_ratio = calculate_range_ratio(values)
        
        # Composite Score (Your Formula)
        cv_score = max(0, 100 * (1 - cv / 0.40))
        range_score = max(0, 100 * (1 - range_ratio / 1.00))
        stability_score = 0.6 * cv_score + 0.4 * range_score
        
        results.append({
            'parameter': param,
            'type': 'numerical',
            'mean': round(mean, 3),
            'median': round(median, 3),
            'std': round(std, 3),
            'cv': round(cv, 3),
            'range_ratio': round(range_ratio, 3),
            'stability_score': round(stability_score, 1),
            'assessment': classify_stability(cv, range_ratio)
        })
        
    return pd.DataFrame(results).sort_values('stability_score', ascending=False)
def plot_parameter_heatmap(study: optuna.Study, param_x: str, param_y: str):
    """
    Visualizes the 'Optimization Landscape' using your requested heatmap style.
    Adapts trial data to create the matrix.
    """
    import matplotlib.tri as tri

    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    # Extract data
    x = [t.params[param_x] for t in trials]
    y = [t.params[param_y] for t in trials]
    z = [t.value for t in trials] # Objective Score (Sharpe)

    # Convert to numpy
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create grid via triangulation (since Optuna points are scattered, not a perfect grid)
    triang = tri.Triangulation(x, y)
    
    # Plot contour/heatmap
    cntr = ax.tricontourf(triang, z, levels=14, cmap="RdYlGn")
    fig.colorbar(cntr, ax=ax, label="Objective Score")

    # Mark Best Parameter
    best_x = study.best_params[param_x]
    best_y = study.best_params[param_y]
    ax.scatter(best_x, best_y, marker='*', s=500, c='blue', edgecolors='black', label='Best Params', zorder=5)

    ax.set_title(f"Sensitivity Surface: {param_x} vs {param_y}")
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.legend()
    plt.show()

def plot_drawdown_sensitivity(study: optuna.Study, param_name: str):
    """
    Plots Parameter Value vs. Max Drawdown (User Attribute).
    Requires 'max_drawdown' to be saved in trial.user_attrs.
    """
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    # Extract data
    param_values = []
    drawdowns = []
    
    for t in trials:
        if param_name in t.params and 'max_drawdown' in t.user_attrs:
            param_values.append(t.params[param_name])
            # Ensure drawdown is negative percent for display (e.g. -0.15)
            dd = t.user_attrs['max_drawdown']
            drawdowns.append(dd * 100 if abs(dd) < 1.0 else dd) 

    if not param_values:
        print(f"No drawdown data found for sensitivity plot of {param_name}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Sort for cleaner line plotting (if continuous)
    sorted_pairs = sorted(zip(param_values, drawdowns))
    p_sorted, d_sorted = zip(*sorted_pairs)

    ax.scatter(p_sorted, d_sorted, alpha=0.6, c='red', edgecolors='k')
    
    # Trend line (Polynomial fit)
    try:
        z = np.polyfit(p_sorted, d_sorted, 2)
        p = np.poly1d(z)
        ax.plot(p_sorted, p(p_sorted), "b--", alpha=0.8, label="Trend")
    except:
        pass

    median_dd = np.median(d_sorted)
    ax.axhline(median_dd, color='blue', linestyle=':', label=f'Median DD ({median_dd:.1f}%)')

    ax.set_xlabel(f'{param_name} Value')
    ax.set_ylabel('Maximum Drawdown (%)')
    ax.set_title(f'Risk Sensitivity: {param_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.show()
# ============================================================================
# ANALYSIS UTILITIES (Fixed for Single Global Study)
# ============================================================================

def calculate_param_importance(study: optuna.Study) -> pd.DataFrame:
    """
    Calculates parameter importance for the single global optimization study.
    """
    try:
        # Check if study has enough completed trials
        valid_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(valid_trials) < 2:
            return pd.DataFrame()
            
        importance = optuna.importance.get_param_importances(study)
        
        # Convert to DataFrame
        importance_df = pd.DataFrame(list(importance.items()), columns=['parameter', 'importance'])
        return importance_df.sort_values('importance', ascending=False)
        
    except Exception as e:
        print(f"Could not calculate parameter importance: {e}")
        return pd.DataFrame()


def get_top_trials_distribution(study: optuna.Study, top_n: int = 10) -> pd.DataFrame:
    """
    Analyzes the distribution of parameters across the top N performing trials.
    This helps identify if the "best" parameters are outliers or part of a cluster.
    """
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    
    # Sort by value (assuming maximize direction)
    sorted_trials = sorted(trials, key=lambda t: t.value, reverse=True)
    top_trials = sorted_trials[:top_n]
    
    if not top_trials:
        return pd.DataFrame()
    
    data = []
    for t in top_trials:
        row = {'objective_value': t.value, 'trial_number': t.number}
        row.update(t.params)
        data.append(row)
        
    return pd.DataFrame(data)


# ============================================================================
# EXPORT RESULTS (Fixed for Single Global Study)
# ============================================================================


import os
import pandas as pd
import optuna

def export_results(results_df: pd.DataFrame,
                   study: optuna.Study,
                   sensitivity_df: pd.DataFrame = None,
                   output_dir: str = "walk_forward_global_results"):
    """
    Exports results, parameter importance, and optimization history.
    
    Args:
        results_df: The main walk-forward results.
        study: The Optuna study object (Method 2 Global Study).
        sensitivity_df: (Optional) The parameter importance DataFrame.
        output_dir: The folder path to save results (Must be a string).
    """
    
    # --- SAFETY CHECK ---
    # This prevents the specific error you saw. 
    # If the user accidentally passes the study object as the 3rd or 4th argument,
    # we force output_dir to be a string.
    if not isinstance(output_dir, str):
        print(f"Warning: 'output_dir' received non-string object ({type(output_dir)}). Using default path.")
        output_dir = "walk_forward_global_results"

    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Export Walk-Forward Performance
    if not results_df.empty:
        path = os.path.join(output_dir, "wf_global_results.csv")
        results_df.to_csv(path, index=False)
        print(f"Results exported to: {path}")
    
    # 2. Export Parameter Importance (Sensitivity)
    # If sensitivity_df was passed in, save it. 
    if sensitivity_df is not None and not sensitivity_df.empty:
        path = os.path.join(output_dir, "global_param_importance.csv")
        sensitivity_df.to_csv(path, index=False)
        print(f"Parameter importance exported to: {path}")
    
    # 3. Export Optimization History (From the Study object)
    try:
        # We manually construct this to avoid potential serialization errors
        history_data = []
        for trial in study.trials:
            row = {
                'number': trial.number,
                'value': trial.value,
                'state': trial.state.name,
                'datetime_start': trial.datetime_start,
                'datetime_complete': trial.datetime_complete
            }
            # Flatten params
            for k, v in trial.params.items():
                row[f'param_{k}'] = v
            # Flatten user attributes (metrics stored during optimization)
            for k, v in trial.user_attrs.items():
                row[f'metric_{k}'] = v
                
            history_data.append(row)
            
        history_df = pd.DataFrame(history_data)
        path = os.path.join(output_dir, "global_optimization_history.csv")
        history_df.to_csv(path, index=False)
        print(f"Optimization history exported to: {path}")
        
    except Exception as e:
        print(f"Could not export optimization history: {e}")


def analyze_all_params_stability(results_df: pd.DataFrame) -> pd.DataFrame:
    param_cols = [col for col in results_df.columns if col.startswith('best_')]
    
    stability_results = []
    for col in param_cols:
        param_name = col.replace('best_', '')
        stability = analyze_param_stability(results_df, param_name)
        if stability:
            stability_results.append(stability)
    
    return pd.DataFrame(stability_results).sort_values('cv') if stability_results else pd.DataFrame()


def analyze_param_stability(results_df: pd.DataFrame, param_name: str) -> Dict:
    param_col = f'best_{param_name}'
    if param_col not in results_df.columns:
        return {}
    
    values = results_df[param_col].dropna()
    
    if values.empty:
        return {}
    
    # FIX: Robust check for categorical parameters
    if not pd.api.types.is_numeric_dtype(values):
        # Return simplified info for categorical/string parameters
        return {
            'parameter': param_name,
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'min': values.min() if not values.empty else np.nan,
            'max': values.max() if not values.empty else np.nan,
            'cv': np.nan,
            'trend_slope': 0.0,
            'trend_r2': 0.0,
            'is_stable': True  # Cannot determine stability mathematically for categories
        }
    
    mean_val = values.mean()
    std_val = values.std()
    cv = (std_val / mean_val) if mean_val != 0 else 0.0
    
    # Perform linear regression only on numeric data
    x = np.arange(len(values))
    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
        trend_r2 = r_value**2
    except Exception:
        slope, trend_r2 = 0.0, 0.0
    
    return {
        'parameter': param_name,
        'mean': mean_val,
        'median': values.median(),
        'std': std_val,
        'min': values.min(),
        'max': values.max(),
        'cv': cv,
        'trend_slope': slope,
        'trend_r2': trend_r2,
        'is_stable': cv < 0.3
    }


def aggregate_param_importance(sensitivity_df: pd.DataFrame) -> pd.DataFrame:
    if sensitivity_df.empty:
        return pd.DataFrame()
    
    return sensitivity_df.groupby('parameter')['importance_score'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('min', 'min'),
        ('max', 'max'),
        ('count', 'count')
    ]).sort_values('mean', ascending=False)

def get_param_distributions(studies: List[optuna.Study], top_n: int = 5) -> Dict[str, Dict]:
    sensitivity_df = calculate_param_importance(studies)
    if sensitivity_df.empty:
        return {}
    
    importance = aggregate_param_importance(sensitivity_df)
    top_params = importance.head(top_n).index.tolist()
    
    distributions = {}
    for param in top_params:
        values = []
        for study in studies:
            if param in study.best_params:
                values.append(study.best_params[param])
        
        if values:
            distributions[param] = {
                'mean': np.mean(values) if pd.api.types.is_numeric_dtype(values) else np.nan,
                'std': np.std(values) if pd.api.types.is_numeric_dtype(values) else np.nan,
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values) if pd.api.types.is_numeric_dtype(values) else np.nan
            }
    
    return distributions

def run_walkforward(data_dict: Dict[str, pd.DataFrame],
                          param_space: Dict[str, Tuple],
                          strategy_class: type,
                          strategy_name: str,
                          trade_config: TradeConfig,
                          wf_config: Dict = None,
                          objective_weights: Tuple[float, float] = (0.9, 0.1),
                          verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    if wf_config is None:
        wf_config = create_wf_config()
    
    results_df, global_study = walk_forward_analysis(
        data_dict=data_dict,
        param_space=param_space,
        strategy_class=strategy_class,
        strategy_name=strategy_name,
        trade_config=trade_config,
        wf_config=wf_config,
        objective_weights=objective_weights,
        verbose=verbose
    )
    
    stability_df = analyze_cluster_stability(global_study, top_n=20)
    print("\n--- Parameter Stability Analysis (Top 20 Trials) ---")
    print(stability_df[['parameter', 'cv', 'range_ratio', 'assessment', 'stability_score']])
    # 3. Generate Visualizations
    # Find the two most sensitive/important parameters to plot
    if not stability_df.empty:
        # Plot Heatmap for top 2 params
        top_params = stability_df['parameter'].head(2).tolist()
        if len(top_params) == 2:
            plot_parameter_heatmap(global_study, top_params[0], top_params[1])
    
        # Plot Drawdown Sensitivity for the #1 most unstable parameter
        # (We want to see why it is unstable!)
        most_unstable_param = stability_df.iloc[-1]['parameter']
        plot_drawdown_sensitivity(global_study, most_unstable_param)

    
    sensitivity_df = calculate_param_importance(global_study)
    
    if verbose and not sensitivity_df.empty:
        print(f"\n{'='*80}")
        print("PARAMETER IMPORTANCE")
        print(f"{'='*80}")
        importance = aggregate_param_importance(sensitivity_df)
        print("\n", importance.to_string())
        
        distributions = get_param_distributions(global_study, top_n=5)
        if distributions:
            print("\n\nTOP PARAMETER DISTRIBUTIONS:")
            print("-" * 60)
            for param, stats in distributions.items():
                print(f"\n{param}:")
                print(f"  Mean:   {stats['mean']:.4f}")
                print(f"  Median: {stats['median']:.4f}")
                print(f"  Std:    {stats['std']:.4f}")
                print(f"  Range:  [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    export_results(
        results_df=results_df,
        study=global_study,
        sensitivity_df=sensitivity_df, 
        output_dir="."
    )

    
    return results_df, sensitivity_df

