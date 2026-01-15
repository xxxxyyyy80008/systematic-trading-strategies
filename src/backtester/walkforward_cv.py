"""
Global Walk-Forward Analysis
------------------------------------------------------------------
Optimizes a single set of parameters across all training windows (Cross-Validation)
and tests the robustness of these static parameters on Out-of-Sample data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.importance import MeanDecreaseImpurityImportanceEvaluator

import warnings
import random
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Callable, Any
from scipy import stats

warnings.filterwarnings('ignore')

# Assumed internal modules (Mocking logic handled if missing)
try:
    from .backtest_engine import BacktestEngine
    from .types_core import StrategyConfig, TradeConfig
except ImportError:
    pass 

# ============================================================================
# 1. MATH & STATS HELPERS
# ============================================================================

def set_random_seed(seed: int = 2570):
    random.seed(seed)
    np.random.seed(seed)

def calculate_cv(values: List[float]) -> float:
    """Calculate coefficient of variation (Standard Deviation / Mean)."""
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

def create_wf_config(training_days: int = 252,
                     testing_days: int = 63,
                     holdout_days: int = 21,
                     min_trades: int = 10,
                     n_trials: int = 100,
                     n_startup_trials: int = 20, 
                     n_jobs: int = 1,
                     timeout: int = 3600,
                     random_seed: int = 42) -> Dict:
    return {
        'training_days': training_days,
        'testing_days': testing_days,
        'holdout_days': holdout_days,
        'min_trades': min_trades,
        'n_trials': n_trials,
        'n_startup_trials': n_startup_trials, 
        'n_jobs': n_jobs,
        'timeout': timeout,
        'random_seed': random_seed
    }

# ============================================================================
# 2. DATA & WINDOW MANAGEMENT
# ============================================================================

def filter_data_by_date(data_dict: Dict[str, pd.DataFrame],
                        start_date: datetime,
                        end_date: datetime) -> Dict[str, pd.DataFrame]:
    filtered = {}
    ts_start = pd.Timestamp(start_date)
    ts_end = pd.Timestamp(end_date)
    
    for ticker, df in data_dict.items():
        if df.empty: continue
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
# 3. METRICS CALCULATION
# ============================================================================

def calculate_trade_metrics(trades_df: pd.DataFrame) -> Dict:
    empty_metrics = {
        'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
        'win_rate': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0,
        'profit_factor': 0.0, 'expectancy': 0.0, 'total_pnl': 0.0
    }
    if trades_df is None or trades_df.empty: return empty_metrics
    
    # Logic to find PnL column
    pnl_col = next((col for col in ['net_pnl', 'pnl', 'profit', 'return', 'profit_loss'] if col in trades_df.columns), None)
    if pnl_col is None: return {**empty_metrics, 'total_trades': len(trades_df)}

    pnl_values = trades_df[pnl_col].replace([np.inf, -np.inf], np.nan).dropna()
    total_trades = len(pnl_values)
    if total_trades == 0: return empty_metrics

    wins = pnl_values[pnl_values > 0]
    losses = pnl_values[pnl_values < 0]
    
    total_wins = wins.sum()
    total_losses = abs(losses.sum())
    profit_factor = total_wins / total_losses if total_losses > 0 else 999.0
    
    return {
        'total_trades': total_trades,
        'winning_trades': len(wins),
        'losing_trades': len(losses),
        'win_rate': len(wins) / total_trades,
        'avg_win': wins.mean() if len(wins) > 0 else 0.0,
        'avg_loss': abs(losses.mean()) if len(losses) > 0 else 0.0,
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
    
    values = daily_values['total_value'].values
    if len(values) < 2: return empty_metrics
    
    # Calculate returns safely
    denom = values[:-1]
    returns = np.divide(np.diff(values), denom, out=np.zeros_like(denom), where=denom!=0)
    
    total_return = ((values[-1] - initial_capital) / initial_capital) * 100
    avg_return = np.mean(returns)
    std_return = np.std(returns, ddof=1) if len(returns) > 1 else 0.0
    sharpe_ratio = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0.0
    
    # Drawdown
    cumulative = np.maximum.accumulate(values)
    drawdown = (values - cumulative) / cumulative
    max_drawdown = abs(np.min(drawdown)) * 100
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': 0.0, 
        'max_drawdown': max_drawdown,
        'calmar_ratio': (total_return / max_drawdown) if max_drawdown > 0 else 0.0
    }

def run_backtest(data_dict: Dict[str, pd.DataFrame],
                params: Dict,
                strategy_class: type,
                strategy_name: str,
                trade_config: TradeConfig) -> Dict:
    if not data_dict: return {'error': 'No data'}

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

def calculate_degradation(train_val: float, test_val: float) -> float:
    """
    Calculates Percentage Degradation (Overfitting).
    Formula: (InSample - OutSample) / InSample
    
    Interpretation:
    - 0.0 to 0.20: Excellent Robustness (Little drop)
    - > 0.50:      High Overfitting (Performance halved)
    - < 0.0:       Underfitting or Regime Shift (Test was better than Train)
    """
    if train_val == 0.0:
        return 0.0
    
    # Standard Degradation Formula
    degradation = (train_val - test_val) / train_val
    
    # Edge Case Correction:
    # If Train was negative (loss) and Test was worse (bigger loss), 
    # the standard formula gives a negative number (looks like improvement).
    # We fix this by ensuring degradation is positive if things got worse.
    if train_val < 0 and test_val < train_val:
        return abs(degradation)
        
    return degradation

# ============================================================================
# 4. OPTIMIZATION CORE (METHOD 2)
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
        
        # 2. Cross-Validation Loop
        window_scores = []
        results_list = [] # <--- FIX: Initialize list to store full results
        total_trades_all_windows = 0
        
        for i, window in enumerate(windows):
            train_data = filter_data_by_date(data_dict, window['train_start'], window['train_end'])
            if not train_data: continue
            
            # Run Backtest
            results = run_backtest(train_data, params, strategy_class, strategy_name, trade_config)
            results_list.append(results) 
            
            if 'error' in results:
                window_scores.append(-1.0)
                continue
                
            score = calculate_objective(
                results.get('win_rate', 0),
                results.get('sharpe_ratio', 0),
                objective_weights[0],
                objective_weights[1]
            )
            
            window_scores.append(score)
            total_trades_all_windows += results.get('total_trades', 0)

            # Pruning Logic
            current_avg_score = np.mean(window_scores)
            trial.report(current_avg_score, i)
            if trial.should_prune():
                trial.set_user_attr("pruned_at_window", i)
                raise optuna.TrialPruned()

        # 3. Final Aggregation
        if not window_scores: return -999.0
            
        avg_score = np.mean(window_scores)
        
        # Calculate Average Drawdown safely
        avg_max_dd = 0.0
        if results_list:
             avg_max_dd = np.mean([res.get('max_drawdown', 0.0) for res in results_list])

        # Activity Constraint
        if total_trades_all_windows < (min_trades * len(window_scores) * 0.5):
            return -999.0

        # Store attributes for analysis
        trial.set_user_attr('avg_score', avg_score)
        trial.set_user_attr('min_score', np.min(window_scores))
        trial.set_user_attr('std_score', np.std(window_scores))
        trial.set_user_attr('max_drawdown', avg_max_dd)
        
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
    
    print(f"\nRunning Global Optimization on {len(windows)} windows...")
    set_random_seed(wf_config['random_seed'])
    
    objective_fn = create_global_objective_function(
        windows, data_dict, param_space, strategy_class, strategy_name,
        trade_config, wf_config['min_trades'], objective_weights
    )
    
    # Pruner Configuration
    pruner = MedianPruner(n_startup_trials=wf_config['n_startup_trials'], n_warmup_steps=3, interval_steps=1)
    sampler = TPESampler(seed=wf_config['random_seed'])

    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
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
    print(study.best_params)
    return study.best_params, study

# ============================================================================
# 5. ANALYSIS & VISUALIZATION
# ============================================================================

def analyze_cluster_stability(study: optuna.Study, top_n: int = 20) -> pd.DataFrame:
    """Checks if top trials converge on similar parameters (Cluster Stability)."""
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    sorted_trials = sorted(completed_trials, key=lambda t: t.value, reverse=True)[:top_n]
    
    if len(sorted_trials) < 2: return pd.DataFrame()

    param_names = sorted_trials[0].params.keys()
    results = []

    for param in param_names:
        values = [t.params[param] for t in sorted_trials]
        
        if isinstance(values[0], str): continue # Skip categorical

        mean = np.mean(values)
        cv = calculate_cv(values)
        range_ratio = calculate_range_ratio(values)
        
        # Composite Stability Score
        cv_score = max(0, 100 * (1 - cv / 0.40))
        range_score = max(0, 100 * (1 - range_ratio / 1.00))
        stability_score = 0.6 * cv_score + 0.4 * range_score
        
        results.append({
            'parameter': param,
            'mean': round(mean, 3),
            'cv': round(cv, 3),
            'range_ratio': round(range_ratio, 3),
            'stability_score': round(stability_score, 1),
            'assessment': classify_stability(cv, range_ratio)
        })
        
    return pd.DataFrame(results).sort_values('stability_score', ascending=False)

def calculate_param_importance(study: optuna.Study) -> pd.DataFrame:
    """
    Calculates parameter importance using Mean Decrease Impurity (MDI).
    
    Fixes the 'setting an array element with a sequence' bug by:
    1. Using Random Forest (MDI) evaluator instead of fANOVA (which crashes on constants).
    2. filtering out parameters that did not vary during the trials.
    """
    try:
        # 1. Filter for completed trials
        valid_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(valid_trials) < 2: 
            return pd.DataFrame()

        # 2. Identify parameters that actually vary
        # Evaluators often crash if a parameter has the same value in ALL trials.
        param_values = {}
        for t in valid_trials:
            for p, v in t.params.items():
                if p not in param_values: param_values[p] = set()
                param_values[p].add(v)
        
        # Only keep params with > 1 unique value
        variable_params = [p for p, vals in param_values.items() if len(vals) > 1]
        
        if not variable_params:
            print("Warning: No parameters varied significantly during optimization. Skipping importance.")
            return pd.DataFrame()

        # 3. Calculate Importance using Random Forest (MDI)
        # MDI is robust to static parameters and mixed types.
        evaluator = MeanDecreaseImpurityImportanceEvaluator()
        
        importance = optuna.importance.get_param_importances(
            study, 
            evaluator=evaluator,
            params=variable_params # Explicitly pass only varying params
        )
        
        # 4. Format Output
        df = pd.DataFrame(list(importance.items()), columns=['parameter', 'importance'])
        return df.sort_values('importance', ascending=False)

    except Exception as e:
        print(f"Param importance error: {e}")
        # Return empty DF so the script continues gracefully
        return pd.DataFrame()


def plot_parameter_heatmap(study: optuna.Study, param_x: str, param_y: str):
    """Visualizes 2D Parameter Sensitivity."""
    try:
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not trials: return

        x = np.array([t.params[param_x] for t in trials])
        y = np.array([t.params[param_y] for t in trials])
        z = np.array([t.value for t in trials])

        fig, ax = plt.subplots(figsize=(10, 8))
        triang = tri.Triangulation(x, y)
        cntr = ax.tricontourf(triang, z, levels=14, cmap="RdYlGn")
        fig.colorbar(cntr, ax=ax, label="Objective Score")

        best_x = study.best_params[param_x]
        best_y = study.best_params[param_y]
        ax.scatter(best_x, best_y, marker='*', s=500, c='blue', edgecolors='white', label='Best Params', zorder=5)

        ax.set_title(f"Sensitivity Surface: {param_x} vs {param_y}")
        ax.set_xlabel(param_x)
        ax.set_ylabel(param_y)
        ax.legend()
        plt.show()
    except Exception as e:
        print(f"Could not plot heatmap: {e}")

def plot_drawdown_sensitivity(study: optuna.Study, param_name: str):
    """Plots Parameter Value vs. Max Drawdown."""
    try:
        trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        param_values, drawdowns = [], []
        
        for t in trials:
            if param_name in t.params and 'max_drawdown' in t.user_attrs:
                param_values.append(t.params[param_name])
                dd = t.user_attrs['max_drawdown']
                drawdowns.append(dd) # Assuming already percentage

        if not param_values: return

        fig, ax = plt.subplots(figsize=(10, 6))
        sorted_pairs = sorted(zip(param_values, drawdowns))
        p_sorted, d_sorted = zip(*sorted_pairs)

        ax.scatter(p_sorted, d_sorted, alpha=0.6, c='red', edgecolors='k')
        
        try:
            z = np.polyfit(p_sorted, d_sorted, 2)
            p = np.poly1d(z)
            ax.plot(p_sorted, p(p_sorted), "b--", alpha=0.8, label="Trend")
        except: pass

        ax.set_xlabel(f'{param_name} Value')
        ax.set_ylabel('Avg Max Drawdown (%)')
        ax.set_title(f'Risk Sensitivity: {param_name}')
        plt.show()
    except Exception as e:
        print(f"Could not plot sensitivity: {e}")

# ============================================================================
# 6. EXPORT UTILITIES
# ============================================================================

def export_results(results_df: pd.DataFrame,
                   study: optuna.Study,
                   sensitivity_df: pd.DataFrame = None,
                   output_dir: str = "walk_forward_global_results"):
    
    if not isinstance(output_dir, str): output_dir = "walk_forward_global_results"
    os.makedirs(output_dir, exist_ok=True)
    
    if not results_df.empty:
        results_df.to_csv(os.path.join(output_dir, "wf_global_results.csv"), index=False)
    
    if sensitivity_df is not None and not sensitivity_df.empty:
        sensitivity_df.to_csv(os.path.join(output_dir, "global_param_importance.csv"), index=False)
    
    # Export History
    try:
        history_data = []
        for trial in study.trials:
            row = {'number': trial.number, 'value': trial.value, 'state': trial.state.name}
            row.update({f'param_{k}': v for k, v in trial.params.items()})
            row.update({f'metric_{k}': v for k, v in trial.user_attrs.items()})
            history_data.append(row)
        pd.DataFrame(history_data).to_csv(os.path.join(output_dir, "global_optimization_history.csv"), index=False)
        print(f"Results saved to {output_dir}")
    except Exception as e:
        print(f"Export error: {e}")

# ============================================================================
# 7. MAIN CONTROLLER
# ============================================================================

def walk_forward_analysis(data_dict: Dict[str, pd.DataFrame],
                         param_space: Dict[str, Tuple],
                         strategy_class: type,
                         strategy_name: str,
                         trade_config: TradeConfig,
                         wf_config: Dict,
                         objective_weights: Tuple[float, float] = (0.35, 0.65),
                         verbose: bool = False) -> Tuple[pd.DataFrame, optuna.Study]:
    
    start_date, end_date = get_date_range(data_dict)
    windows = generate_windows(start_date, end_date, 
                               wf_config['training_days'], wf_config['testing_days'], wf_config['holdout_days'])
    
    if verbose:
        print(f"Global Walk-Forward: {len(windows)} windows from {start_date.date()} to {end_date.date()}")

    # 1. GLOBAL OPTIMIZATION
    best_params, study = optimize_global_parameters(
        windows, data_dict, param_space, strategy_class, strategy_name,
        trade_config, wf_config, objective_weights
    )
    
    # 2. OOS TESTING & DEGRADATION ANALYSIS
    results = []
    for i, window in enumerate(windows, 1):
        # A. Filter Data
        train_data = filter_data_by_date(data_dict, window['train_start'], window['train_end'])
        test_data = filter_data_by_date(data_dict, window['test_start'], window['test_end'])
        holdout_data = filter_data_by_date(data_dict, window['holdout_start'], window['holdout_end'])
        
        # B. Run Backtests
        # Note: We re-run Train here to get exact metrics for this specific window
        # (The optimizer gave us an average, but we want exact window-by-window comparison)
        train_res = run_backtest(train_data, best_params, strategy_class, strategy_name, trade_config)
        test_res = run_backtest(test_data, best_params, strategy_class, strategy_name, trade_config)
        holdout_res = run_backtest(holdout_data, best_params, strategy_class, strategy_name, trade_config)
        
        # C. Calculate Degradation (The new part)
        deg_sharpe = calculate_degradation(train_res.get('sharpe_ratio', 0), test_res.get('sharpe_ratio', 0))
        deg_return = calculate_degradation(train_res.get('total_return', 0), test_res.get('total_return', 0))
        
        row = {
            'window': i,
            'test_start': window['test_start'],
            'global_score': study.best_value,
            
            # Metrics
            **{f'best_{k}': v for k, v in best_params.items()},
            **{f'train_{k}': v for k, v in train_res.items() if k != 'error'},
            **{f'test_{k}': v for k, v in test_res.items() if k != 'error'},
            **{f'holdout_{k}': v for k, v in holdout_res.items() if k != 'error'},
            
            # Degradation Metrics
            'degradation_sharpe': deg_sharpe,
            'degradation_return': deg_return
        }
        results.append(row)

    results_df = pd.DataFrame(results)

    # D. Summary Print
    if verbose and not results_df.empty:
        avg_deg_sharpe = results_df['degradation_sharpe'].mean()
        print("\n--- Degradation Analysis ---")
        print(f"Avg Sharpe Degradation: {avg_deg_sharpe:.2%} (Lower is better)")
        print(f"  < 10%: Very Robust")
        print(f"  > 50%: High Overfitting")

    return results_df, study


def run_walkforward(data_dict: Dict[str, pd.DataFrame],
                          param_space: Dict[str, Tuple],
                          strategy_class: type,
                          strategy_name: str,
                          trade_config: TradeConfig,
                          wf_config: Dict = None,
                          objective_weights: Tuple[float, float] = (0.9, 0.1),
                          verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    if wf_config is None: wf_config = create_wf_config()
    
    # A. Execute Main Logic
    results_df, global_study = walk_forward_analysis(
        data_dict, param_space, strategy_class, strategy_name, 
        trade_config, wf_config, objective_weights, verbose
    )
    
    # B. Stability Analysis (Cluster Method)
    stability_df = analyze_cluster_stability(global_study, top_n=20)
    if verbose:
        print("\n--- Parameter Stability (Cluster Analysis) ---")
        if not stability_df.empty:
            print(stability_df[['parameter', 'cv', 'range_ratio', 'assessment']].to_string())

    # C. Importance Analysis
    sensitivity_df = calculate_param_importance(global_study)
    if verbose and not sensitivity_df.empty:
        print("\n--- Parameter Importance ---")
        print(sensitivity_df.head(5).to_string())

    # D. Visualization (Top 2 Important Params)
    if not sensitivity_df.empty:
        top_params = sensitivity_df['parameter'].head(2).tolist()
        if len(top_params) == 2:
            plot_parameter_heatmap(global_study, top_params[0], top_params[1])
            
    # E. Export
    export_results(results_df, global_study, sensitivity_df, output_dir="wf_method2_results")
    
    return results_df, sensitivity_df


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
        
        window_scores = []
        total_trades_all_windows = 0
        max_drawdowns = []
        
        # 2. Cross-Validation Loop
        for i, window in enumerate(windows):
            train_data = filter_data_by_date(data_dict, window['train_start'], window['train_end'])
            
            # Skip empty windows safely
            if not train_data: continue
            
            results = run_backtest(train_data, params, strategy_class, strategy_name, trade_config)
            
            # CRASH PENALTY: If code fails, return worst possible score
            if 'error' in results:
                window_scores.append(-10.0) 
                continue
            
            # --- OPTION 3: ROBUST METRIC CALCULATION ---
            
            # A. Get Base Metrics
            sharpe = results.get('sharpe_ratio', 0)
            win_rate = results.get('win_rate', 0)
            max_dd = results.get('max_drawdown', 100.0) / 100.0 # Convert 20% to 0.2
            
            # B. Penalize Volatility (Deflated Score)
            # Instead of raw Sharpe, we divide by (1 + MaxDD). 
            # If Drawdown is 0%, Score = Sharpe. 
            # If Drawdown is 50%, Score = Sharpe / 1.5 (Massive Penalty).
            risk_adjusted_score = (sharpe * objective_weights[1] + win_rate * objective_weights[0]) / (1 + (max_dd * 2))
            
            # C. Hard Drawdown Cap
            # If any single window has > 25% drawdown, severely punish this trial.
            if max_dd > 0.25:
                risk_adjusted_score -= 1.0 

            window_scores.append(risk_adjusted_score)
            max_drawdowns.append(max_dd)
            total_trades_all_windows += results.get('total_trades', 0)

            # Pruning (Early Stopping)
            # We use a conservative mean for pruning to stop bad trials faster
            current_conservative_score = np.mean(window_scores) - (0.5 * np.std(window_scores))
            trial.report(current_conservative_score, i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # 3. Final Aggregation (Robustness Check)
        if not window_scores: return -999.0
            
        # --- KEY CHANGE: CONSISTENCY OVER PEAK PERFORMANCE ---
        
        mean_score = np.mean(window_scores)
        std_score = np.std(window_scores)
        min_score = np.min(window_scores)
        
        # The "Robust Score":
        # We take the Average, but subtract half the Standard Deviation.
        # This explicitly penalizes strategies that are "Boom and Bust" (High Mean, High Std).
        # We also add 20% weight to the WORST window (Min Score) to ensure safety.
        robust_score = (0.7 * (mean_score - 0.5 * std_score)) + (0.3 * min_score)
        
        # Constraint: Minimum Activity
        if total_trades_all_windows < (min_trades * len(window_scores) * 0.5):
            return -999.0

        # Store attributes for analysis
        trial.set_user_attr('avg_score', mean_score)
        trial.set_user_attr('robust_score', robust_score)
        trial.set_user_attr('worst_window_score', min_score)
        trial.set_user_attr('max_drawdown', np.mean(max_drawdowns))
        
        return robust_score

    return objective
