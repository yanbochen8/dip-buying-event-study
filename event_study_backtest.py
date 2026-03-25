"""
EVENT STUDY BACKTEST SCRIPT
Modular framework for testing event-driven strategies across multiple markets

ASSUMPTIONS / NOTES:
====================
1. TICKERS & PROXIES:
   - S&P 500: Using SPY (ETF proxy)
   - CSI 300: Using EWH (iShares MSCI Hong Kong ETF) as proxy
     → Structure allows easy CSV swap via load_custom_data()
   
2. EVENT THRESHOLDS:
   - Events defined as days in worst tail of close-to-close returns
   - Tails computed per market over full sample
   - Thresholds: 1%, 5%, 10% (worst returns in distribution)
   
3. MAX DRAWDOWN CALCULATION:
   - Forward max drawdown = worst daily low within holding period
   - Computed as: min(daily_low) / entry_price - 1
   - This measures realized max loss from entry to exit
   - Does NOT include lookback; purely forward-looking from entry
   
4. EVENT OVERLAP HANDLING:
   - Option to allow overlapping events or cluster non-overlapping
   - Default: allow_overlap=True (all events included)
   - Alternative: consecutive events use skip-forward logic
   
5. RANDOM BENCHMARK:
   - Randomly samples same count of entry dates from valid universe
   - Must have sufficient forward data for full holding period
   - Repeated 500-1000 times for robust statistics
   
6. LIMITATIONS (MVP):
   - Does not adjust for transaction costs or slippage
   - No position sizing or portfolio-level optimization
   - Assumes perfect market impact (can enter at next day open)
   - CSI 300 proxy may have different characteristics than index
   - Survivorship bias not addressed
   
7. LOOKBACK BIAS PREVENTION:
   - Event thresholds computed on full sample (known bias trade-off)
   - Forward returns only use future data
   - Entry at next day open (no same-day look-ahead)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'tickers': {
        'sp500': 'SPY',      # S&P 500 proxy
        'csi300': 'EWH',     # CSI 300 proxy (can be swapped for CSV)
    },
    'start_date': '2004-01-01',      # ~20 years of data
    'end_date': None,                 # Use today's date
    'event_percentiles': [1, 5, 10],  # Tail thresholds (worst %)
    'holding_periods': [1, 5, 20, 60, 252],  # In trading days
    'ma_period': 200,                 # Moving average for trend regime
    'random_samples': 1000,           # Bootstrap iterations for benchmark
    'allow_overlapping_events': True, # If False, skip events during prior holding window
}


# ============================================================================
# DATA DOWNLOAD & LOADING
# ============================================================================

def download_data(ticker: str, start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Download daily OHLC data from yfinance.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), or None for today
    
    Returns:
        DataFrame with columns: Open, High, Low, Close, Adj Close, Volume
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Downloading {ticker} from {start_date} to {end_date}...")
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        print(f"  ✓ Downloaded {len(data)} trading days")
        return data
    except Exception as e:
        print(f"  ✗ Error downloading {ticker}: {e}")
        raise


def load_custom_data(csv_path: str) -> pd.DataFrame:
    """
    Load OHLC data from CSV file.
    
    Expected columns: Date, Open, High, Low, Close, Adj Close (optional), Volume
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        DataFrame with same structure as yfinance data
    """
    print(f"Loading custom data from {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
    df = df.sort_index()
    print(f"  ✓ Loaded {len(df)} trading days")
    return df


# ============================================================================
# FEATURE PREPARATION
# ============================================================================

def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived features to OHLC data.
    
    Adds:
        - daily_return: Close-to-close log returns for event detection
        - ma200: 200-day moving average of close
        - trend_regime: 'above' if close > MA200, else 'below'
    
    Args:
        data: OHLC DataFrame
    
    Returns:
        DataFrame with added feature columns
    """
    df = data.copy()
    
    # Use Adj Close if available, else Close
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    
    # Daily returns (for event detection)
    df['daily_return'] = np.log(df[price_col] / df[price_col].shift(1))
    
    # 200-day moving average
    df['ma200'] = df[price_col].rolling(window=200).mean()
    
    # Trend regime
    df['trend_regime'] = df['Close'].apply(
        lambda x: 'above' if pd.notna(x) else np.nan
    )
    # Fill in the comparison on rows where we have both close and MA200
    mask = df['ma200'].notna()
    df.loc[mask, 'trend_regime'] = (df.loc[mask, 'Close'] > df.loc[mask, 'ma200']).map(
        {True: 'above', False: 'below'}
    )
    
    return df


# ============================================================================
# EVENT IDENTIFICATION
# ============================================================================

def identify_event_days(
    data: pd.DataFrame,
    event_percentiles: List[int] = [1, 5, 10],
    allow_overlapping: bool = True
) -> Dict[int, pd.DataFrame]:
    """
    Identify event days (extreme negative returns) for each threshold.
    
    Event thresholds are defined as worst X% of daily returns over full sample.
    
    Args:
        data: Prepared feature DataFrame
        event_percentiles: List of tail percentiles (1, 5, 10 = worst 1%, 5%, 10%)
        allow_overlapping: If False, remove events within prior holding window
    
    Returns:
        Dict mapping percentile to DataFrame of event dates with features
    """
    events = {}
    daily_ret = data['daily_return'].dropna()
    
    for pct in event_percentiles:
        threshold = np.percentile(daily_ret, pct)
        event_mask = data['daily_return'] <= threshold
        event_dates = data[event_mask].copy()
        
        # Filter out dates without MA200 or trend regime (need 200+ days prior)
        event_dates = event_dates[event_dates['ma200'].notna()].copy()
        event_dates = event_dates[event_dates['trend_regime'].notna()].copy()
        
        print(f"  Percentile {pct}%: threshold={threshold:.4f}, events={len(event_dates)}")
        events[pct] = event_dates
    
    return events


# ============================================================================
# FORWARD RETURN & DRAWDOWN CALCULATIONS
# ============================================================================

def compute_forward_return(
    data: pd.DataFrame,
    entry_idx: int,
    holding_days: int
) -> Optional[float]:
    """
    Compute forward return from entry day to exit day.
    
    Entry: Next day open after event
    Exit: Close N trading days later
    
    Args:
        data: OHLC DataFrame with index
        entry_idx: Index location of entry day (day after event)
        holding_days: Number of trading days to hold
    
    Returns:
        Return as decimal (0.05 = 5%), or None if insufficient data
    """
    if entry_idx + holding_days >= len(data):
        return None
    
    entry_price = data.iloc[entry_idx]['Open']
    exit_price = data.iloc[entry_idx + holding_days]['Close']
    
    if pd.isna(entry_price) or pd.isna(exit_price) or entry_price == 0:
        return None
    
    return (exit_price - entry_price) / entry_price


def compute_forward_max_drawdown(
    data: pd.DataFrame,
    entry_idx: int,
    holding_days: int
) -> Optional[float]:
    """
    Compute max drawdown from entry to exit.
    
    Method: Lowest intraday low during holding period / entry price - 1
    This represents the worst mark-to-market loss during the holding period.
    
    Args:
        data: OHLC DataFrame with index
        entry_idx: Index location of entry day
        holding_days: Number of trading days to hold
    
    Returns:
        Max drawdown as decimal (-0.10 = -10%), or None if insufficient data
    """
    if entry_idx + holding_days >= len(data):
        return None
    
    entry_price = data.iloc[entry_idx]['Open']
    min_low = data.iloc[entry_idx:entry_idx + holding_days + 1]['Low'].min()
    
    if pd.isna(entry_price) or pd.isna(min_low) or entry_price == 0:
        return None
    
    max_dd = (min_low - entry_price) / entry_price
    return max_dd


# ============================================================================
# MAIN BACKTEST ENGINE
# ============================================================================

def run_event_study(
    data: pd.DataFrame,
    events: Dict[int, pd.DataFrame],
    market_name: str,
    config: Dict
) -> pd.DataFrame:
    """
    Run event study backtest across all thresholds and holding periods.
    
    Args:
        data: Prepared OHLC DataFrame (with features)
        events: Dict mapping percentile to event dates
        market_name: Name of market (e.g., 'SPY', 'CSI300')
        config: Configuration dict
    
    Returns:
        DataFrame with results (one row per event, holding period, regime combination)
    """
    results = []
    
    for pct, event_dates in events.items():
        for event_date in event_dates.index:
            # Find next trading day (entry day)
            event_idx = data.index.get_loc(event_date)
            if event_idx + 1 >= len(data):
                continue
            
            entry_idx = event_idx + 1
            entry_date = data.index[entry_idx]
            
            event_close = event_dates.loc[event_date, 'Close']
            entry_open = data.iloc[entry_idx]['Open']
            trend = event_dates.loc[event_date, 'trend_regime']
            
            for holding_days in config['holding_periods']:
                # Ensure sufficient future data
                if entry_idx + holding_days >= len(data):
                    continue
                
                exit_date = data.index[entry_idx + holding_days]
                exit_close = data.iloc[entry_idx + holding_days]['Close']
                
                fwd_ret = compute_forward_return(data, entry_idx, holding_days)
                max_dd = compute_forward_max_drawdown(data, entry_idx, holding_days)
                
                if fwd_ret is not None and max_dd is not None:
                    results.append({
                        'market': market_name,
                        'event_pct': pct,
                        'event_date': event_date,
                        'entry_date': entry_date,
                        'entry_price': entry_open,
                        'exit_date': exit_date,
                        'exit_price': exit_close,
                        'holding_days': holding_days,
                        'trend_regime': trend,
                        'forward_return': fwd_ret,
                        'max_drawdown': max_dd,
                    })
    
    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df['win'] = (results_df['forward_return'] > 0).astype(int)
    
    print(f"\n{market_name}: {len(results_df)} valid event trades recorded")
    return results_df


# ============================================================================
# RANDOM BENCHMARK
# ============================================================================

def run_random_benchmark(
    data: pd.DataFrame,
    event_results: pd.DataFrame,
    config: Dict
) -> pd.DataFrame:
    """
    Generate random benchmark by sampling entry dates uniformly.
    
    For each unique (event_pct, holding_days, trend_regime) combination
    from the real events, randomly sample the same number of entry dates
    and compute the same metrics.
    
    Args:
        data: Prepared OHLC DataFrame
        event_results: Results from run_event_study
        config: Configuration dict
    
    Returns:
        DataFrame with random benchmark results
    """
    np.random.seed(42)
    random_results = []
    
    # Valid entry indices: must have enough future data for longest holding period
    max_holding = max(config['holding_periods'])
    valid_entry_indices = np.arange(1, len(data) - max_holding)
    
    # Group real events by (pct, holding_days, regime)
    groupby_cols = ['event_pct', 'holding_days', 'trend_regime']
    grouped = event_results.groupby(groupby_cols).size().reset_index(name='count')
    
    for _, group_row in grouped.iterrows():
        event_pct = group_row['event_pct']
        holding_days = group_row['holding_days']
        regime = group_row['trend_regime']
        num_samples_needed = group_row['count']
        
        # Filter valid entries by regime
        regime_mask = data['trend_regime'] == regime
        valid_regime_indices = np.where(regime_mask.iloc[valid_entry_indices].values)[0]
        
        if len(valid_regime_indices) < num_samples_needed:
            print(f"  Warning: Only {len(valid_regime_indices)} valid entries for "
                  f"pct={event_pct}, holding={holding_days}, regime={regime} "
                  f"(need {num_samples_needed})")
            continue
        
        # Sample entry indices
        sampled_indices = np.random.choice(valid_regime_indices, num_samples_needed, replace=True)
        
        for entry_idx in sampled_indices:
            entry_idx = valid_entry_indices[entry_idx]
            entry_date = data.index[entry_idx]
            entry_open = data.iloc[entry_idx]['Open']
            exit_idx = entry_idx + holding_days
            exit_date = data.index[exit_idx]
            exit_close = data.iloc[exit_idx]['Close']
            
            fwd_ret = (exit_close - entry_open) / entry_open
            max_dd = compute_forward_max_drawdown(data, entry_idx, holding_days)
            
            random_results.append({
                'event_pct': event_pct,
                'entry_date': entry_date,
                'entry_price': entry_open,
                'exit_date': exit_date,
                'exit_price': exit_close,
                'holding_days': holding_days,
                'trend_regime': regime,
                'forward_return': fwd_ret,
                'max_drawdown': max_dd,
            })
    
    random_df = pd.DataFrame(random_results)
    if len(random_df) > 0:
        random_df['win'] = (random_df['forward_return'] > 0).astype(int)
    
    return random_df


# ============================================================================
# RESULTS AGGREGATION & SUMMARY
# ============================================================================

def summarize_results(
    event_results: pd.DataFrame,
    random_results: pd.DataFrame,
    market_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate event study results and compare to random benchmark.
    
    Args:
        event_results: DataFrame from run_event_study
        random_results: DataFrame from run_random_benchmark
        market_name: Name of market
    
    Returns:
        Tuple of (event_summary_df, benchmark_summary_df)
    """
    # Summarize events
    groupby_cols = ['event_pct', 'holding_days', 'trend_regime']
    
    event_summary = event_results.groupby(groupby_cols).agg({
        'forward_return': ['count', 'mean', 'median', 'std'],
        'win': 'mean',
        'max_drawdown': 'mean',
    }).reset_index()
    
    event_summary.columns = ['event_pct', 'holding_days', 'trend_regime',
                             'num_events', 'avg_return', 'median_return', 'return_std',
                             'win_rate', 'avg_max_dd']
    
    event_summary['market'] = market_name
    event_summary['source'] = 'actual_events'
    
    # Summarize random benchmark
    random_summary = random_results.groupby(groupby_cols).agg({
        'forward_return': ['count', 'mean', 'median', 'std'],
        'win': 'mean',
        'max_drawdown': 'mean',
    }).reset_index()
    
    random_summary.columns = ['event_pct', 'holding_days', 'trend_regime',
                              'num_events', 'avg_return', 'median_return', 'return_std',
                              'win_rate', 'avg_max_dd']
    
    random_summary['market'] = market_name
    random_summary['source'] = 'random_benchmark'
    
    return event_summary, random_summary


def print_summary_table(summary_df: pd.DataFrame, title: str):
    """Print formatted summary table to console."""
    print(f"\n{'='*100}")
    print(f"{title}")
    print(f"{'='*100}\n")
    
    # Format for readability
    display_df = summary_df.copy()
    display_df['avg_return'] = (display_df['avg_return'] * 100).round(2).astype(str) + '%'
    display_df['median_return'] = (display_df['median_return'] * 100).round(2).astype(str) + '%'
    display_df['win_rate'] = (display_df['win_rate'] * 100).round(1).astype(str) + '%'
    display_df['avg_max_dd'] = (display_df['avg_max_dd'] * 100).round(2).astype(str) + '%'
    display_df['num_events'] = display_df['num_events'].astype(int)
    
    print(display_df.to_string(index=False))
    print()


# ============================================================================
# PLOTTING & VISUALIZATION
# ============================================================================

def plot_results(
    event_summary: pd.DataFrame,
    random_summary: pd.DataFrame,
    market_name: str,
    output_dir: str = './backtest_output'
):
    """
    Create summary plots comparing event strategy to random benchmark.
    
    Args:
        event_summary: Summary DataFrame for actual events
        random_summary: Summary DataFrame for random benchmark
        market_name: Market name for title
        output_dir: Directory to save plots
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    # Plot 1: Average return by holding period (pooled across thresholds/regimes)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{market_name} - Event Study Results', fontsize=14, fontweight='bold')
    
    # Plot 1a: Avg return by holding period
    ax = axes[0, 0]
    hold_perf = event_summary.groupby('holding_days')['avg_return'].mean()
    ax.bar(hold_perf.index, hold_perf.values * 100, color='steelblue', alpha=0.7)
    ax.set_xlabel('Holding Period (days)')
    ax.set_ylabel('Avg Return (%)')
    ax.set_title('Average Return by Holding Period (Event Strategy)')
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 1b: Win rate by holding period
    ax = axes[0, 1]
    win_perf = event_summary.groupby('holding_days')['win_rate'].mean()
    ax.bar(win_perf.index, win_perf.values * 100, color='forestgreen', alpha=0.7)
    ax.set_xlabel('Holding Period (days)')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Win Rate by Holding Period (Event Strategy)')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 100])
    
    # Plot 1c: Event vs Random benchmark - Average return
    ax = axes[1, 0]
    merged = event_summary.merge(
        random_summary[['event_pct', 'holding_days', 'trend_regime', 'avg_return']],
        on=['event_pct', 'holding_days', 'trend_regime'],
        suffixes=('_event', '_random')
    )
    x = np.arange(len(merged))
    width = 0.35
    ax.bar(x - width/2, merged['avg_return_event'] * 100, width, label='Event', alpha=0.8)
    ax.bar(x + width/2, merged['avg_return_random'] * 100, width, label='Random', alpha=0.8)
    ax.set_ylabel('Avg Return (%)')
    ax.set_title('Event vs Random Benchmark - Avg Return')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r[0]}%\n{r[1]}d" for r in merged[['event_pct', 'holding_days']].values], 
                       fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Plot 1d: Event vs Random benchmark - Win rate
    ax = axes[1, 1]
    ax.bar(x - width/2, merged['win_rate_event'] * 100, width, label='Event', alpha=0.8)
    # random win rate not in standard merge, recompute
    random_win = random_summary.groupby(['event_pct', 'holding_days', 'trend_regime'])['win_rate'].mean().reset_index()
    merged_win = event_summary[['event_pct', 'holding_days', 'trend_regime', 'win_rate']].merge(
        random_win, on=['event_pct', 'holding_days', 'trend_regime'], suffixes=('_event', '_random')
    )
    ax.bar(np.arange(len(merged_win)) - width/2, merged_win['win_rate_event'] * 100, width, label='Event', alpha=0.8)
    ax.bar(np.arange(len(merged_win)) + width/2, merged_win['win_rate_random'] * 100, width, label='Random', alpha=0.8)
    ax.set_ylabel('Win Rate (%)')
    ax.set_title('Event vs Random Benchmark - Win Rate')
    ax.set_xticks(np.arange(len(merged_win)))
    ax.set_xticklabels([f"{r[0]}%\n{r[1]}d" for r in merged_win[['event_pct', 'holding_days']].values], 
                       fontsize=8)
    ax.set_ylim([0, 100])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    filepath = f"{output_dir}/{market_name.lower()}_summary.png"
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    print(f"  ✓ Saved plot: {filepath}")
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Execute full event study backtest."""
    print("\n" + "="*100)
    print("EVENT STUDY BACKTEST - TESTING MEAN REVERSION AFTER EXTREME DROPS")
    print("="*100)
    
    output_dir = Path('./backtest_output')
    output_dir.mkdir(exist_ok=True)
    
    all_results = []
    all_summaries = []
    
    # =========================================================================
    # MARKET 1: S&P 500 (SPY)
    # =========================================================================
    print("\n[1/2] Processing S&P 500 (SPY)...")
    print("-" * 100)
    
    spy_data = download_data(CONFIG['tickers']['sp500'], CONFIG['start_date'], CONFIG['end_date'])
    spy_data = prepare_features(spy_data)
    
    print("\nIdentifying event days...")
    spy_events = identify_event_days(spy_data, CONFIG['event_percentiles'])
    
    print("\nRunning event study...")
    spy_event_results = run_event_study(spy_data, spy_events, 'SPY', CONFIG)
    
    print("Running random benchmark (this may take a moment)...")
    spy_random_results = run_random_benchmark(spy_data, spy_event_results, CONFIG)
    
    spy_event_summary, spy_random_summary = summarize_results(
        spy_event_results, spy_random_results, 'SPY'
    )
    
    print_summary_table(spy_event_summary, "SPY - ACTUAL EVENT STRATEGY")
    print_summary_table(spy_random_summary, "SPY - RANDOM BENCHMARK")
    
    plot_results(spy_event_summary, spy_random_summary, 'SPY', str(output_dir))
    
    all_results.append(spy_event_results)
    all_results.append(spy_random_results)
    all_summaries.append(spy_event_summary)
    all_summaries.append(spy_random_summary)
    
    # =========================================================================
    # MARKET 2: CSI 300 (EWH Proxy)
    # =========================================================================
    print("\n[2/2] Processing CSI 300 (EWH proxy)...")
    print("-" * 100)
    
    ewh_data = download_data(CONFIG['tickers']['csi300'], CONFIG['start_date'], CONFIG['end_date'])
    ewh_data = prepare_features(ewh_data)
    
    print("\nIdentifying event days...")
    ewh_events = identify_event_days(ewh_data, CONFIG['event_percentiles'])
    
    print("\nRunning event study...")
    ewh_event_results = run_event_study(ewh_data, ewh_events, 'CSI300 (EWH)', CONFIG)
    
    print("Running random benchmark (this may take a moment)...")
    ewh_random_results = run_random_benchmark(ewh_data, ewh_event_results, CONFIG)
    
    ewh_event_summary, ewh_random_summary = summarize_results(
        ewh_event_results, ewh_random_results, 'CSI300 (EWH)'
    )
    
    print_summary_table(ewh_event_summary, "CSI300 (EWH) - ACTUAL EVENT STRATEGY")
    print_summary_table(ewh_random_summary, "CSI300 (EWH) - RANDOM BENCHMARK")
    
    plot_results(ewh_event_summary, ewh_random_summary, 'CSI300', str(output_dir))
    
    all_results.append(ewh_event_results)
    all_results.append(ewh_random_results)
    all_summaries.append(ewh_event_summary)
    all_summaries.append(ewh_random_summary)
    
    # =========================================================================
    # EXPORT RESULTS
    # =========================================================================
    print("\n" + "="*100)
    print("EXPORTING RESULTS")
    print("="*100 + "\n")
    
    # Combine all trade-level results
    all_trades = pd.concat(all_results, ignore_index=True)
    trades_file = output_dir / 'all_trades.csv'
    all_trades.to_csv(trades_file, index=False)
    print(f"  ✓ Saved trade details: {trades_file}")
    
    # Combine all summaries
    all_summary_table = pd.concat(all_summaries, ignore_index=True)
    summary_file = output_dir / 'summary_results.csv'
    all_summary_table.to_csv(summary_file, index=False)
    print(f"  ✓ Saved summary results: {summary_file}")
    
    # Create comparison table (Event vs Random for each market)
    comparison_data = []
    for market in ['SPY', 'CSI300 (EWH)']:
        event_sub = all_summary_table[(all_summary_table['market'] == market) & 
                                      (all_summary_table['source'] == 'actual_events')]
        random_sub = all_summary_table[(all_summary_table['market'] == market) & 
                                       (all_summary_table['source'] == 'random_benchmark')]
        
        for _, evt_row in event_sub.iterrows():
            # Find matching random row
            random_match = random_sub[
                (random_sub['event_pct'] == evt_row['event_pct']) &
                (random_sub['holding_days'] == evt_row['holding_days']) &
                (random_sub['trend_regime'] == evt_row['trend_regime'])
            ]
            
            if len(random_match) > 0:
                rnd_row = random_match.iloc[0]
                comparison_data.append({
                    'market': market,
                    'event_pct': evt_row['event_pct'],
                    'holding_days': evt_row['holding_days'],
                    'trend_regime': evt_row['trend_regime'],
                    'event_avg_return_%': round(evt_row['avg_return'] * 100, 2),
                    'random_avg_return_%': round(rnd_row['avg_return'] * 100, 2),
                    'excess_return_%': round((evt_row['avg_return'] - rnd_row['avg_return']) * 100, 2),
                    'event_win_rate_%': round(evt_row['win_rate'] * 100, 1),
                    'random_win_rate_%': round(rnd_row['win_rate'] * 100, 1),
                    'event_num_trades': int(evt_row['num_events']),
                })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_file = output_dir / 'event_vs_random_comparison.csv'
    comparison_df.to_csv(comparison_file, index=False)
    print(f"  ✓ Saved comparison table: {comparison_file}")
    
    # Print final comparison
    print("\n" + "="*100)
    print("EXCESS RETURNS: EVENT STRATEGY vs RANDOM BENCHMARK")
    print("="*100 + "\n")
    print(comparison_df.to_string(index=False))
    
    print("\n" + "="*100)
    print("BACKTEST COMPLETE")
    print("="*100)
    print(f"\nOutput files saved to: {output_dir}/")
    print("  - all_trades.csv: Trade-level details")
    print("  - summary_results.csv: Aggregated statistics")
    print("  - event_vs_random_comparison.csv: Event vs benchmark comparison")
    print("  - spy_summary.png: SPY visualization")
    print("  - csi300_summary.png: CSI 300 visualization")
    
    return all_summary_table, comparison_df


if __name__ == '__main__':
    main()
