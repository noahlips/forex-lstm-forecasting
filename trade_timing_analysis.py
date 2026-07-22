"""
Trade Timing Analysis - Verify model is not just lucky on a few trades
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def analyze_trade_timing(results, horizon=5):
    """
    Analyze when the model decides to trade.
    CRITICAL: If trades are clustered in one period, results may be misleading.
    """
    r = results[horizon]
    y_pred = r['y_pred']
    returns = np.nan_to_num(r['returns'], nan=0.0)
    
    # Find trade indices
    trade_indices = np.where(y_pred != -1)[0]
    n_trades = len(trade_indices)
    
    print("\n" + "="*60)
    print("TRADE TIMING ANALYSIS")
    print("="*60)
    
    # 1. Trade distribution over time
    total_days = len(y_pred)
    quarters = [total_days // 4, total_days // 2, 3 * total_days // 4, total_days]
    quarter_labels = ['Q1 (0-25%)', 'Q2 (25-50%)', 'Q3 (50-75%)', 'Q4 (75-100%)']
    
    trades_per_quarter = []
    for i, q in enumerate(quarters):
        start = quarters[i-1] if i > 0 else 0
        trades_in_q = np.sum((trade_indices >= start) & (trade_indices < q))
        trades_per_quarter.append(trades_in_q)
        print(f"{quarter_labels[i]}: {trades_in_q} trades ({trades_in_q/n_trades*100:.1f}%)")
    
    # Check if trades are evenly distributed
    expected_per_quarter = n_trades / 4
    max_deviation = max(abs(t - expected_per_quarter) for t in trades_per_quarter)
    
    if max_deviation > expected_per_quarter * 0.5:
        print("\n⚠️  WARNING: Trades are NOT evenly distributed!")
        print("   This may indicate the model is exploiting a specific market regime.")
    else:
        print("\n✓ Trades are reasonably distributed over time.")
    
    # 2. Long vs Short distribution
    long_mask = (y_pred == 1)
    short_mask = (y_pred == 0)
    n_long = np.sum(long_mask)
    n_short = np.sum(short_mask)
    
    print(f"\nTrade Direction:")
    print(f"  Long:  {n_long} ({n_long/n_trades*100:.1f}%)")
    print(f"  Short: {n_short} ({n_short/n_trades*100:.1f}%)")
    
    # 3. Performance breakdown by quarter
    print(f"\nPerformance by Quarter:")
    for i, q in enumerate(quarters):
        start = quarters[i-1] if i > 0 else 0
        
        # Trades in this quarter
        trades_in_q_mask = (trade_indices >= start) & (trade_indices < q)
        trades_in_q_idx = trade_indices[trades_in_q_mask]
        
        if len(trades_in_q_idx) > 0:
            # Calculate returns for these trades
            capital = 1000.0
            for idx in trades_in_q_idx:
                ret = returns[idx]
                if y_pred[idx] == 1:
                    capital *= (1 + ret - 0.0002)
                elif y_pred[idx] == 0:
                    capital *= (1 - ret - 0.0002)
            
            perf = (capital - 1000) / 1000
            print(f"  {quarter_labels[i]}: {perf:+.2%} ({len(trades_in_q_idx)} trades)")
        else:
            print(f"  {quarter_labels[i]}: No trades")
    
    # 4. Visualize trade timing
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Cumulative capital over time
    plt.subplot(2, 1, 1)
    capital = 1000.0
    capital_history = [capital]
    
    for i in range(len(y_pred)):
        ret = returns[i]
        if y_pred[i] == 1:
            capital *= (1 + ret - 0.0002)
        elif y_pred[i] == 0:
            capital *= (1 - ret - 0.0002)
        capital_history.append(capital)
    
    plt.plot(capital_history, label='AI Model', color='blue', linewidth=2)
    plt.axhline(y=1000, color='black', linestyle='--', alpha=0.3)
    
    # Mark trades
    for idx in trade_indices:
        color = 'green' if y_pred[idx] == 1 else 'red'
        plt.axvline(x=idx, color=color, alpha=0.2, linewidth=0.5)
    
    plt.title('Capital Evolution with Trade Markers (Green=Long, Red=Short)')
    plt.ylabel('Capital ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Trade density over time
    plt.subplot(2, 1, 2)
    
    # Create bins for trade density
    bins = np.linspace(0, total_days, 50)
    trade_density, bin_edges = np.histogram(trade_indices, bins=bins)
    
    plt.bar(bin_edges[:-1], trade_density, width=bin_edges[1]-bin_edges[0], 
            alpha=0.7, color='blue', edgecolor='black')
    plt.axhline(y=n_trades/50, color='red', linestyle='--', 
                label='Expected (uniform distribution)', alpha=0.7)
    plt.title('Trade Density Over Time')
    plt.xlabel('Time (Days)')
    plt.ylabel('Number of Trades')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('trade_timing_analysis.png', dpi=150)
    print(f"\n✓ Graph saved: trade_timing_analysis.png")
    plt.close()
    
    # 5. Statistical test: Are trades independent?
    # Calculate time between consecutive trades
    if n_trades > 1:
        time_diffs = np.diff(trade_indices)
        avg_time_between = np.mean(time_diffs)
        std_time_between = np.std(time_diffs)
        
        print(f"\nTrade Spacing:")
        print(f"  Average days between trades: {avg_time_between:.1f} ± {std_time_between:.1f}")
        print(f"  Min spacing: {np.min(time_diffs)} days")
        print(f"  Max spacing: {np.max(time_diffs)} days")
        
        # If trades are clustered (std is very high), it's suspicious
        if std_time_between > avg_time_between * 2:
            print("  ⚠️  High variance in spacing - trades may be clustered")
        else:
            print("  ✓ Spacing variance is reasonable")
    
    return trade_indices

