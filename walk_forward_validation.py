"""
Walk-Forward Validation for Best Configuration
Tests if the model is robust across different market conditions
"""
import numpy as np
import tensorflow as tf
from data import prepare_dataset, create_sequences, split_temporal, balance_training_set, normalize_features
from model import build_classifier, compile_model, train_model, get_callbacks
from analysis import predict_and_evaluate, run_backtest

def walk_forward_validation(best_config):
    """
    Test the best configuration on 4 consecutive time periods
    CRITICAL: If performance is good only in 1-2 periods, model is not robust
    """
    print("\n" + "="*60)
    print("WALK-FORWARD VALIDATION (Best Config)")
    print("="*60)
    print(f"Testing config: {best_config['name']}")
    
    # Prepare data
    tf.keras.utils.set_random_seed(42)
    np.random.seed(42)
    
    HORIZONS = [1, 5]
    dataset = prepare_dataset(start_date='2000-01-01', horizons=HORIZONS)
    X, y, _ = create_sequences(dataset, HORIZONS, window_size=60)
    
    returns_cols = [f'real_return_{h}d' for h in HORIZONS]
    returns_aligned = dataset[returns_cols].values[60:]
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_temporal(X, y, test_size=0.2, val_size=0.1)
    test_idx = int(len(X) * 0.8)
    returns_test = returns_aligned[test_idx:]
    
    X_train, y_train = balance_training_set(X_train, y_train)
    X_train_norm, X_val_norm, X_test_norm, scaler = normalize_features(X_train, X_val, X_test)
    
    # Train model with best config
    model = build_classifier(
        input_shape=(X_train_norm.shape[1], X_train_norm.shape[2]),
        n_horizons=len(HORIZONS),
        units=best_config['units'],
        dropout=best_config['dropout']
    )
    
    model = compile_model(model, learning_rate=best_config['learning_rate'])
    
    train_model(
        model, X_train_norm, y_train, X_val_norm, y_val,
        epochs=best_config['epochs'],
        batch_size=best_config['batch_size'],
        callbacks=get_callbacks(patience=best_config['patience'])
    )
    
    # Split test set into 4 periods
    n_periods = 4
    period_size = len(X_test_norm) // n_periods
    
    period_results = []
    
    for period_idx in range(n_periods):
        start_idx = period_idx * period_size
        end_idx = start_idx + period_size if period_idx < n_periods-1 else len(X_test_norm)
        
        X_test_period = X_test_norm[start_idx:end_idx]
        y_test_period = y_test[start_idx:end_idx]
        returns_test_period = returns_test[start_idx:end_idx]
        
        print(f"\n{'='*60}")
        print(f"PERIOD {period_idx + 1}/{n_periods} (Days {start_idx}-{end_idx})")
        print(f"{'='*60}")
        
        results = predict_and_evaluate(
            model, X_test_period, y_test_period, HORIZONS, returns_test_period,
            confidence_threshold=best_config['confidence_threshold']
        )
        
        run_backtest(results, horizon=5)
        
        # Calculate performance
        r = results[5]
        y_pred = r['y_pred']
        real_returns = np.nan_to_num(r['returns'], nan=0.0)
        
        capital_market = 1000.0
        capital_ai = 1000.0
        
        for i in range(len(y_pred)):
            ret = real_returns[i]
            capital_market *= (1 + ret)
            
            if y_pred[i] == 1:
                capital_ai *= (1 + ret - 0.0002)
            elif y_pred[i] == 0:
                capital_ai *= (1 - ret - 0.0002)
        
        perf_market = (capital_market - 1000) / 1000
        perf_ai = (capital_ai - 1000) / 1000
        
        period_results.append({
            'period': period_idx + 1,
            'market_return': perf_market,
            'ai_return': perf_ai,
            'outperformance': perf_ai - perf_market,
            'n_trades': r['n_trades'],
            'accuracy': results[5]['n_trades']
        })
    
    # Summary
    print("\n" + "="*60)
    print("WALK-FORWARD SUMMARY")
    print("="*60)
    print(f"{'Period':<10} {'Market':<12} {'AI Model':<12} {'Outperf.':<12} {'Trades'}")
    print("-"*60)
    
    for r in period_results:
        print(f"Period {r['period']:<3} {r['market_return']:>+10.2%} {r['ai_return']:>+10.2%} "
              f"{r['outperformance']:>+10.2%} {r['n_trades']:>8}")
    
    print("-"*60)
    avg_market = np.mean([r['market_return'] for r in period_results])
    avg_ai = np.mean([r['ai_return'] for r in period_results])
    avg_outperf = np.mean([r['outperformance'] for r in period_results])
    
    print(f"{'Average':<10} {avg_market:>+10.2%} {avg_ai:>+10.2%} {avg_outperf:>+10.2%}")
    
    std_ai = np.std([r['ai_return'] for r in period_results])
    print(f"{'Std Dev':<10} {'':>12} {std_ai:>10.2%}")
    
    # Consistency check
    positive_periods = sum(1 for r in period_results if r['ai_return'] > 0)
    outperform_periods = sum(1 for r in period_results if r['outperformance'] > 0)
    
    print(f"\nConsistency:")
    print(f"  Positive returns: {positive_periods}/{n_periods} periods")
    print(f"  Outperformed market: {outperform_periods}/{n_periods} periods")
    
    if positive_periods >= 3 and outperform_periods >= 3:
        print("  ✓ Model shows GOOD consistency across periods")
    elif positive_periods >= 2 and outperform_periods >= 2:
        print("  ⚠️  Model shows MODERATE consistency")
    else:
        print("  ✗ Model is NOT consistent - may be overfitting to specific period")
    
    return period_results

# Usage:
if __name__ == "__main__":
    best_config = {
        'name': 'Higher_LR',
        'units': 64,
        'dropout': 0.4,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 100,
        'patience': 20,
        'confidence_threshold': 0.65
    }
    
    walk_forward_validation(best_config)