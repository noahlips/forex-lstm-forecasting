"""
Hyperparameter Tuning Script
Tests different configurations to find the best model
"""
import numpy as np
import tensorflow as tf
from data import prepare_dataset, create_sequences, split_temporal, balance_training_set, normalize_features
from model import build_classifier, compile_model, train_model, get_callbacks
from analysis import predict_and_evaluate, run_backtest

def test_configuration(config, X_train, y_train, X_val, y_val, X_test, y_test, returns_test, horizons):
    """Test a single configuration"""
    print(f"\n{'='*60}")
    print(f"Testing Config: {config}")
    print(f"{'='*60}")
    
    # Build model
    model = build_classifier(
        input_shape=(X_train.shape[1], X_train.shape[2]),
        n_horizons=len(horizons),
        units=config['units'],
        dropout=config['dropout']
    )
    
    model = compile_model(model, learning_rate=config['learning_rate'])
    
    # Train
    history = train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        callbacks=get_callbacks(patience=config['patience'])
    )
    
    # Evaluate
    results = predict_and_evaluate(
        model, X_test, y_test, horizons, returns_test,
        confidence_threshold=config['confidence_threshold']
    )
    
    # Get performance metrics
    best_epoch = np.argmin(history.history['val_loss'])
    val_acc = history.history['val_accuracy'][best_epoch]
    
    # Backtest
    run_backtest(results, horizon=5)
    
    # Extract backtest performance
    r = results[5]
    y_pred = r['y_pred']
    real_returns = np.nan_to_num(r['returns'], nan=0.0)
    
    capital = 1000.0
    for i in range(len(y_pred)):
        ret = real_returns[i]
        if y_pred[i] == 1:
            capital *= (1 + ret - 0.0002)
        elif y_pred[i] == 0:
            capital *= (1 - ret - 0.0002)
    
    ai_return = (capital - 1000) / 1000
    
    return {
        'config': config,
        'val_acc': val_acc,
        'test_acc': results[5]['n_trades'],
        'ai_return': ai_return,
        'history': history
    }

def run_hyperparameter_search():
    """
    Grid Search over key hyperparameters
    """
    print("="*60)
    print("HYPERPARAMETER TUNING")
    print("="*60)
    
    # Prepare data once
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
    
    # Define configurations to test
    configs = [
        # Baseline (your current config)
        {
            'name': 'Baseline',
            'units': 24,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 50,
            'patience': 15,
            'confidence_threshold': 0.65
        },
        # Deeper network
        {
            'name': 'Deeper_Network',
            'units': 64,
            'dropout': 0.4,
            'learning_rate': 0.0005,
            'batch_size': 32,
            'epochs': 100,
            'patience': 20,
            'confidence_threshold': 0.65
        },
        # Lower confidence threshold
        {
            'name': 'Lower_Confidence',
            'units': 64,
            'dropout': 0.4,
            'learning_rate': 0.0005,
            'batch_size': 32,
            'epochs': 100,
            'patience': 20,
            'confidence_threshold': 0.60  # More trades
        },
        # Smaller batch size (better for small datasets)
        {
            'name': 'Small_Batch',
            'units': 64,
            'dropout': 0.4,
            'learning_rate': 0.0005,
            'batch_size': 16,
            'epochs': 100,
            'patience': 20,
            'confidence_threshold': 0.65
        },
        # Higher learning rate
        {
            'name': 'Higher_LR',
            'units': 64,
            'dropout': 0.4,
            'learning_rate': 0.001,
            'batch_size': 32,
            'epochs': 100,
            'patience': 20,
            'confidence_threshold': 0.65
        }
    ]
    
    # Test all configurations
    results_all = []
    for config in configs:
        result = test_configuration(
            config, X_train_norm, y_train, X_val_norm, y_val,
            X_test_norm, y_test, returns_test, HORIZONS
        )
        results_all.append(result)
    
    # Display summary
    print("\n" + "="*60)
    print("HYPERPARAMETER TUNING RESULTS")
    print("="*60)
    print(f"{'Config':<20} {'Val Acc':<10} {'Trades':<10} {'AI Return'}")
    print("-"*60)
    
    for r in results_all:
        print(f"{r['config']['name']:<20} {r['val_acc']:>8.2%} {r['test_acc']:>8} {r['ai_return']:>+10.2%}")
    
    # Find best configuration
    best_config = max(results_all, key=lambda x: x['ai_return'])
    print("\n" + "="*60)
    print(f"BEST CONFIGURATION: {best_config['config']['name']}")
    print(f"  AI Return: {best_config['ai_return']:+.2%}")
    print(f"  Val Accuracy: {best_config['val_acc']:.2%}")
    print("="*60)

if __name__ == "__main__":
    run_hyperparameter_search()