import os
import pickle
import numpy as np
import tensorflow as tf
from datetime import datetime

from data import prepare_dataset, create_sequences, split_temporal, balance_training_set, normalize_features
from model import build_classifier, compile_model, train_model, get_callbacks
from analysis import perform_eda, run_baseline_model, get_optimal_threshold, predict_and_evaluate, run_backtest

def main():
    tf.keras.utils.set_random_seed(42)
    np.random.seed(42)
    
    print("Running Forex Project...")
    
    WINDOW_SIZE = 60
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    EPOCHS = 50
    BATCH_SIZE = 32
    HORIZONS = [1, 5]
    CONFIDENCE_THRESHOLD = 0.65
    
    dataset = prepare_dataset(start_date='2000-01-01', horizons=HORIZONS)
    
    X, y, _ = create_sequences(dataset, HORIZONS, WINDOW_SIZE)
    
    returns_cols = [f'real_return_{h}d' for h in HORIZONS]
    returns_aligned = dataset[returns_cols].values[WINDOW_SIZE:]
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_temporal(X, y, TEST_SIZE, VAL_SIZE)
    test_idx = int(len(X) * (1 - TEST_SIZE))
    returns_test = returns_aligned[test_idx:]
    
    X_train, y_train = balance_training_set(X_train, y_train)
    X_train_norm, X_val_norm, X_test_norm, scaler = normalize_features(X_train, X_val, X_test)
    
    baseline_acc = run_baseline_model(X_train_norm, y_train, X_test_norm, y_test, horizon_index=1)
    print(f"Baseline Accuracy: {baseline_acc:.2%}")
    
    model = build_classifier(
        input_shape=(X_train_norm.shape[1], X_train_norm.shape[2]),
        n_horizons=len(HORIZONS),
        units=24,
        dropout=0.2
    )
    model = compile_model(model, learning_rate=0.001)
    
    train_model(
        model, X_train_norm, y_train, X_val_norm, y_val,
        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=get_callbacks()
    )
    
    optimal_thresh = get_optimal_threshold(model, X_val_norm, horizon_index=1)
    
    results = predict_and_evaluate(
        model, X_test_norm, y_test, HORIZONS, returns_test,
        confidence_threshold=CONFIDENCE_THRESHOLD
    )
    
    run_backtest(results, horizon=5)
    
    from trade_timing_analysis import analyze_trade_timing
    analyze_trade_timing(results, horizon=5)

    # Then run walk-forward validation
    from walk_forward_validation import walk_forward_validation
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

    
    save_dir = f"saved_models/final"
    os.makedirs(save_dir, exist_ok=True)
    model.save(f"{save_dir}/model.keras")
    with open(f"{save_dir}/scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Done.")

if __name__ == "__main__":
    main()