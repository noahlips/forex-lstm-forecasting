import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier

def perform_eda(dataset, target_col='target_5d'):
    os.makedirs("eda_outputs", exist_ok=True)
    
    plt.figure(figsize=(6, 4))
    sns.countplot(x=target_col, data=dataset)
    plt.title(f'Target Distribution ({target_col})')
    plt.savefig(f"eda_outputs/target_distribution.png")
    plt.close()
    
    feature_cols = [c for c in dataset.columns if 'target' not in c and c != 'Close']
    corr_matrix = dataset[feature_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    plt.title('Feature Correlation Matrix')
    plt.savefig(f"eda_outputs/correlation_matrix.png")
    plt.close()

def run_baseline_model(X_train, y_train, X_test, y_test, horizon_index=1):
    n_train = X_train.shape[0]
    X_train_flat = X_train.reshape(n_train, -1)
    n_test = X_test.shape[0]
    X_test_flat = X_test.reshape(n_test, -1)
    
    y_train_h = y_train[:, horizon_index]
    y_test_h = y_test[:, horizon_index]
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_flat, y_train_h)
    y_pred = rf.predict(X_test_flat)
    return accuracy_score(y_test_h, y_pred)

def get_optimal_threshold(model, X_val, horizon_index=1):
    y_prob_val = model.predict(X_val, verbose=0)
    prob_h = y_prob_val[:, horizon_index]
    return np.median(prob_h)

def predict_and_evaluate(model, X_test, y_test, horizons, test_returns, confidence_threshold=0.65):
    y_prob_all = model.predict(X_test, verbose=0)
    results = {}
    
    print(f"Evaluation (Confidence Threshold: {confidence_threshold})")
    
    for i, h in enumerate(horizons):
        true_h = y_test[:, i]
        prob_h = y_prob_all[:, i]
        returns_h = test_returns[:, i]
        
        pred_h = np.full(len(prob_h), -1, dtype=int)
        pred_h[prob_h > confidence_threshold] = 1
        pred_h[prob_h < (1 - confidence_threshold)] = 0
        
        traded_mask = (pred_h != -1)
        n_trades = np.sum(traded_mask)
        
        acc = accuracy_score(true_h[traded_mask], pred_h[traded_mask]) if n_trades > 0 else 0.0
        
        results[h] = {
            'y_true': true_h, 'y_pred': pred_h, 'prob': prob_h, 'returns': returns_h, 'n_trades': n_trades
        }
        
        print(f"Horizon {h}d: Trades {n_trades}/{len(pred_h)} ({n_trades/len(pred_h):.1%}) | Accuracy {acc:.2%}")
        
    return results

def run_backtest(results, horizon=5):
    r = results[horizon]
    y_pred = r['y_pred']
    real_returns = np.nan_to_num(r['returns'], nan=0.0)
    
    capital_market = 1000.0
    capital_model = 1000.0
    transaction_cost = 0.0002
    
    hist_market = [capital_market]
    hist_model = [capital_model]
    
    n_trades = 0
    
    for i in range(len(y_pred)):
        ret = real_returns[i]
        capital_market *= (1 + ret)
        
        daily_gain = 0
        if y_pred[i] == 1:
            daily_gain = ret - transaction_cost
            n_trades += 1
        elif y_pred[i] == 0:
            daily_gain = -ret - transaction_cost
            n_trades += 1
            
        capital_model *= (1 + daily_gain)
        hist_market.append(capital_market)
        hist_model.append(capital_model)
    
    perf_m = (capital_market - 1000) / 1000
    perf_ai = (capital_model - 1000) / 1000
    
    print(f"Backtest H{horizon}: Market {perf_m:+.2%} | AI {perf_ai:+.2%} | Trades {n_trades}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(hist_market, label='Market', color='gray', alpha=0.6)
    plt.plot(hist_model, label='AI Model', color='blue')
    plt.title(f'Backtest Horizon {horizon}d')
    plt.legend()
    plt.savefig(f"backtest_h{horizon}.png")
    plt.close()
