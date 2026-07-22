import pandas as pd
import yfinance as yf
import numpy as np
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from sklearn.preprocessing import StandardScaler

def download_data(start_date='2000-01-01', ticker='EURUSD=X'):
    print(f"Downloading {ticker}...")
    df = yf.download(ticker, start=start_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=['Close'])
    print(f"  ✓ {len(df)} rows downloaded")
    return df

def add_macro_features(df):
    """Unchanged - macro features are good"""
    data = pd.DataFrame(index=df.index)
    try:
        import pandas_datareader as pdr
        fed_rate = pdr.get_data_fred('DFF', start='2000-01-01')
        ecb_rate = pdr.get_data_fred('ECBDFR', start='2000-01-01')
        rate_diff = (fed_rate - ecb_rate).squeeze()
        data['rate_differential'] = rate_diff.reindex(df.index, method='ffill')
        data['rate_diff_change'] = data['rate_differential'].pct_change(20)
        print("  ✓ Interest rates fetched")
    except:
        data['rate_differential'] = 0
        data['rate_diff_change'] = 0

    try:
        vix = yf.download('^VIX', start='2000-01-01', progress=False)['Close']
        data['vix'] = vix.reindex(df.index, method='ffill')
        data['vix_change'] = data['vix'].pct_change(5)
        print("  ✓ VIX fetched")
    except:
        data['vix'] = 0
        data['vix_change'] = 0

    try:
        dxy = yf.download('DX-Y.NYB', start='2000-01-01', progress=False)['Close']
        data['dxy'] = dxy.reindex(df.index, method='ffill')
        data['dxy_momentum'] = data['dxy'].pct_change(20)
        print("  ✓ DXY fetched")
    except:
        data['dxy'] = 0
        data['dxy_momentum'] = 0
    
    return data

def add_technical_features(df):
    """
    ENHANCED TECHNICAL FEATURES:
    - Added Stochastic Oscillator (momentum)
    - Added ATR (volatility)
    - Added EMA (faster than SMA)
    - Added rolling statistics (mean, std)
    - Added price rate of change (ROC)
    """
    data = pd.DataFrame(index=df.index)
    close = df['Close']
    
    # Basic returns
    data['return_1d'] = close.pct_change(1)
    data['return_3d'] = close.pct_change(3)
    data['return_5d'] = close.pct_change(5)
    data['return_10d'] = close.pct_change(10)
    
    # Lag features (CRITICAL for time series)
    for lag in [1, 2, 3, 5, 10]:
        data[f'return_lag_{lag}'] = data['return_1d'].shift(lag)
    
    # Rolling statistics (capture recent trend)
    data['return_mean_5'] = data['return_1d'].rolling(5).mean()
    data['return_mean_20'] = data['return_1d'].rolling(20).mean()
    data['return_std_5'] = data['return_1d'].rolling(5).std()
    data['return_std_20'] = data['return_1d'].rolling(20).std()
    
    # Volatility
    data['volatility_5d'] = data['return_1d'].rolling(5).std()
    data['volatility_20d'] = data['return_1d'].rolling(20).std()
    data['volatility_60d'] = data['return_1d'].rolling(60).std()
    
    # Moving Averages
    sma_10 = SMAIndicator(close, 10).sma_indicator()
    sma_20 = SMAIndicator(close, 20).sma_indicator()
    sma_50 = SMAIndicator(close, 50).sma_indicator()
    sma_200 = SMAIndicator(close, 200).sma_indicator()
    
    ema_12 = EMAIndicator(close, 12).ema_indicator()
    ema_26 = EMAIndicator(close, 26).ema_indicator()
    
    data['dist_sma10'] = (close - sma_10) / sma_10
    data['dist_sma20'] = (close - sma_20) / sma_20
    data['dist_sma50'] = (close - sma_50) / sma_50
    data['dist_sma200'] = (close - sma_200) / sma_200
    
    data['ma_cross_10_20'] = (sma_10 - sma_20) / sma_20
    data['ma_cross_20_50'] = (sma_20 - sma_50) / sma_50
    data['ma_cross_50_200'] = (sma_50 - sma_200) / sma_200
    
    data['ema_cross'] = (ema_12 - ema_26) / ema_26
    
    # RSI (Contrarian)
    data['rsi_14'] = RSIIndicator(close, 14).rsi()
    data['rsi_overbought'] = (data['rsi_14'] > 70).astype(int)
    data['rsi_oversold'] = (data['rsi_14'] < 30).astype(int)
    
    # Stochastic Oscillator (NEW)
    try:
        stoch = StochasticOscillator(df['High'], df['Low'], close, 14, 3)
        data['stoch_k'] = stoch.stoch()
        data['stoch_d'] = stoch.stoch_signal()
    except:
        data['stoch_k'] = 50
        data['stoch_d'] = 50
    
    # MACD
    macd = MACD(close)
    data['macd'] = macd.macd()
    data['macd_signal'] = macd.macd_signal()
    data['macd_diff'] = macd.macd_diff()
    data['macd_norm'] = data['macd_diff'] / close
    
    # Bollinger Bands
    bb = BollingerBands(close, 20)
    data['bb_width'] = bb.bollinger_wband()
    data['bb_pos'] = (close - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
    data['bb_upper_touch'] = (close > bb.bollinger_hband()).astype(int)
    data['bb_lower_touch'] = (close < bb.bollinger_lband()).astype(int)
    
    # ATR (Volatility - NEW)
    try:
        atr = AverageTrueRange(df['High'], df['Low'], close, 14)
        data['atr'] = atr.average_true_range()
        data['atr_pct'] = data['atr'] / close
    except:
        data['atr'] = 0
        data['atr_pct'] = 0
    
    # ADX (Trend Strength)
    try:
        adx = ADXIndicator(df['High'], df['Low'], close, 14)
        data['adx'] = adx.adx()
        data['di_pos'] = adx.adx_pos()
        data['di_neg'] = adx.adx_neg()
        data['di_diff'] = data['di_pos'] - data['di_neg']
    except:
        data['adx'] = 25
        data['di_pos'] = 0
        data['di_neg'] = 0
        data['di_diff'] = 0
    
    # Momentum
    data['mom_5'] = close.pct_change(5)
    data['mom_10'] = close.pct_change(10)
    data['mom_20'] = close.pct_change(20)
    
    # Rate of Change (ROC - NEW)
    data['roc_5'] = ((close - close.shift(5)) / close.shift(5)) * 100
    data['roc_10'] = ((close - close.shift(10)) / close.shift(10)) * 100
    
    # Price position in recent range (NEW)
    data['price_position_20'] = (close - close.rolling(20).min()) / (close.rolling(20).max() - close.rolling(20).min())
    
    print(f"  ✓ {len([c for c in data.columns])} technical features created")
    return data

def prepare_dataset(start_date='2000-01-01', horizons=[1, 5]):
    raw_data = download_data(start_date)
    if len(raw_data) < 200:
        raise ValueError("Insufficient data.")
        
    macro_features = add_macro_features(raw_data)
    technical_features = add_technical_features(raw_data)
    features = pd.concat([technical_features, macro_features], axis=1)
    
    targets = pd.DataFrame(index=raw_data.index)
    for h in horizons:
        future_close = raw_data['Close'].shift(-h)
        targets[f'target_{h}d'] = (future_close > raw_data['Close']).astype(int)
        current_close = raw_data['Close']
        targets[f'real_return_{h}d'] = (future_close - current_close) / current_close
    
    dataset = pd.concat([features, targets], axis=1)
    initial_len = len(dataset)
    dataset = dataset.replace([np.inf, -np.inf], np.nan).dropna()
    final_len = len(dataset)
    
    print(f"  ✓ Data cleaning: {initial_len} → {final_len} rows")
    
    if len(dataset) == 0:
        raise ValueError("Dataset empty after cleaning.")
        
    return dataset

def create_sequences(data, horizons, window_size=60):
    """Unchanged - working fine"""
    feature_cols = [c for c in data.columns if not c.startswith('target_') and not c.startswith('real_return')]
    target_cols = [f'target_{h}d' for h in horizons]
    
    features = data[feature_cols].values
    targets = data[target_cols].values
    
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(features[i-window_size:i])
        y.append(targets[i])
    
    print(f"  ✓ Sequences created: {len(X)} samples x {window_size} timesteps x {len(feature_cols)} features")
    return np.array(X), np.array(y), feature_cols

def split_temporal(X, y, test_size=0.2, val_size=0.1):
    """Unchanged - correct temporal split"""
    n = len(X)
    test_idx = int(n * (1 - test_size))
    val_idx = int(test_idx * (1 - val_size))
    
    X_train, y_train = X[:val_idx], y[:val_idx]
    X_val, y_val = X[val_idx:test_idx], y[val_idx:test_idx]
    X_test, y_test = X[test_idx:], y[test_idx:]
    
    print(f"Split: Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def balance_training_set(X_train, y_train):
    """Unchanged - undersampling is correct"""
    if y_train.ndim > 1:
        y_target = y_train[:, 0] 
    else:
        y_target = y_train
    
    indices_0 = np.where(y_target == 0)[0]
    indices_1 = np.where(y_target == 1)[0]
    
    min_samples = min(len(indices_0), len(indices_1))
    indices_0_sel = np.random.choice(indices_0, min_samples, replace=False)
    indices_1_sel = np.random.choice(indices_1, min_samples, replace=False)
    
    balanced_idx = np.concatenate([indices_0_sel, indices_1_sel])
    np.random.shuffle(balanced_idx)
    
    print(f"  ✓ Balanced: {len(indices_0)} downs + {len(indices_1)} ups → {len(balanced_idx)} samples")
    return X_train[balanced_idx], y_train[balanced_idx]

def normalize_features(X_train, X_val, X_test):
    """Unchanged - StandardScaler is correct"""
    n_samples, n_timesteps, n_features = X_train.shape
    X_train_flat = X_train.reshape(-1, n_features)
    
    scaler = StandardScaler()
    scaler.fit(X_train_flat)
    
    X_train_norm = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_val_norm = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test_norm = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
    
    print(f"  ✓ Features normalized (mean=0, std=1)")
    return X_train_norm, X_val_norm, X_test_norm, scaler