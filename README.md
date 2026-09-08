# Predicting EUR/USD direction with an LSTM

Machine Learning project from my 4th year at ESILV. The idea was to see whether a recurrent network could predict the direction of the EUR/USD close, one day and five days ahead, from daily data going back to 2000.

Short answer: sort of, and mostly not in a way you would trade on. The interesting part was building the evaluation properly, because it is very easy to fool yourself on financial time series.

![Backtest](docs/backtest_h5.png)

## The data

Prices come from Yahoo Finance. On top of the usual technical indicators I added a few macro series, since EUR/USD is driven at least as much by rate differentials as by chart patterns:

- Fed funds rate minus ECB deposit rate, from FRED
- VIX, as a risk appetite proxy
- Dollar index (DXY)

The technical side is about 40 features: returns at several lags, rolling means and standard deviations, distance to SMA 10/20/50/200, EMA and moving average crossovers, RSI, Stochastic, MACD, Bollinger band position and width, ATR, ADX with directional indicators, momentum and rate of change, and where the price sits in its recent range.

Everything is turned into 60-day sequences.

## The model

Two stacked bidirectional LSTM layers (64 then 32 units), with batch normalisation, dropout at 0.4 and L2 on the kernels, then two dense layers, then a sigmoid output per horizon. Adam with gradient clipping, early stopping on validation loss, and learning rate reduction on plateau.

Class weights handle the slight imbalance between up and down days.

## Not fooling yourself

This is where most of the work went.

The split is strictly temporal: train, then validation, then test, in chronological order. No shuffling across the boundary. The scaler is fitted on the training set only, and the class balancing is applied to the training set only. Both of those are easy to get wrong and both inflate your results if you do.

Predictions are only turned into trades when the model is confident enough, with the threshold picked on the validation set rather than the test set.

Then there is `walk_forward_validation.py`, which retrains on a rolling window and tests on the segment right after, repeatedly. A single train/test split on financial data tells you how one market regime went. Walk-forward tells you whether the thing survives several.

![Trade timing](docs/trade_timing_analysis.png)

`trade_timing_analysis.py` looks at when the strategy wins and loses, which is more useful than a single accuracy number.

## Files

| | |
|---|---|
| `main.py` | Runs the whole thing end to end |
| `data.py` | Download, features, sequences, temporal split, scaling |
| `model.py` | Architecture, callbacks, training |
| `analysis.py` | EDA, logistic regression baseline, threshold tuning, backtest |
| `walk_forward_validation.py` | Rolling retrain and evaluation |
| `hyperparameter_tuning.py` | Grid over architectures and learning rates |
| `trade_timing_analysis.py` | When the strategy wins and loses |

## Running it

```bash
pip install -r requirements.txt
python main.py
```

Data downloads on its own, no API key needed.

## Caveat

This was a school project. Do not trade on it. Backtest performance on 25 years of a single currency pair is not evidence that anything works going forward, and I would treat any positive result here as a starting point for more careful work rather than a conclusion.
