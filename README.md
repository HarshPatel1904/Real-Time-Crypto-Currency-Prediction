# CryptoScope — Multi-Cryptocurrency AI Price Range Predictor

CryptoScope is an end-to-end machine learning and deep learning web application that fetches completed candlestick and order-flow data from Binance, engineers 50 scale-invariant market features, and forecasts the **next-candle expected price range (`[Low – High]`)**, **midpoint close**, **expected % change**, and **directional bias (`UP` / `DOWN`)** across **8 cryptocurrencies** and **3 timeframes (`1m`, `5m`, `15m`)**.

---

## Key Features

- **8 Supported Cryptocurrencies (24 Pre-Trained Models):** Dedicated models and scalers for `BTCUSDT`, `ETHUSDT`, `BNBUSDT`, `SOLUSDT`, `XRPUSDT`, `ADAUSDT`, `DOGEUSDT`, and `AVAXUSDT` across `1m`, `5m`, and `15m` intervals.
- **Expected Price Range Band (`[Low – High]`):** Instead of a brittle single-point estimate, outputs a volatility-adaptive prediction band (`RANGE_Z_SCORE = 1.60`) combining `vol_20`, `vol_10`, and `atr_14_norm`, rendered as a shaded forecast band on the Chart.js graph.
- **Timeframe Selector (`1m` | `5m` | `15m`):** Instant switching between 1-minute, 5-minute, and 15-minute models directly from the dashboard header, with a manual **Refresh ↻** snapshot control.
- **Prediction Verification Log:** Live audit table comparing the model's predicted `[Low – High]` range and direction against actual closing prices over the last 10 completed candles, reporting `IN RANGE ✓ / OUTSIDE ✗`, midpoint `$` / `%` error, and hit rates.
- **Hybrid Deep Learning + Tabular Ensemble:** Combines a dual-head `Conv1D + Bidirectional LSTM + Softmax Attention` neural network with recency-weighted `HistGradientBoostingRegressor`, `Ridge`, `HistGradientBoostingClassifier`, and validation-calibrated blending.
- **Resilient Binance Market Ingestion:** Automatic 3-endpoint fallback (`data-api.binance.vision`, `api.binance.com`, `api-gcp.binance.com`) with HTTP `Date` server-time synchronization to prevent local clock skew from dropping newly closed candles.
- **In-Memory Model Caching:** `_MODEL_CACHE` caches loaded Keras and scikit-learn artifacts per `(symbol, interval)` for fast API inference.

---

## System Architecture & Project Structure

```text
multi_crypto_ai_predictor/
├── app.py                  # Flask server & REST API endpoints
├── config.py               # Assets, intervals, hyperparameters, and 50-feature registry
├── data.py                 # Binance REST klines fetcher with fallback & server-time sync
├── features.py             # 50 stationary technical, microstructure & order-flow features
├── model.py                # Hybrid Conv1D-BiLSTM-Attention + HGBR/Ridge/HGBC pipeline
├── train.py                # CLI script to train a single cryptocurrency model
├── train_all.py            # CLI wrapper to train all supported cryptocurrencies
├── train_all_models.py     # Detailed multi-crypto CLI batch trainer & evaluation report
├── requirements.txt        # Python dependencies
├── models/                 # Saved 1m models (models/<SYMBOL>/) & 5m/15m (models/<tf>/<SYMBOL>/)
├── templates/
│   └── index.html          # CryptoScope web dashboard template
└── static/
    ├── app.js              # Dashboard state, Chart.js forecast band & verification table
    └── style.css           # Light-theme dashboard styling
```

---

## Feature Engineering (`50` Stationary Features)

All features in [features.py](features.py) and [config.py](config.py) are scale-invariant so models generalize cleanly across changing price levels:

1. **Multi-Horizon Log Returns (`8`):** `ret_1`, `ret_2`, `ret_3`, `ret_5`, `ret_10`, `ret_20`, `ret_48`, `ret1_z`
2. **Trend & VWAP Deviations (`10`):** `dist_sma_5/10/20/50`, `dist_ema_5/10/20/50`, `dist_vwap_5`, `dist_vwap_15`
3. **Momentum & Oscillators (`7`):** `macd_norm`, `macd_hist_norm`, Wilder `rsi_7`, `rsi_14`, `rsi_28`, Bollinger `%B` (`bb_pos`) & `bb_width`
4. **Volatility Regime (`5`):** `atr_14_norm`, `vol_10`, `vol_20`, `vol_60`, `vol_ratio`
5. **Candle Microstructure (`7`):** `hl_range`, `body_ratio`, `candle_ret`, `upper_wick`, `lower_wick`, `close_pos`, `stoch_14`
6. **Volume & Binance Order Flow (`9`):** `rel_vol_5`, `rel_vol_20`, `taker_buy_imbalance`, `taker_buy_ma5`, `taker_buy_ma20`, `tbi_diff`, `signed_vol_delta_1m`, `rel_trades_20`, `rel_avg_trade_size`
7. **Cyclical UTC Time Encodings (`4`):** `sin_hour`, `cos_hour`, `sin_dow`, `cos_dow`

---

## How Prediction & Range Estimation Work

1. **Chronological Split & Scaling:** Historical data is split `70%` train / `15%` validation / `15%` unseen test. `RobustScaler` (clipped to `[-8, 8]`) and `StandardScaler` are fit strictly on training rows.
2. **Hybrid Ensemble:**
   - **Neural Branch:** `Input(36, 50)` → `Conv1D(48, causal, swish)` → `LayerNorm` → `SpatialDropout1D(0.15)` → `Bidirectional(LSTM(48))` → `LayerNorm` → `[Last Step + Softmax Attention]` → `Dense(32)` → `ret` (Huber loss) & `dir` (BCE loss) heads.
   - **Tabular Branch:** `HistGradientBoostingRegressor (L1)`, `Ridge`, and `HistGradientBoostingClassifier` trained on `[step t, 4-bar mean, 12-bar mean]` with exponential recency sample weights.
   - **Calibration:** `LogisticRegression` calibrates `UP` probability and validation-selected convex weights blend all predictions into $\hat{r}_{t+1}$.
3. **Price & Range Reconstruction:**
   $$\hat{P}_{t+1} = P_t \cdot e^{\hat{r}_{t+1}}$$
   $$\sigma = \max\big(0.65 \cdot \text{vol}_{20} + 0.35 \cdot (0.5 \cdot \text{ATR}_{14}),\; 0.85 \cdot \text{vol}_{10},\; 10^{-4}\big)$$
   $$\text{Predicted Low} = \hat{P}_{t+1} \cdot e^{-1.60\sigma}, \qquad \text{Predicted High} = \hat{P}_{t+1} \cdot e^{+1.60\sigma}$$

---

## Installation & Running

### 1. Setup Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run the Web Dashboard
```powershell
.\venv\Scripts\python.exe app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

### 3. CLI Training (Optional)
All 24 models (`1m`, `5m`, `15m` × 8 coins) are already included in `models/`. To retrain from live Binance data:
```powershell
# Train a single symbol (default 1m interval from config.py)
.\venv\Scripts\python.exe train.py BTCUSDT

# Train all 8 symbols
.\venv\Scripts\python.exe train_all_models.py
```

### 4. Inspect Live Data or Computed Features via CLI
```powershell
# View latest 10 live Binance candles
.\venv\Scripts\python.exe -c "from data import get_klines; print(get_klines('BTCUSDT', '1m', 10, include_live=True)[['open_time','open','high','low','close','volume']])"

# View all 50 computed features for the latest 5 completed candles
.\venv\Scripts\python.exe -c "import pandas as pd; pd.set_option('display.max_rows', 60); from config import FEATURES; from data import get_klines; from features import add_features; print(add_features(get_klines('BTCUSDT', '1m', 300))[FEATURES].tail(5).T)"
```

---

## REST API Endpoints

All API routes accept an optional `?interval=1m|5m|15m` query parameter:

| Endpoint | Method | Description |
| --- | --- | --- |
| `/` | `GET` | Renders the CryptoScope web dashboard |
| `/api/coins` | `GET` | Lists all 8 cryptocurrencies and training status for the selected interval |
| `/api/market/<symbol>` | `GET` | Returns the latest 120 completed OHLCV candles |
| `/api/predict/<symbol>` | `GET` | Returns current price, `[predicted_low, predicted_high]`, midpoint, change `%`, `UP/DOWN`, and confidence `%` |
| `/api/verification/<symbol>` | `GET` | Audits the last 10 completed candles (`Actual vs. Predicted Range`, error, direction & range hit rate) |
| `/api/model/<symbol>` | `GET` | Returns stored out-of-sample evaluation metrics from `metadata.pkl` |
| `/api/train/<symbol>` | `POST` | Trains/retrains a symbol model for the selected interval |

---

## Disclaimer

This project is built for educational and machine learning demonstration purposes only and does not constitute financial or investment advice.
