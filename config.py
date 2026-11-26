# =========================================================
# CRYPTOCURRENCY CONFIGURATION
# =========================================================

CRYPTOCURRENCIES = {
    "BTCUSDT": {
        "name": "Bitcoin",
        "symbol": "BTC",
        "icon": "₿",
        "class": "bitcoin"
    },
    "ETHUSDT": {
        "name": "Ethereum",
        "symbol": "ETH",
        "icon": "Ξ",
        "class": "ethereum"
    },
    "BNBUSDT": {
        "name": "BNB",
        "symbol": "BNB",
        "icon": "BNB",
        "class": "bnb"
    },
    "SOLUSDT": {
        "name": "Solana",
        "symbol": "SOL",
        "icon": "S",
        "class": "solana"
    },
    "XRPUSDT": {
        "name": "XRP",
        "symbol": "XRP",
        "icon": "X",
        "class": "xrp"
    },
    "ADAUSDT": {
        "name": "Cardano",
        "symbol": "ADA",
        "icon": "A",
        "class": "cardano"
    },
    "DOGEUSDT": {
        "name": "Dogecoin",
        "symbol": "DOGE",
        "icon": "Ð",
        "class": "dogecoin"
    },
    "AVAXUSDT": {
        "name": "Avalanche",
        "symbol": "AVAX",
        "icon": "A",
        "class": "avalanche"
    }
}


# =========================================================
# TIME INTERVAL
# =========================================================

INTERVAL = "1m"

SUPPORTED_INTERVALS = ["1m", "5m", "15m"]

INTERVAL_LABELS = {
    "1m": "1 minute",
    "5m": "5 minute",
    "15m": "15 minute",
    "30m": "30 minute",
    "1h": "1 hour",
    "4h": "4 hour",
}
INTERVAL_DISPLAY = INTERVAL_LABELS.get(INTERVAL, INTERVAL)

# Prediction interval multiplier (~85-87% empirical coverage across 1m/5m/15m)
RANGE_Z_SCORE = 1.60

# Minimum calibrated probability (%) to issue an active UP/DOWN signal
DIRECTION_CONFIDENCE_THRESHOLD = 52.0


# =========================================================
# HISTORICAL DATA CONFIGURATION
# =========================================================

# Number of historical days used for model training (45 days of 1m candles)
TRAINING_DAYS = 45

# 1-minute candles per day (24 * 60 = 1,440)
CANDLES_PER_DAY = 1440

# Total historical candles for training (45 days = 64,800 candles)
TRAINING_DATA_LIMIT = TRAINING_DAYS * CANDLES_PER_DAY


# =========================================================
# LIVE DASHBOARD DATA
# =========================================================

# Candles displayed in the live chart
LIVE_DATA_LIMIT = 120

# Completed candles fetched when generating a prediction
# (96-bar indicator warmup + 36-bar sequence length + margin)
PREDICTION_DATA_LIMIT = 300

# Backward compatibility for older training scripts
DATA_LIMIT = TRAINING_DATA_LIMIT


# =========================================================
# HYBRID SEQUENCE MODEL CONFIGURATION
# =========================================================

SEQUENCE_LENGTH = 36

MODEL_EPOCHS = 18

BATCH_SIZE = 128


# =========================================================
# STATIONARY MULTI-HORIZON & ORDER-FLOW FEATURES
# =========================================================

FEATURES = [
    # Multi-horizon log returns & standardized move
    "ret_1",
    "ret_2",
    "ret_3",
    "ret_5",
    "ret_10",
    "ret_20",
    "ret_48",
    "ret1_z",
    # Relative moving-average & VWAP deviations (scale-invariant)
    "dist_sma_5",
    "dist_ema_5",
    "dist_sma_10",
    "dist_ema_10",
    "dist_sma_20",
    "dist_ema_20",
    "dist_sma_50",
    "dist_ema_50",
    "dist_vwap_5",
    "dist_vwap_15",
    # Normalized MACD
    "macd_norm",
    "macd_hist_norm",
    # Normalized RSI oscillators
    "rsi_7",
    "rsi_14",
    "rsi_28",
    # Stationary Bollinger Bands
    "bb_pos",
    "bb_width",
    # Volatility & ATR regime
    "atr_14_norm",
    "vol_10",
    "vol_20",
    "vol_60",
    "vol_ratio",
    # Candle microstructure & stochastic position
    "hl_range",
    "body_ratio",
    "candle_ret",
    "upper_wick",
    "lower_wick",
    "close_pos",
    "stoch_14",
    # Volume & Binance taker order-flow imbalance
    "rel_vol_20",
    "rel_vol_5",
    "taker_buy_imbalance",
    "taker_buy_ma5",
    "taker_buy_ma20",
    "tbi_diff",
    "signed_vol_delta_1m",
    "rel_trades_20",
    "rel_avg_trade_size",
    # Cyclical UTC time-of-day & day-of-week
    "sin_hour",
    "cos_hour",
    "sin_dow",
    "cos_dow",
]
