import numpy as np
import pandas as pd

from config import FEATURES


def add_features(df):
    """Build stationary multi-horizon, technical, candle-geometry, order-flow,
    and cyclical time features from OHLCV klines."""
    df = df.copy()
    numeric_cols = [
        "open", "high", "low", "close", "volume",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    eps = 1e-9
    c = df["close"]
    o = df["open"] if "open" in df.columns else c
    h = df["high"] if "high" in df.columns else c
    l = df["low"] if "low" in df.columns else c
    v = df["volume"]

    # Multi-horizon log returns
    df["ret_1"] = np.log(c / c.shift(1))
    for k in (2, 3, 5, 10, 20, 48):
        df[f"ret_{k}"] = np.log(c / c.shift(k))

    # Relative moving-average deviations
    for k in (5, 10, 20, 50):
        sma = c.rolling(k).mean()
        ema = c.ewm(span=k, adjust=False).mean()
        df[f"dist_sma_{k}"] = (c - sma) / (sma + eps)
        df[f"dist_ema_{k}"] = (c - ema) / (ema + eps)

    # Normalized MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_sig = macd.ewm(span=9, adjust=False).mean()
    df["macd_norm"] = macd / (c + eps)
    df["macd_hist_norm"] = (macd - macd_sig) / (c + eps)

    # Wilder-smoothed RSI centered to [-1, 1]
    delta = c.diff()
    for w in (7, 14, 28):
        gain = delta.clip(lower=0).ewm(alpha=1.0 / w, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / w, adjust=False).mean()
        rs = gain / (loss + eps)
        df[f"rsi_{w}"] = (100.0 - 100.0 / (1.0 + rs) - 50.0) / 50.0

    # Stationary Bollinger Bands
    mid = c.rolling(20).mean()
    std = c.rolling(20).std()
    bb_up = mid + 2.0 * std
    bb_dn = mid - 2.0 * std
    df["bb_pos"] = (c - bb_dn) / (bb_up - bb_dn + eps) - 0.5
    df["bb_width"] = (bb_up - bb_dn) / (mid + eps)

    # Volatility & ATR regime
    tr = pd.concat(
        [h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    df["atr_14_norm"] = tr.rolling(14).mean() / (c + eps)
    df["vol_10"] = df["ret_1"].rolling(10).std()
    df["vol_20"] = df["ret_1"].rolling(20).std()
    df["vol_60"] = df["ret_1"].rolling(60).std()
    df["vol_ratio"] = df["vol_10"] / (df["vol_60"] + eps)
    df["ret1_z"] = df["ret_1"] / (df["vol_20"] + eps)

    # Short-horizon VWAP deviations
    pv = ((h + l + c) / 3.0) * v
    for k in (5, 15):
        vwap = pv.rolling(k).sum() / (v.rolling(k).sum() + eps)
        df[f"dist_vwap_{k}"] = (c - vwap) / (vwap + eps)

    # Candle geometry & stochastic oscillator
    hl = h - l + eps
    df["hl_range"] = (h - l) / (c + eps)
    df["body_ratio"] = (c - o) / hl
    df["candle_ret"] = np.log((c + eps) / (o + eps))
    df["upper_wick"] = (h - np.maximum(o, c)) / hl
    df["lower_wick"] = (np.minimum(o, c) - l) / hl
    df["close_pos"] = (c - l) / hl - 0.5

    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    df["stoch_14"] = (c - low14) / (high14 - low14 + eps) - 0.5

    # Volume & Binance taker order flow
    log_v = np.log1p(v.clip(lower=0))
    df["rel_vol_20"] = log_v - log_v.rolling(20).mean()
    df["rel_vol_5"] = log_v - log_v.rolling(5).mean()

    if "taker_buy_base" in df.columns and df["taker_buy_base"].notna().any():
        tbr = (df["taker_buy_base"] / (v + eps)).clip(0.0, 1.0)
        df["taker_buy_imbalance"] = tbr - 0.5
        df["taker_buy_ma5"] = tbr.rolling(5).mean() - 0.5
        df["taker_buy_ma20"] = tbr.rolling(20).mean() - 0.5
        df["tbi_diff"] = df["taker_buy_imbalance"] - df["taker_buy_ma5"]
        df["signed_vol_delta_1m"] = (2.0 * df["taker_buy_base"] - v) / (v.rolling(20).mean() + eps)
    else:
        df["taker_buy_imbalance"] = 0.0
        df["taker_buy_ma5"] = 0.0
        df["taker_buy_ma20"] = 0.0
        df["tbi_diff"] = 0.0
        df["signed_vol_delta_1m"] = 0.0

    if "trades" in df.columns and df["trades"].notna().any():
        log_tr = np.log1p(df["trades"].clip(lower=0))
        df["rel_trades_20"] = log_tr - log_tr.rolling(20).mean()
        if "quote_volume" in df.columns and df["quote_volume"].notna().any():
            avg_sz = df["quote_volume"] / (df["trades"] + 1.0)
            log_sz = np.log1p(avg_sz.clip(lower=0))
            df["rel_avg_trade_size"] = log_sz - log_sz.rolling(20).mean()
        else:
            df["rel_avg_trade_size"] = 0.0
    else:
        df["rel_trades_20"] = 0.0
        df["rel_avg_trade_size"] = 0.0

    # Cyclical UTC time features
    if "open_time" in df.columns and df["open_time"].notna().any():
        ts = pd.to_datetime(df["open_time"], utc=True)
        hour = ts.dt.hour + ts.dt.minute / 60.0
        df["sin_hour"] = np.sin(2.0 * np.pi * hour / 24.0)
        df["cos_hour"] = np.cos(2.0 * np.pi * hour / 24.0)
        dow = ts.dt.dayofweek
        df["sin_dow"] = np.sin(2.0 * np.pi * dow / 7.0)
        df["cos_dow"] = np.cos(2.0 * np.pi * dow / 7.0)
    else:
        df["sin_hour"] = 0.0
        df["cos_hour"] = 0.0
        df["sin_dow"] = 0.0
        df["cos_dow"] = 0.0

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.dropna(subset=FEATURES).reset_index(drop=True)
