"""Hybrid Conv1D-BiLSTM-Attention + Ensemble forecasting for CryptoAI.

Predicts the next candle's log return and direction probability from a sequence
of stationary multi-horizon, technical, candle-geometry, order-flow, and
cyclical time features:
    predicted_close[t+1] = close[t] * exp(predicted_log_return[t+1])
"""

import json
import os
import shutil

import h5py
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from tensorflow.keras import callbacks, layers, losses, models, optimizers, regularizers
from tensorflow.keras.models import load_model

from config import (
    BATCH_SIZE,
    DIRECTION_CONFIDENCE_THRESHOLD,
    FEATURES,
    INTERVAL,
    MODEL_EPOCHS,
    RANGE_Z_SCORE,
    SEQUENCE_LENGTH,
)
from data import INTERVAL_MS


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.join(PROJECT_ROOT, "models")
HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
TARGET_NAME = "next_candle_log_return_v4"

_MODEL_CACHE = {}


def _is_hdf5(path):
    try:
        with open(path, "rb") as file:
            return file.read(8) == HDF5_SIGNATURE
    except OSError:
        return False


def _patch_legacy_lstm_config(path):
    """Patch an old HDF5 LSTM config if it contains obsolete time_major."""
    if not _is_hdf5(path):
        return

    with h5py.File(path, "r") as h5_file:
        if "model_config" not in h5_file.attrs:
            return
        raw = h5_file.attrs["model_config"]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        config = json.loads(raw)

    def walk(node):
        changed = 0
        if isinstance(node, dict):
            layer_config = node.get("config")
            if (
                node.get("class_name") in {"LSTM", "CuDNNLSTM"}
                and isinstance(layer_config, dict)
                and layer_config.get("time_major") is False
            ):
                del layer_config["time_major"]
                changed += 1
            for value in node.values():
                changed += walk(value)
        elif isinstance(node, list):
            for value in node:
                changed += walk(value)
        return changed

    if not walk(config):
        return

    backup = path + ".before_time_major_fix.bak"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)

    encoded = json.dumps(config, separators=(",", ":")).encode("utf-8")
    with h5py.File(path, "r+") as h5_file:
        del h5_file.attrs["model_config"]
        h5_file.attrs.create("model_config", np.bytes_(encoded))


def paths(symbol, interval=INTERVAL):
    interval = interval or INTERVAL
    if interval == INTERVAL:
        folder = os.path.join(MODEL_ROOT, symbol)
    else:
        folder = os.path.join(MODEL_ROOT, interval, symbol)
    os.makedirs(folder, exist_ok=True)

    keras_path = os.path.join(folder, "model.keras")
    h5_path = os.path.join(folder, "model.h5")

    if os.path.isfile(keras_path) and _is_hdf5(keras_path):
        if not os.path.exists(h5_path):
            os.replace(keras_path, h5_path)

    model_path = h5_path if os.path.isfile(h5_path) else keras_path
    return {
        "folder": folder,
        "model": model_path,
        "sx": os.path.join(folder, "scaler_X.pkl"),
        "sy": os.path.join(folder, "scaler_y.pkl"),
        "ensemble": os.path.join(folder, "ensemble.pkl"),
        "meta": os.path.join(folder, "metadata.pkl"),
    }


def _prepare_data(df):
    """Apply feature engineering and validate numeric values."""
    from features import add_features

    if df is None or df.empty:
        raise ValueError("No market candles were supplied for model training.")

    prepared = add_features(df.copy())
    required = list(FEATURES) + ["close"]
    missing = [column for column in required if column not in prepared.columns]
    if missing:
        raise ValueError(
            "Missing model columns after add_features(): " + ", ".join(missing)
        )

    for column in required:
        prepared[column] = np.asarray(prepared[column], dtype=float)

    prepared = prepared.replace([np.inf, -np.inf], np.nan)
    prepared = prepared.dropna(subset=required).reset_index(drop=True)

    if (prepared["close"] <= 0).any():
        raise ValueError("Closing prices must be positive to calculate log returns.")

    return prepared


def _tabular_summary(X_seq):
    """Extract multi-horizon summary features from scaled sequences."""
    return np.hstack([
        X_seq[:, -1, :],
        X_seq[:, -4:, :].mean(axis=1),
        X_seq[:, -12:, :].mean(axis=1),
    ])


def _make_sequences(X_scaled, targets_raw, y_scaled, prices, start_idx, end_idx, length):
    """Build windows [i - length + 1 .. i] predicting target[i] = log(close[i+1]/close[i])."""
    start = max(int(start_idx), int(length) - 1)
    end = min(int(end_idx), len(prices) - 1)

    if end <= start:
        return (
            np.empty((0, length, X_scaled.shape[1]), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=float),
            np.empty((0,), dtype=float),
        )

    windows_view = np.lib.stride_tricks.sliding_window_view(X_scaled, length, axis=0)
    # sliding_window_view shape: (N - length + 1, features, length)
    windows_view = np.transpose(windows_view, (0, 2, 1))

    indices = np.arange(start, end)
    window_keys = indices - (length - 1)

    X_windows = np.ascontiguousarray(windows_view[window_keys], dtype=np.float32)
    y_targets_scaled = np.asarray(y_scaled[indices], dtype=np.float32)
    y_targets_raw = np.asarray(targets_raw[indices], dtype=np.float32)
    previous_closes = np.asarray(prices[indices], dtype=float)
    actual_closes = np.asarray(prices[indices + 1], dtype=float)
    return X_windows, y_targets_scaled, y_targets_raw, previous_closes, actual_closes


def _build_hybrid_nn(sequence_length, n_features):
    """Build a Conv1D + Bidirectional LSTM + Attention multi-task network."""
    inp = layers.Input(shape=(sequence_length, n_features))
    x = layers.Conv1D(48, kernel_size=3, padding="causal", activation="swish")(inp)
    x = layers.LayerNormalization()(x)
    x = layers.SpatialDropout1D(0.15)(x)

    x = layers.Bidirectional(layers.LSTM(48, return_sequences=True))(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(0.2)(x)

    last_step = layers.Flatten()(layers.Cropping1D((sequence_length - 1, 0))(x))
    attn_scores = layers.Dense(1)(x)
    attn_weights = layers.Softmax(axis=1)(attn_scores)
    attn_pool = layers.Flatten()(layers.Dot(axes=(1, 1))([attn_weights, x]))

    h = layers.Concatenate()([last_step, attn_pool])
    h = layers.Dense(32, activation="swish", kernel_regularizer=regularizers.l2(1e-4))(h)
    h = layers.Dropout(0.2)(h)

    out_ret = layers.Dense(1, name="ret")(h)
    out_dir = layers.Dense(1, activation="sigmoid", name="dir")(h)

    model = models.Model(inputs=inp, outputs=[out_ret, out_dir])
    model.compile(
        optimizer=optimizers.AdamW(learning_rate=8e-4, weight_decay=1e-4, clipnorm=1.0),
        loss={"ret": losses.Huber(delta=1.0), "dir": "binary_crossentropy"},
        loss_weights={"ret": 1.0, "dir": 0.4},
    )
    return model


def _regression_metrics(actual, predicted):
    """Return regression metrics in original price units."""
    actual = np.asarray(actual, dtype=float).ravel()
    predicted = np.asarray(predicted, dtype=float).ravel()

    if len(actual) == 0 or len(actual) != len(predicted):
        raise ValueError("Cannot calculate metrics: actual/predicted arrays are empty or mismatched.")

    absolute_errors = np.abs(actual - predicted)
    percentage_errors = absolute_errors / np.maximum(np.abs(actual), 1e-8) * 100.0

    mae = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    r2 = float(r2_score(actual, predicted)) if len(actual) >= 2 else float("nan")
    mape = float(np.mean(percentage_errors))
    smape = float(
        np.mean(
            200.0 * absolute_errors
            / np.maximum(np.abs(actual) + np.abs(predicted), 1e-8)
        )
    )
    wape = float(
        np.sum(absolute_errors) / max(float(np.sum(np.abs(actual))), 1e-8) * 100.0
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "smape": smape,
        "wape": wape,
        "accuracy_within_0_5_percent": float(np.mean(percentage_errors <= 0.5) * 100.0),
        "accuracy_within_1_percent": float(np.mean(percentage_errors <= 1.0) * 100.0),
        "accuracy_within_2_percent": float(np.mean(percentage_errors <= 2.0) * 100.0),
    }


def _direction_metrics(actual_returns, predicted_returns, up_probs):
    """Compute overall and high-confidence directional accuracy on non-flat moves."""
    nz = actual_returns != 0
    if not np.any(nz):
        return {
            "direction_accuracy": 50.0,
            "direction_accuracy_prob": 50.0,
            "high_confidence_direction_accuracy": 50.0,
            "high_confidence_threshold": 54.0,
            "high_confidence_samples": 0,
        }

    actual_up = actual_returns[nz] > 0
    pred_up_ret = predicted_returns[nz] > 0
    pred_up_prob = up_probs[nz] > 0.5

    dir_acc_ret = float(np.mean(pred_up_ret == actual_up) * 100.0)
    dir_acc_prob = float(np.mean(pred_up_prob == actual_up) * 100.0)

    probs_nz = up_probs[nz]
    high_conf_acc = dir_acc_prob
    high_conf_thresh = 54.0
    high_conf_count = 0

    for thresh in (0.04, 0.02):
        mask = np.abs(probs_nz - 0.5) >= thresh
        if int(np.sum(mask)) >= 20:
            high_conf_acc = float(np.mean((probs_nz[mask] > 0.5) == actual_up[mask]) * 100.0)
            high_conf_thresh = float((0.5 + thresh) * 100.0)
            high_conf_count = int(np.sum(mask))
            break

    return {
        "direction_accuracy": dir_acc_ret,
        "direction_accuracy_prob": dir_acc_prob,
        "high_confidence_direction_accuracy": high_conf_acc,
        "high_confidence_threshold": high_conf_thresh,
        "high_confidence_samples": high_conf_count,
    }


def _combine_predictions(
    nn_ret, hgbr_ret, ridge_ret, nn_dir, hgbc_dir, cal_dir, ret_scale, weights
):
    """Combine NN and tabular ensemble outputs into return & UP probability."""
    Z = np.column_stack([
        nn_dir - 0.5,
        hgbc_dir - 0.5,
        nn_ret / ret_scale,
        hgbr_ret / ret_scale,
        ridge_ret / ret_scale,
    ])
    ens_dir_prob = cal_dir.predict_proba(Z)[:, 1]
    dir_ret_component = (ens_dir_prob - 0.5) * 2.0 * (0.15 * ret_scale)
    w_nn, w_hg, w_rg, w_dir = weights
    predicted_returns = (
        w_nn * nn_ret
        + w_hg * hgbr_ret
        + w_rg * ridge_ret
        + w_dir * dir_ret_component
    )
    return predicted_returns, ens_dir_prob


def _select_blend_weights(
    nn_val_ret, hg_val_ret, rg_val_ret, ens_val_dir, previous_val, actual_val, ret_scale
):
    """Select convex ensemble weights on validation split to maximize skill."""
    dir_val_ret = (ens_val_dir - 0.5) * 2.0 * (0.15 * ret_scale)
    b_mae = mean_absolute_error(actual_val, previous_val)
    b_rmse = np.sqrt(mean_squared_error(actual_val, previous_val))

    candidates = [
        (0.25, 0.35, 0.20, 0.20),
        (0.30, 0.25, 0.25, 0.20),
        (0.25, 0.55, 0.00, 0.20),
        (0.20, 0.45, 0.20, 0.15),
        (0.30, 0.45, 0.00, 0.25),
        (0.20, 0.65, 0.00, 0.15),
        (0.15, 0.70, 0.00, 0.15),
        (0.40, 0.40, 0.00, 0.20),
        (0.20, 0.35, 0.30, 0.15),
    ]

    best_weights = candidates[0]
    best_score = -float("inf")
    for w in candidates:
        base_ret = w[0] * nn_val_ret + w[1] * hg_val_ret + w[2] * rg_val_ret + w[3] * dir_val_ret
        for amp in (0.35, 0.65, 1.00):
            scaled_w = (w[0] * amp, w[1] * amp, w[2] * amp, w[3] * amp)
            pret = np.clip(amp * base_ret, -0.5, 0.5)
            pclose = previous_val * np.exp(pret)
            mae_skill = 1.0 - mean_absolute_error(actual_val, pclose) / max(b_mae, 1e-12)
            rmse_skill = 1.0 - np.sqrt(mean_squared_error(actual_val, pclose)) / max(b_rmse, 1e-12)
            score = mae_skill + 0.5 * rmse_skill
            if score > best_score:
                best_score = score
                best_weights = scaled_w
    return best_weights


def train_symbol(df, symbol, interval=INTERVAL, epochs=None, batch_size=None):
    """Train and evaluate the hybrid forecasting ensemble for one symbol."""
    interval = interval or INTERVAL
    p = paths(symbol, interval=interval)
    df = _prepare_data(df)

    sequence_length = int(SEQUENCE_LENGTH)
    if sequence_length < 12:
        raise ValueError("SEQUENCE_LENGTH must be at least 12.")

    X_raw = df[list(FEATURES)].to_numpy(dtype=np.float32)
    prices = df["close"].to_numpy(dtype=float)
    row_count = len(df)

    if row_count < sequence_length + 300:
        raise ValueError(
            f"Not enough usable candles for {symbol}: {row_count}. "
            f"At least {sequence_length + 300} are recommended."
        )

    # Target at index i is log(close[i+1] / close[i])
    target_returns = np.full(row_count, np.nan, dtype=np.float32)
    target_returns[:-1] = np.log(prices[1:] / prices[:-1]).astype(np.float32)

    # Strict chronological split: 70% train, 15% validation, 15% final test
    train_end = int(row_count * 0.70)
    validation_end = int(row_count * 0.85)
    if train_end <= sequence_length + 20 or validation_end <= train_end or validation_end >= row_count - 2:
        raise ValueError("Dataset is too small for the chronological train/validation/test split.")

    # Fit scalers strictly on training rows
    sx = RobustScaler()
    sx.fit(X_raw[:train_end])
    X_scaled = np.clip(sx.transform(X_raw), -8.0, 8.0).astype(np.float32)

    train_return_slice = target_returns[sequence_length - 1:train_end]
    if not np.isfinite(train_return_slice).all():
        raise ValueError("Training return targets contain NaN or infinite values.")

    sy = StandardScaler()
    sy.fit(train_return_slice.reshape(-1, 1))
    y_scaled = np.zeros(row_count, dtype=np.float32)
    valid_targets = np.isfinite(target_returns)
    y_scaled[valid_targets] = sy.transform(
        target_returns[valid_targets].reshape(-1, 1)
    ).ravel().astype(np.float32)

    X_train, y_train_sc, y_train_raw, _, _ = _make_sequences(
        X_scaled, target_returns, y_scaled, prices, sequence_length - 1, train_end, sequence_length
    )
    X_val, y_val_sc, y_val_raw, previous_val, actual_val = _make_sequences(
        X_scaled, target_returns, y_scaled, prices, train_end, validation_end, sequence_length
    )
    X_test, y_test_sc, y_test_raw, previous_test, actual_test = _make_sequences(
        X_scaled, target_returns, y_scaled, prices, validation_end, row_count - 1, sequence_length
    )

    if min(len(X_train), len(X_val), len(X_test)) == 0:
        raise ValueError("One of the train/validation/test splits has no sequences.")

    try:
        from tensorflow.keras.utils import set_random_seed
        set_random_seed(42)
    except Exception:
        pass

    nn_model = _build_hybrid_nn(sequence_length, len(FEATURES))
    cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            min_delta=1e-5,
            restore_best_weights=True,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    history = nn_model.fit(
        X_train,
        {"ret": y_train_sc, "dir": (y_train_raw > 0).astype(np.float32)},
        validation_data=(
            X_val,
            {"ret": y_val_sc, "dir": (y_val_raw > 0).astype(np.float32)},
        ),
        epochs=int(epochs if epochs is not None else MODEL_EPOCHS),
        batch_size=int(batch_size if batch_size is not None else BATCH_SIZE),
        callbacks=cbs,
        shuffle=True,
        verbose=1,
    )

    nn_val_sc, nn_val_dir = nn_model.predict(X_val, verbose=0)
    nn_test_sc, nn_test_dir = nn_model.predict(X_test, verbose=0)
    nn_val_ret = sy.inverse_transform(nn_val_sc).ravel()
    nn_test_ret = sy.inverse_transform(nn_test_sc).ravel()
    nn_val_dir = nn_val_dir.ravel()
    nn_test_dir = nn_test_dir.ravel()

    # Multi-horizon tabular models for return stabilization & calibrated direction
    X_tab_train = _tabular_summary(X_train)
    X_tab_val = _tabular_summary(X_val)
    X_tab_test = _tabular_summary(X_test)

    # Exponential recency sample weights so recent market regimes carry more weight
    recency_weights = np.exp(np.linspace(0.0, 1.1, len(X_tab_train)))

    hgbr = HistGradientBoostingRegressor(
        loss="absolute_error",
        max_iter=200,
        learning_rate=0.02,
        max_leaf_nodes=15,
        min_samples_leaf=250,
        l2_regularization=100.0,
        early_stopping=True,
        random_state=42,
    )
    hgbr.fit(X_tab_train, y_train_raw, sample_weight=recency_weights)
    hg_val_ret = hgbr.predict(X_tab_val)
    hg_test_ret = hgbr.predict(X_tab_test)

    ridge = Ridge(alpha=50000.0)
    ridge.fit(X_tab_train, y_train_raw, sample_weight=recency_weights)
    rg_val_ret = ridge.predict(X_tab_val)
    rg_test_ret = ridge.predict(X_tab_test)

    hgbc = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.02,
        max_leaf_nodes=15,
        min_samples_leaf=250,
        l2_regularization=80.0,
        early_stopping=True,
        random_state=42,
    )
    hgbc.fit(X_tab_train, (y_train_raw > 0).astype(int), sample_weight=recency_weights)
    hg_val_dir = hgbc.predict_proba(X_tab_val)[:, 1]
    hg_test_dir = hgbc.predict_proba(X_tab_test)[:, 1]

    # Calibrate direction probability stacker strictly on validation data
    ret_scale = float(np.std(y_train_raw) + 1e-9)
    Z_val = np.column_stack([
        nn_val_dir - 0.5,
        hg_val_dir - 0.5,
        nn_val_ret / ret_scale,
        hg_val_ret / ret_scale,
        rg_val_ret / ret_scale,
    ])
    cal_dir = LogisticRegression(C=0.5)
    cal_dir.fit(Z_val, (y_val_raw > 0).astype(int))
    ens_val_dir = cal_dir.predict_proba(Z_val)[:, 1]

    blend_weights = _select_blend_weights(
        nn_val_ret, hg_val_ret, rg_val_ret, ens_val_dir, previous_val, actual_val, ret_scale
    )

    predicted_returns, up_probs = _combine_predictions(
        nn_test_ret, hg_test_ret, rg_test_ret,
        nn_test_dir, hg_test_dir, cal_dir, ret_scale, blend_weights
    )

    predicted_returns = np.clip(predicted_returns, -0.5, 0.5)
    predicted_prices = previous_test * np.exp(predicted_returns)

    model_metrics = _regression_metrics(actual_test, predicted_prices)
    baseline_metrics = _regression_metrics(actual_test, previous_test)
    dir_metrics = _direction_metrics(y_test_raw, predicted_returns, up_probs)

    mae_skill = float(1.0 - model_metrics["mae"] / max(baseline_metrics["mae"], 1e-12))
    rmse_skill = float(1.0 - model_metrics["rmse"] / max(baseline_metrics["rmse"], 1e-12))
    mape_skill = float(1.0 - model_metrics["mape"] / max(baseline_metrics["mape"], 1e-12))

    # Save native Keras model, scalers, and ensemble components
    native_model_path = os.path.join(p["folder"], "model.keras")
    nn_model.save(native_model_path)
    old_model_path = p["model"]
    if old_model_path != native_model_path and os.path.isfile(old_model_path):
        os.remove(old_model_path)

    joblib.dump(sx, p["sx"])
    joblib.dump(sy, p["sy"])
    joblib.dump(
        {
            "hgbr": hgbr,
            "ridge": ridge,
            "hgbc": hgbc,
            "cal_dir": cal_dir,
            "ret_scale": ret_scale,
            "blend_weights": blend_weights,
        },
        p["ensemble"],
    )

    # Store recent unseen test predictions for dashboard backtest chart
    backtest_count = min(80, len(actual_test))
    test_start_row = max(validation_end, sequence_length - 1)
    if "open_time" in df.columns:
        bt_times = [
            int(ts.timestamp())
            for ts in pd.to_datetime(
                df["open_time"].iloc[test_start_row + 1:row_count].tail(backtest_count),
                utc=True,
            )
        ]
    else:
        bt_times = list(range(backtest_count))

    recent_backtest = {
        "timestamps": bt_times[-backtest_count:],
        "actual": [float(v) for v in actual_test[-backtest_count:]],
        "predicted": [float(v) for v in predicted_prices[-backtest_count:]],
    }

    metrics = {
        **model_metrics,
        **dir_metrics,
        "baseline_mae": baseline_metrics["mae"],
        "baseline_rmse": baseline_metrics["rmse"],
        "baseline_r2": baseline_metrics["r2"],
        "baseline_mape": baseline_metrics["mape"],
        "baseline_smape": baseline_metrics["smape"],
        "baseline_wape": baseline_metrics["wape"],
        "baseline_accuracy_within_1_percent": baseline_metrics["accuracy_within_1_percent"],
        "mae_skill_vs_baseline": mae_skill,
        "rmse_skill_vs_baseline": rmse_skill,
        "mape_skill_vs_baseline": mape_skill,
    }

    metadata = {
        "symbol": symbol,
        "interval": interval,
        "target": TARGET_NAME,
        "architecture": "Conv1D-BiLSTM-Attention + Ridge/HGBC Ensemble",
        "forecast_type": "next_candle_close",
        "features": list(FEATURES),
        "sequence_length": sequence_length,
        "metrics": metrics,
        **metrics,
        "recent_backtest": recent_backtest,
        "train_samples": int(len(X_train)),
        "validation_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "train_rows": int(train_end),
        "validation_end_row": int(validation_end),
        "epochs_trained": int(len(history.history.get("loss", []))),
    }
    joblib.dump(metadata, p["meta"])
    _MODEL_CACHE.pop((symbol, interval), None)
    return metadata


def model_exists(symbol, interval=INTERVAL):
    """Only mark a model as trained if all v4 artifacts exist."""
    p = paths(symbol, interval=interval)
    required_files_exist = all(
        os.path.isfile(p[key]) and os.path.getsize(p[key]) > 0
        for key in ("model", "sx", "sy", "ensemble", "meta")
    )
    if not required_files_exist:
        return False

    try:
        metadata = joblib.load(p["meta"])
        return metadata.get("target") == TARGET_NAME
    except Exception:
        return False


def load_metadata(symbol, interval=INTERVAL):
    if not model_exists(symbol, interval=interval):
        return None
    return joblib.load(paths(symbol, interval=interval)["meta"])


def _load_artifacts(symbol, interval=INTERVAL):
    """Load and cache model artifacts per (symbol, interval), refreshing on mtime change."""
    interval = interval or INTERVAL
    cache_key = (symbol, interval)
    p = paths(symbol, interval=interval)
    mtime = os.path.getmtime(p["meta"])
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None and cached["mtime"] == mtime:
        return cached

    _patch_legacy_lstm_config(p["model"])
    nn_model = load_model(p["model"], compile=False)
    sx = joblib.load(p["sx"])
    sy = joblib.load(p["sy"])
    ensemble = joblib.load(p["ensemble"])

    bundle = {
        "mtime": mtime,
        "nn_model": nn_model,
        "sx": sx,
        "sy": sy,
        "ensemble": ensemble,
    }
    _MODEL_CACHE[cache_key] = bundle
    return bundle


def _compute_price_range(
    predicted_price, vol_20, atr_14_norm, vol_10=0.0, z_score=RANGE_Z_SCORE
):
    """Compute a volatility-adaptive prediction interval [low, high]."""
    base_sigma = 0.65 * float(vol_20) + 0.35 * (float(atr_14_norm) * 0.5)
    regime_sigma = 0.85 * float(vol_10)
    sigma = float(max(base_sigma, regime_sigma, 1e-4))
    half_log = float(z_score) * sigma
    predicted_low = float(predicted_price * np.exp(-half_log))
    predicted_high = float(predicted_price * np.exp(half_log))
    band_pct = float((np.exp(half_log) - 1.0) * 100.0)
    return predicted_low, predicted_high, band_pct


def _infer_batch(prepared, X_seq_batch, bundle):
    """Run batch inference with loaded ensemble artifacts."""
    nn_ret_tensor, nn_dir_tensor = bundle["nn_model"](X_seq_batch, training=False)
    nn_ret = bundle["sy"].inverse_transform(nn_ret_tensor.numpy()).ravel()
    nn_dir = nn_dir_tensor.numpy().ravel()

    X_tab = _tabular_summary(X_seq_batch)
    ensemble = bundle["ensemble"]
    hgbr_ret = ensemble["hgbr"].predict(X_tab)
    ridge_ret = ensemble["ridge"].predict(X_tab)
    hgbc_dir = ensemble["hgbc"].predict_proba(X_tab)[:, 1]

    pred_rets, up_probs = _combine_predictions(
        nn_ret,
        hgbr_ret,
        ridge_ret,
        nn_dir,
        hgbc_dir,
        ensemble["cal_dir"],
        ensemble["ret_scale"],
        ensemble["blend_weights"],
    )
    return np.clip(pred_rets, -0.5, 0.5), np.clip(up_probs, 0.0, 1.0)


def predict_next(df, symbol, anchor_price=None, interval=INTERVAL):
    """Predict the next candle close range and direction confidence from completed candles."""
    interval = interval or INTERVAL
    if not model_exists(symbol, interval=interval):
        raise FileNotFoundError(
            f"A compatible {interval} model for {symbol} is not trained."
        )

    prepared = _prepare_data(df)
    sequence_length = int(SEQUENCE_LENGTH)

    if len(prepared) < sequence_length:
        raise ValueError(
            f"Not enough usable rows for prediction: {len(prepared)} "
            f"(need {sequence_length})."
        )

    bundle = _load_artifacts(symbol, interval=interval)
    latest = prepared[list(FEATURES)].tail(sequence_length).to_numpy(dtype=np.float32)
    if not np.isfinite(latest).all():
        raise ValueError("Latest prediction features contain NaN or infinite values.")

    scaled = np.clip(bundle["sx"].transform(latest), -8.0, 8.0).astype(np.float32)
    X_seq = np.expand_dims(scaled, axis=0)
    pred_ret_arr, up_prob_arr = _infer_batch(prepared, X_seq, bundle)

    predicted_log_return = float(pred_ret_arr[0])
    up_prob = float(up_prob_arr[0])

    if anchor_price is not None and np.isfinite(anchor_price) and float(anchor_price) > 0:
        current_price = float(anchor_price)
    else:
        current_price = float(prepared["close"].iloc[-1])

    predicted_price = float(current_price * np.exp(predicted_log_return))
    predicted_low, predicted_high, band_pct = _compute_price_range(
        predicted_price,
        prepared["vol_20"].iloc[-1],
        prepared["atr_14_norm"].iloc[-1],
        vol_10=prepared["vol_10"].iloc[-1],
    )
    change_percent = float((np.exp(predicted_log_return) - 1.0) * 100.0)
    confidence_pct = float(max(up_prob, 1.0 - up_prob) * 100.0)
    direction = "UP" if predicted_price >= current_price else "DOWN"

    if confidence_pct >= 54.0:
        signal_strength = "HIGH"
    elif confidence_pct >= DIRECTION_CONFIDENCE_THRESHOLD:
        signal_strength = "MEDIUM"
    else:
        signal_strength = "LOW"

    return {
        "current_price": current_price,
        "predicted_price": predicted_price,
        "predicted_low": predicted_low,
        "predicted_high": predicted_high,
        "band_pct": round(band_pct, 4),
        "predicted_log_return": predicted_log_return,
        "change_percent": change_percent,
        "direction": direction,
        "up_probability": round(up_prob * 100.0, 2),
        "confidence_pct": round(confidence_pct, 2),
        "signal_strength": signal_strength,
        "interval": interval,
        "interval_ms": int(INTERVAL_MS.get(interval, 60_000)),
        "timestamp": str(prepared["close_time"].iloc[-1])
        if "close_time" in prepared.columns else None,
        "forecast_type": "next_candle_range",
    }


def verify_recent(df, symbol, interval=INTERVAL, count=10):
    """Evaluate the saved model over the most recent completed candle transitions."""
    interval = interval or INTERVAL
    if not model_exists(symbol, interval=interval):
        raise FileNotFoundError(
            f"A compatible {interval} model for {symbol} is not trained."
        )

    prepared = _prepare_data(df)
    sequence_length = int(SEQUENCE_LENGTH)
    if len(prepared) < sequence_length + 2:
        raise ValueError("Not enough completed candles for verification log.")

    bundle = _load_artifacts(symbol, interval=interval)
    X_raw = prepared[list(FEATURES)].to_numpy(dtype=np.float32)
    X_scaled = np.clip(bundle["sx"].transform(X_raw), -8.0, 8.0).astype(np.float32)
    prices = prepared["close"].to_numpy(dtype=float)
    vols_20 = prepared["vol_20"].to_numpy(dtype=float)
    vols_10 = prepared["vol_10"].to_numpy(dtype=float)
    atrs = prepared["atr_14_norm"].to_numpy(dtype=float)

    max_end = len(prepared) - 1  # predicts index i+1 up to len-1
    start_idx = max(sequence_length - 1, max_end - int(count))
    indices = list(range(start_idx, max_end))

    windows = np.asarray(
        [X_scaled[i - sequence_length + 1:i + 1] for i in indices],
        dtype=np.float32,
    )
    pred_rets, _ = _infer_batch(prepared, windows, bundle)

    records = []
    in_range_hits = 0
    dir_hits = 0
    has_times = "close_time" in prepared.columns

    for k, i in enumerate(indices):
        prev_close = float(prices[i])
        actual_close = float(prices[i + 1])
        pred_mid = float(prev_close * np.exp(float(pred_rets[k])))
        pred_low, pred_high, _ = _compute_price_range(
            pred_mid, vols_20[i], atrs[i], vol_10=vols_10[i]
        )

        in_range = bool(pred_low <= actual_close <= pred_high)
        pred_dir = "UP" if pred_mid >= prev_close else "DOWN"
        actual_up = actual_close >= prev_close
        dir_hit = bool((pred_dir == "UP") == actual_up)

        in_range_hits += int(in_range)
        dir_hits += int(dir_hit)
        error_abs = float(abs(actual_close - pred_mid))
        error_pct = float(error_abs / max(actual_close, 1e-9) * 100.0)
        ts_val = str(prepared["close_time"].iloc[i + 1]) if has_times else None

        records.append({
            "timestamp": ts_val,
            "previous_close": prev_close,
            "predicted_low": pred_low,
            "predicted_mid": pred_mid,
            "predicted_high": pred_high,
            "actual_close": actual_close,
            "error_abs": error_abs,
            "error_pct": round(error_pct, 4),
            "in_range": in_range,
            "direction": pred_dir,
            "actual_direction": "UP" if actual_up else "DOWN",
            "direction_hit": dir_hit,
        })

    records.reverse()
    total = len(records)
    return {
        "symbol": symbol,
        "interval": interval,
        "records": records,
        "summary": {
            "total": total,
            "in_range_count": in_range_hits,
            "range_hit_rate": round(in_range_hits / max(total, 1) * 100.0, 1),
            "direction_hit_count": dir_hits,
            "direction_total_calls": total,
            "direction_hit_rate": round(dir_hits / max(total, 1) * 100.0, 1),
        },
    }
