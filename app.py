from flask import Flask, jsonify, render_template, request

from config import (
    CRYPTOCURRENCIES,
    INTERVAL,
    INTERVAL_DISPLAY,
    INTERVAL_LABELS,
    LIVE_DATA_LIMIT,
    PREDICTION_DATA_LIMIT,
    SUPPORTED_INTERVALS,
    TRAINING_DATA_LIMIT,
)

from data import get_klines

from model import (
    load_metadata,
    model_exists,
    predict_next,
    train_symbol,
    verify_recent,
)


app = Flask(__name__)


# =========================================================
# Validate cryptocurrency
# =========================================================

def validate(symbol):

    symbol = symbol.upper()

    if symbol not in CRYPTOCURRENCIES:
        raise ValueError(
            f"Unsupported cryptocurrency: {symbol}"
        )

    return symbol


def validate_interval(raw_interval=None):
    interval = (raw_interval or request.args.get("interval") or INTERVAL).lower()
    if interval not in SUPPORTED_INTERVALS:
        raise ValueError(
            f"Unsupported interval: {interval}. Supported: {', '.join(SUPPORTED_INTERVALS)}"
        )
    return interval


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        cryptocurrencies=CRYPTOCURRENCIES,
        interval=INTERVAL,
        interval_display=INTERVAL_DISPLAY,
        supported_intervals=SUPPORTED_INTERVALS,
        interval_labels=INTERVAL_LABELS,
    )


# =========================================================
# GET ALL CRYPTOCURRENCIES
# =========================================================

@app.route("/api/coins")
def coins():
    interval = validate_interval()
    result = []

    for pair, info in CRYPTOCURRENCIES.items():

        result.append({

            # IMPORTANT:
            # Full Binance trading pair
            "pair": pair,

            # Display symbol
            "symbol": info["symbol"],

            "name": info["name"],

            "icon": info["icon"],

            "class": info["class"],

            "interval": interval,

            # Check whether model exists for selected interval
            "trained": model_exists(pair, interval=interval)

        })

    return jsonify(result)


# =========================================================
# MARKET DATA
# =========================================================

@app.route("/api/market/<symbol>")
def market(symbol):

    try:

        # Validate symbol & interval
        symbol = validate(symbol)
        interval = validate_interval()

        # Get completed Binance market candles (static snapshot)
        df = get_klines(
            symbol,
            interval,
            LIVE_DATA_LIMIT
        )

        candles = []

        for _, row in df.iterrows():

            candles.append({

                "time":
                    int(
                        row["open_time"].timestamp()
                    ),

                "open":
                    float(row["open"]),

                "high":
                    float(row["high"]),

                "low":
                    float(row["low"]),

                "close":
                    float(row["close"]),

                "volume":
                    float(row["volume"])

            })

        return jsonify({

            "symbol": symbol,

            "interval": interval,

            "name":
                CRYPTOCURRENCIES[
                    symbol
                ]["name"],

            "candles":
                candles

        })

    except Exception as e:

        print(
            f"Market error for {symbol}:",
            e
        )

        return jsonify({

            "error":
                str(e)

        }), 500


# =========================================================
# PREDICTION
# =========================================================

@app.route("/api/predict/<symbol>")
def prediction(symbol):

    try:

        # Validate cryptocurrency & interval
        symbol = validate(symbol)
        interval = validate_interval()

        # Check whether model exists
        if not model_exists(symbol, interval=interval):

            return jsonify({

                "error":
                    f"Model for {symbol} ({interval}) is not trained yet.",

                "code":
                    "MODEL_NOT_FOUND"

            }), 409

        # Download latest completed market data
        df = get_klines(
            symbol,
            interval,
            PREDICTION_DATA_LIMIT
        )

        # Generate range prediction
        result = predict_next(
            df,
            symbol,
            interval=interval,
        )

        return jsonify({

            "symbol":
                symbol,

            "name":
                CRYPTOCURRENCIES[
                    symbol
                ]["name"],

            **result

        })

    except Exception as e:

        print(
            f"Prediction error for {symbol}:",
            e
        )

        return jsonify({

            "error":
                str(e)

        }), 500


# =========================================================
# PREDICTION VERIFICATION LOG
# =========================================================

@app.route("/api/verification/<symbol>")
def verification(symbol):

    try:

        symbol = validate(symbol)
        interval = validate_interval()

        if not model_exists(symbol, interval=interval):
            return jsonify({
                "error": f"Model for {symbol} ({interval}) is not trained yet.",
                "code": "MODEL_NOT_FOUND",
            }), 409

        df = get_klines(symbol, interval, PREDICTION_DATA_LIMIT)
        result = verify_recent(df, symbol, interval=interval, count=10)
        return jsonify(result)

    except Exception as e:

        print(f"Verification error for {symbol}:", e)
        return jsonify({"error": str(e)}), 500


# =========================================================
# MODEL INFORMATION
# =========================================================

@app.route("/api/model/<symbol>")
def model_info(symbol):

    try:

        # Validate
        symbol = validate(symbol)
        interval = validate_interval()

        # Load model metadata
        metadata = load_metadata(symbol, interval=interval)

        # Model does not exist
        if metadata is None:

            return jsonify({

                "trained": False,

                "symbol":
                    symbol,

                "name":
                    CRYPTOCURRENCIES[
                        symbol
                    ]["name"]

            })

        # Model exists
        return jsonify({

            "trained": True,

            "symbol":
                symbol,

            "name":
                CRYPTOCURRENCIES[
                    symbol
                ]["name"],

            **metadata

        })

    except Exception as e:

        print(
            f"Model info error for {symbol}:",
            e
        )

        return jsonify({

            "error":
                str(e)

        }), 500


# =========================================================
# TRAIN MODEL
# =========================================================

@app.route(
    "/api/train/<symbol>",
    methods=["POST"]
)
def train(symbol):

    try:

        # Validate
        symbol = validate(symbol)
        interval = validate_interval()

        print(
            f"\nStarting training for {symbol} ({interval})..."
        )

        # Download latest training data
        df = get_klines(
            symbol,
            interval,
            TRAINING_DATA_LIMIT
        )

        # Train model
        result = train_symbol(
            df,
            symbol,
            interval=interval,
        )

        print(
            f"Training completed for {symbol}"
        )

        return jsonify({

            "success":
                True,

            "symbol":
                symbol,

            "name":
                CRYPTOCURRENCIES[
                    symbol
                ]["name"],

            **result

        })

    except Exception as e:

        print(
            f"Training error for {symbol}:",
            e
        )

        return jsonify({

            "success":
                False,

            "error":
                str(e)

        }), 500


# =========================================================
# START FLASK SERVER
# =========================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5000,

        debug=True

    )