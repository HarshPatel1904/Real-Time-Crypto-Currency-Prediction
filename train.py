import sys

from config import CRYPTOCURRENCIES, INTERVAL, TRAINING_DATA_LIMIT
from data import get_klines
from model import train_symbol


def main():
    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else "BTCUSDT"
    if symbol not in CRYPTOCURRENCIES:
        raise SystemExit("Available: " + ", ".join(CRYPTOCURRENCIES))

    print(f"Downloading {symbol} ({INTERVAL}, {TRAINING_DATA_LIMIT:,} candles)...")
    df = get_klines(symbol, INTERVAL, TRAINING_DATA_LIMIT)
    print(f"Training {symbol} hybrid forecasting model...")
    r = train_symbol(df, symbol)
    print("Training completed.")
    print(f"MAE: {r['mae']:.6f} (baseline {r['baseline_mae']:.6f}, skill {r['mae_skill_vs_baseline']*100:+.3f}%)")
    print(f"RMSE: {r['rmse']:.6f} (baseline {r['baseline_rmse']:.6f}, skill {r['rmse_skill_vs_baseline']*100:+.3f}%)")
    print(f"MAPE: {r['mape']:.4f}%")
    print(f"R2: {r['r2']:.6f}")
    print(f"Direction accuracy (all): {r['direction_accuracy']:.2f}%")
    print(
        f"Direction accuracy (high conf >= {r['high_confidence_threshold']:.0f}%): "
        f"{r['high_confidence_direction_accuracy']:.2f}% (n={r['high_confidence_samples']})"
    )
    print(f"Accuracy within 1%: {r['accuracy_within_1_percent']:.2f}%")


if __name__ == "__main__":
    main()
