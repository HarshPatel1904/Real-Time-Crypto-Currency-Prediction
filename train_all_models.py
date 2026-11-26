import gc
import os
import time
import traceback
from pathlib import Path

from config import CRYPTOCURRENCIES, INTERVAL, TRAINING_DATA_LIMIT
from data import get_klines, INTERVAL_MS
from model import train_symbol


PROJECT_DIR = Path(__file__).resolve().parent
os.chdir(PROJECT_DIR)


def main():
    results = {}
    symbols = list(CRYPTOCURRENCIES.keys())

    interval_ms = INTERVAL_MS.get(INTERVAL)
    training_days = (
        TRAINING_DATA_LIMIT * interval_ms / 86_400_000
        if interval_ms else None
    )

    print("=" * 72)
    print("MULTI-CRYPTO HYBRID NEXT-CLOSE PRICE MODEL TRAINING")
    print("=" * 72)
    print("Interval:", INTERVAL)
    print("Training candles:", f"{TRAINING_DATA_LIMIT:,}")
    if training_days is not None:
        print("Approximate training period:", f"{training_days:.2f} days")
    print("Total cryptocurrencies:", len(symbols))
    print("Architecture: Conv1D-BiLSTM-Attention + Ridge/HGBC Ensemble")

    for index, symbol in enumerate(symbols, start=1):
        print("\n" + "=" * 72)
        print(f"TRAINING {index}/{len(symbols)}: {symbol}")
        print("=" * 72)
        df = None

        try:
            print("Downloading historical candles...")
            df = get_klines(symbol, INTERVAL, TRAINING_DATA_LIMIT)
            print("Downloaded completed candles:", len(df))
            print("Starting hybrid ensemble training...")

            metrics = train_symbol(df, symbol)
            results[symbol] = {"status": "SUCCESS", **metrics}

            print("\nTraining completed:", symbol)
            print(f"MAE: {metrics['mae']:.6f} (skill {metrics['mae_skill_vs_baseline']*100:+.3f}%)")
            print(f"RMSE: {metrics['rmse']:.6f} (skill {metrics['rmse_skill_vs_baseline']*100:+.3f}%)")
            print(f"MAPE: {metrics['mape']:.4f}%")
            print(f"R2: {metrics['r2']:.6f}")
            print(f"Direction accuracy (all): {metrics['direction_accuracy']:.2f}%")
            print(
                f"Direction accuracy (high conf >= {metrics['high_confidence_threshold']:.0f}%): "
                f"{metrics['high_confidence_direction_accuracy']:.2f}% "
                f"(n={metrics['high_confidence_samples']})"
            )
            print(f"Price accuracy within 0.5%: {metrics['accuracy_within_0_5_percent']:.2f}%")
            print(f"Price accuracy within 1.0%: {metrics['accuracy_within_1_percent']:.2f}%")

        except Exception as error:
            results[symbol] = {"status": "FAILED", "error": str(error)}
            print("\nTraining failed:", symbol)
            print(error)
            traceback.print_exc()

        finally:
            if df is not None:
                del df
            gc.collect()
            try:
                from tensorflow.keras import backend as K
                K.clear_session()
            except Exception:
                pass

        time.sleep(1)

    print("\n" + "=" * 72)
    print("FINAL HYBRID MODEL TRAINING REPORT")
    print("=" * 72)

    successful = 0
    for symbol, result in results.items():
        print("\n", symbol)
        print("Status:", result["status"])
        if result["status"] != "SUCCESS":
            print("Error:", result.get("error", "Unknown error"))
            continue

        successful += 1
        print(f"MAE: {result['mae']:.6f} (baseline {result['baseline_mae']:.6f}, skill {result['mae_skill_vs_baseline']*100:+.3f}%)")
        print(f"RMSE: {result['rmse']:.6f} (baseline {result['baseline_rmse']:.6f}, skill {result['rmse_skill_vs_baseline']*100:+.3f}%)")
        print(f"MAPE: {result['mape']:.4f}% | R2: {result['r2']:.6f}")
        print(f"Direction Accuracy: {result['direction_accuracy']:.2f}% | High-Conf: {result['high_confidence_direction_accuracy']:.2f}% (n={result['high_confidence_samples']})")
        print(f"Within 0.5%: {result['accuracy_within_0_5_percent']:.2f}% | Within 1%: {result['accuracy_within_1_percent']:.2f}%")
        print(f"Samples: train={result['train_samples']}, val={result['validation_samples']}, test={result['test_samples']}")

    print("\nSuccessfully trained:")
    print(f"{successful}/{len(symbols)}")
    print("All training tasks completed.")


if __name__ == "__main__":
    main()
