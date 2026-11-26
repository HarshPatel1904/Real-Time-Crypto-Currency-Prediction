
import time
import requests
import pandas as pd

# =========================================================
# BINANCE API ENDPOINTS
# =========================================================

BASE_URLS = [
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
    "https://api-gcp.binance.com/api/v3/klines",
]

MAX_REQUEST_LIMIT = 1000
REQUEST_DELAY = 0.12

session = requests.Session()
_LAST_SERVER_UTC = None

session.headers.update({
    "User-Agent": "CryptoAI-Predictor/1.0"
})


# =========================================================
# INTERVAL MILLISECONDS
# =========================================================

INTERVAL_MS = {
    "1s": 1000,
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000
}


# =========================================================
# CONVERT DATETIME TO MILLISECONDS
# =========================================================

def to_milliseconds(value):

    if value is None:
        return None

    if isinstance(value, (int, float)):
        return int(value)

    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return int(timestamp.timestamp() * 1000)


# =========================================================
# BINANCE REQUEST WITH FALLBACK
# =========================================================

def request_klines(params):

    errors = []

    for url in BASE_URLS:

        try:

            response = session.get(
                url,
                params=params,
                timeout=(5, 20)
            )

            # Do not continue sending requests after rate limiting.
            if response.status_code in (418, 429):

                retry_after = response.headers.get(
                    "Retry-After",
                    "unknown"
                )

                raise RuntimeError(
                    f"Binance rate limit reached. "
                    f"Retry after: {retry_after} seconds."
                )

            # Invalid symbols or parameters need correction,
            # not another endpoint.
            if response.status_code == 400:
                raise RuntimeError(
                    f"Binance rejected the request: {response.text}"
                )

            response.raise_for_status()

            data = response.json()

            if not isinstance(data, list):
                raise RuntimeError(
                    f"Unexpected Binance response: {data}"
                )

            date_header = response.headers.get("Date")
            if date_header:
                try:
                    parsed_dt = pd.Timestamp(date_header)
                    if parsed_dt.tzinfo is None:
                        parsed_dt = parsed_dt.tz_localize("UTC")
                    else:
                        parsed_dt = parsed_dt.tz_convert("UTC")
                    global _LAST_SERVER_UTC
                    _LAST_SERVER_UTC = parsed_dt
                except Exception:
                    pass

            return data

        except requests.exceptions.HTTPError as error:

            status = (
                error.response.status_code
                if error.response is not None
                else None
            )

            if status in (418, 429, 400):
                raise RuntimeError(str(error)) from error

            errors.append(f"{url}: {error}")

        except requests.exceptions.RequestException as error:

            errors.append(f"{url}: {error}")

        except ValueError as error:

            errors.append(f"{url}: Invalid JSON: {error}")

    raise RuntimeError(
        "All Binance endpoints failed:\n"
        + "\n".join(errors)
    )


# =========================================================
# GET HISTORICAL CRYPTOCURRENCY DATA
# =========================================================

def get_klines(
    symbol="BTCUSDT",
    interval="1m",
    limit=1000,
    start_time=None,
    end_time=None,
    include_live=False
):

    limit = int(limit)

    if limit <= 0:
        raise ValueError("Limit must be greater than zero.")

    if interval not in INTERVAL_MS:
        raise ValueError(
            f"Unsupported interval: {interval}"
        )

    interval_ms = INTERVAL_MS[interval]

    # Convert optional timestamps
    start_ms = to_milliseconds(start_time)
    end_ms = to_milliseconds(end_time)

    candles = {}

    # =====================================================
    # CASE 1: NORMAL LATEST DATA REQUEST
    # =====================================================

    if (
        start_ms is None
        and end_ms is None
        and limit <= MAX_REQUEST_LIMIT
    ):

        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }

        data = request_klines(params)

        for candle in data:
            candles[int(candle[0])] = candle

    # =====================================================
    # CASE 2: HISTORICAL PAGINATION
    # =====================================================

    else:

        # Use the latest completed candle as the endpoint.
        if end_ms is None:

            current_ms = int(time.time() * 1000)

            end_exclusive = (
                current_ms // interval_ms
            ) * interval_ms

            end_ms = end_exclusive - 1

        # Calculate historical start if not supplied.
        if start_ms is None:

            start_ms = (
                end_ms + 1 - limit * interval_ms
            )

        if start_ms > end_ms:
            raise ValueError(
                "start_time must be before end_time."
            )

        cursor = start_ms

        print(
            f"\nDownloading {symbol} historical data..."
        )

        while (
            len(candles) < limit
            and cursor <= end_ms
        ):

            remaining = limit - len(candles)

            batch_limit = min(
                MAX_REQUEST_LIMIT,
                remaining
            )

            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": batch_limit
            }

            data = request_klines(params)

            if not data:
                break

            page_open_times = []

            for candle in data:

                open_time = int(candle[0])

                if (
                    cursor <= open_time
                    and open_time <= end_ms
                ):

                    candles[open_time] = candle

                    page_open_times.append(
                        open_time
                    )

            if not page_open_times:
                break

            last_open_time = max(
                page_open_times
            )

            next_cursor = (
                last_open_time + interval_ms
            )

            if next_cursor <= cursor:
                raise RuntimeError(
                    "Historical pagination did not advance."
                )

            cursor = next_cursor

            print(
                f"{symbol}: "
                f"{len(candles):,}/{limit:,} candles"
            )

            if (
                len(candles) < limit
                and cursor <= end_ms
            ):
                time.sleep(REQUEST_DELAY)

    # =====================================================
    # CONVERT TO DATAFRAME
    # =====================================================

    rows = sorted(
        candles.values(),
        key=lambda item: int(item[0])
    )

    # Keep exactly the requested most recent candles.
    rows = rows[-limit:]

    if not rows:
        raise RuntimeError(
            f"No market data returned for {symbol}."
        )

    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore"
    ]

    df = pd.DataFrame(
        rows,
        columns=columns
    )

    # Convert numeric fields
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Convert timestamps to UTC
    df["open_time"] = pd.to_datetime(
        df["open_time"],
        unit="ms",
        utc=True
    )

    df["close_time"] = pd.to_datetime(
        df["close_time"],
        unit="ms",
        utc=True
    )

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    df = (
        df.drop_duplicates(subset=["open_time"])
          .sort_values("open_time")
          .reset_index(drop=True)
    )

    # Exclude the currently forming candle using exchange server time when
    # available so local PC clock skew never drops a newly closed candle.
    if not include_live:
        now_utc = _LAST_SERVER_UTC if _LAST_SERVER_UTC is not None else pd.Timestamp.now(tz="UTC")
        df = df[df["close_time"] <= now_utc].copy().reset_index(drop=True)

    print(
        f"\nCompleted {symbol}: "
        f"{len(df):,} candles downloaded."
    )

    return df


def get_klines_with_live_price(symbol="BTCUSDT", interval="1m", limit=300):
    """Fetch candles once and return (completed_df, latest_live_price)."""
    df_all = get_klines(symbol, interval, limit, include_live=True)
    live_price = float(df_all["close"].iloc[-1])
    now_utc = _LAST_SERVER_UTC if _LAST_SERVER_UTC is not None else pd.Timestamp.now(tz="UTC")
    completed_df = df_all[df_all["close_time"] <= now_utc].copy().reset_index(drop=True)
    if completed_df.empty:
        completed_df = df_all.iloc[:-1].copy().reset_index(drop=True)
    return completed_df, live_price
