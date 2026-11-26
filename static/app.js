// ============================================================
// GLOBAL VARIABLES
// ============================================================

const INTERVAL_DISPLAY_MAP = {
    "1m": "1 minute",
    "5m": "5 minute",
    "15m": "15 minute"
};

let selectedSymbol = "BTCUSDT";
let selectedInterval = document.body?.dataset?.interval || "1m";
let chart = null;
let latestCandles = [];
let latestPrediction = null;


// ============================================================
// FORMAT HELPERS
// ============================================================

function formatPrice(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) {
        return "--";
    }

    let decimals = 2;
    if (Math.abs(number) < 1) {
        decimals = 6;
    } else if (Math.abs(number) < 10) {
        decimals = 4;
    }

    return "$" + number.toLocaleString("en-US", {
        minimumFractionDigits: decimals,
        maximumFractionDigits: decimals
    });
}

function formatTime(isoString) {
    if (!isoString) {
        return "--";
    }
    const ms = Date.parse(String(isoString).replace(" ", "T"));
    if (!Number.isFinite(ms)) {
        return "--";
    }
    return new Date(ms).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit"
    });
}

function setText(id, text) {
    const el = document.getElementById(id);
    if (el) {
        el.innerText = text;
    }
}

function intervalDisplay() {
    return INTERVAL_DISPLAY_MAP[selectedInterval] || selectedInterval;
}


// ============================================================
// CONNECTION & MODEL STATUS
// ============================================================

function setConnection(connected, text) {
    const dot = document.getElementById("connectionDot");
    const status = document.getElementById("connectionText");
    if (dot) {
        dot.style.background = connected ? "#22c55e" : "#ef4444";
    }
    if (status) {
        status.innerText = text;
    }
}

function setModelStatus(statusText, statusColor) {
    const status = document.getElementById("modelStatus");
    const dot = document.getElementById("modelDot");
    if (status) {
        status.innerText = statusText;
    }
    if (dot) {
        dot.style.background = statusColor;
    }
}


// ============================================================
// SELECT CRYPTOCURRENCY & TIMEFRAME
// ============================================================

function selectCoin(symbol) {
    selectedSymbol = symbol;
    document.querySelectorAll(".coin-card").forEach(card => {
        card.classList.toggle("active", card.dataset.symbol === symbol);
    });
    resetDashboard();
    loadSelectedCoin();
}

function selectTimeframe(interval) {
    if (!INTERVAL_DISPLAY_MAP[interval] || interval === selectedInterval) {
        return;
    }
    selectedInterval = interval;
    document.querySelectorAll(".tf-btn").forEach(btn => {
        btn.classList.toggle("active", btn.dataset.interval === interval);
    });
    setText("metaInterval", intervalDisplay());
    resetDashboard();
    loadSelectedCoin();
}


// ============================================================
// RESET DASHBOARD
// ============================================================

function resetDashboard() {
    latestPrediction = null;

    setText("selectedName", "Loading...");
    setText("selectedPair", "--");
    setText("currentPrice", "--");
    setText("predictedPriceRange", "Loading...");
    setText("predictionChange", "--");
    setText("direction", "--");
    setText("currentPriceFoot", `Latest closed ${intervalDisplay()} candle`);
    setText("predictedPriceFoot", `Expected next ${intervalDisplay()} range`);
    setText("directionConfidence", "Predicted upward / downward bias");
    setText("chartSubtitle", "Loading market data...");
    setText("verificationSummary", "Loading recent candles...");

    const tbody = document.getElementById("verificationTableBody");
    if (tbody) {
        tbody.innerHTML = '<tr><td colspan="7" class="empty-cell">Loading verification log...</td></tr>';
    }

    const dirEl = document.getElementById("direction");
    if (dirEl) {
        dirEl.style.color = "#94a3b8";
    }

    setModelStatus("Checking model...", "#64748b");
    setConnection(false, "Loading snapshot...");
}


// ============================================================
// LOAD SELECTED CRYPTO & TIMEFRAME (STATIC SNAPSHOT)
// ============================================================

async function loadSelectedCoin() {
    try {
        const results = await Promise.allSettled([
            loadCoinInfo(),
            loadMarket(),
            loadPrediction(),
            loadVerification()
        ]);

        if (results[1].status === "rejected") {
            setConnection(false, "Connection failed");
        }
    } catch (error) {
        console.error("Dashboard loading error:", error);
        setConnection(false, "Connection error");
        setModelStatus("Error loading model", "#ef4444");
    }
}


// ============================================================
// LOAD CRYPTO INFORMATION
// ============================================================

async function loadCoinInfo() {
    try {
        setModelStatus("Checking model...", "#64748b");
        const response = await fetch(`/api/coins?interval=${encodeURIComponent(selectedInterval)}`);
        if (!response.ok) {
            throw new Error("Failed to load cryptocurrency list.");
        }

        const coins = await response.json();
        const coin = coins.find(item => item.pair === selectedSymbol);
        if (!coin) {
            throw new Error("Cryptocurrency " + selectedSymbol + " was not found.");
        }

        setText("selectedName", coin.name);
        setText("selectedPair", coin.symbol + "/USDT");
        setText("metaInterval", intervalDisplay());
        setText("chartSubtitle", `${coin.symbol}/USDT · ${intervalDisplay()} candles`);

        if (coin.trained) {
            setModelStatus(`Model ready (${selectedInterval})`, "#22c55e");
        } else {
            setModelStatus(`Model not trained (${selectedInterval})`, "#f59e0b");
        }
    } catch (error) {
        console.error("Coin information error:", error);
        setModelStatus("Model status unavailable", "#ef4444");
        throw error;
    }
}


// ============================================================
// LOAD MARKET DATA
// ============================================================

async function loadMarket() {
    try {
        const response = await fetch(`/api/market/${selectedSymbol}?interval=${encodeURIComponent(selectedInterval)}`);
        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || "Failed to load market data.");
        }

        latestCandles = data.candles || [];
        if (latestCandles.length === 0) {
            throw new Error("No market candles received.");
        }

        const latest = latestCandles[latestCandles.length - 1];
        setText("currentPrice", formatPrice(latest.close));

        updateChart();
        setConnection(true, `Static snapshot · ${selectedInterval}`);
    } catch (error) {
        console.error("Market data error:", error);
        setConnection(false, "Market connection failed");
        throw error;
    }
}


// ============================================================
// LOAD RANGE PREDICTION
// ============================================================

async function loadPrediction() {
    try {
        const response = await fetch(`/api/predict/${selectedSymbol}?interval=${encodeURIComponent(selectedInterval)}`);
        const data = await response.json();

        if (response.status === 409 && data.code === "MODEL_NOT_FOUND") {
            setPredictionUnavailable(`Model not trained (${selectedInterval})`);
            return;
        }

        if (!response.ok || data.error) {
            throw new Error(data.error || "Prediction failed.");
        }

        latestPrediction = data;

        setText("currentPrice", formatPrice(data.current_price));
        setText(
            "predictedPriceRange",
            `${formatPrice(data.predicted_low)} – ${formatPrice(data.predicted_high)}`
        );

        const midStr = formatPrice(data.predicted_price);
        const bandStr = Number(data.band_pct).toFixed(3);
        const closedLabel = formatTime(data.timestamp);
        const intervalMs = Number(data.interval_ms) || 60000;
        const closedMs = Date.parse(String(data.timestamp || "").replace(" ", "T"));
        const nextLabel = Number.isFinite(closedMs)
            ? formatTime(new Date(closedMs + intervalMs).toISOString())
            : "--";

        setText("currentPriceFoot", `Closed at ${closedLabel}`);
        setText("predictedPriceFoot", `Mid ${midStr} · ±${bandStr}% (${nextLabel})`);

        const changePct = Number(data.change_percent);
        setText(
            "predictionChange",
            (changePct >= 0 ? "+" : "") + changePct.toFixed(3) + "%"
        );

        const direction = document.getElementById("direction");
        if (direction) {
            direction.innerText = data.direction;
            direction.style.color = data.direction === "UP" ? "#22c55e" : "#ef4444";
        }

        if (Number.isFinite(Number(data.confidence_pct))) {
            const conf = Number(data.confidence_pct).toFixed(1);
            const strength = data.signal_strength || "LOW";
            setText("directionConfidence", `${conf}% confidence · ${strength} signal`);
        }

        setModelStatus(`Model ready (${selectedInterval})`, "#22c55e");
        updateChart();
    } catch (error) {
        console.error("Prediction error:", error);
        setText("predictedPriceRange", "Prediction unavailable");
        setText("predictionChange", "--");
        const direction = document.getElementById("direction");
        if (direction) {
            direction.innerText = "ERROR";
            direction.style.color = "#ef4444";
        }
        setModelStatus("Prediction error", "#ef4444");
    }
}

function setPredictionUnavailable(message) {
    latestPrediction = null;
    setText("predictedPriceRange", message);
    setText("predictionChange", "--");
    setText("predictedPriceFoot", `No trained ${selectedInterval} model`);
    setText("directionConfidence", "Train model for this timeframe");
    const direction = document.getElementById("direction");
    if (direction) {
        direction.innerText = "--";
        direction.style.color = "#94a3b8";
    }
    setModelStatus(`Model not trained (${selectedInterval})`, "#f59e0b");
}


// ============================================================
// LOAD PREDICTION VERIFICATION LOG
// ============================================================

async function loadVerification() {
    const tbody = document.getElementById("verificationTableBody");
    try {
        const response = await fetch(`/api/verification/${selectedSymbol}?interval=${encodeURIComponent(selectedInterval)}`);
        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || "Verification log unavailable.");
        }

        const records = data.records || [];
        const summary = data.summary || {};

        setText(
            "verificationSummary",
            `${summary.in_range_count || 0}/${summary.total || 0} In Range (${summary.range_hit_rate || 0}%) · ${summary.direction_hit_count || 0}/${summary.total || 0} Direction (${summary.direction_hit_rate || 0}%)`
        );

        if (!tbody) {
            return;
        }
        if (records.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" class="empty-cell">No verification records available.</td></tr>';
            return;
        }

        tbody.innerHTML = records.map(row => {
            const dirClass = row.direction === "UP" ? "dir-up" : "dir-down";
            const dirDetail = row.direction_hit ? "✓" : "✗";
            const rangeBadge = row.in_range
                ? '<span class="badge-hit">IN RANGE ✓</span>'
                : '<span class="badge-miss">OUTSIDE ✗</span>';

            return `
                <tr>
                    <td>${formatTime(row.timestamp)}</td>
                    <td>${formatPrice(row.previous_close)}</td>
                    <td><strong>${formatPrice(row.predicted_low)} – ${formatPrice(row.predicted_high)}</strong></td>
                    <td>${formatPrice(row.actual_close)}</td>
                    <td>${formatPrice(row.error_abs)} (${Number(row.error_pct).toFixed(3)}%)</td>
                    <td><span class="${dirClass}">${row.direction}</span> (${dirDetail})</td>
                    <td>${rangeBadge}</td>
                </tr>
            `;
        }).join("");
    } catch (error) {
        console.error("Verification log error:", error);
        setText("verificationSummary", "Verification unavailable");
        if (tbody) {
            tbody.innerHTML = '<tr><td colspan="7" class="empty-cell">Unable to load verification history.</td></tr>';
        }
    }
}


// ============================================================
// LIVE PRICE & FORECAST BAND CHART
// ============================================================

function updateChart() {
    const canvas = document.getElementById("priceChart");
    if (!canvas || !latestCandles || latestCandles.length === 0) {
        return;
    }

    const labels = latestCandles.map(candle =>
        new Date(candle.time * 1000).toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit"
        })
    );
    const prices = latestCandles.map(candle => candle.close);

    const datasets = [
        {
            label: `${selectedSymbol} Price`,
            data: prices,
            borderColor: "#4169e1",
            backgroundColor: "rgba(65, 105, 225, 0.08)",
            fill: true,
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.18
        }
    ];

    if (
        latestPrediction &&
        Number.isFinite(Number(latestPrediction.predicted_low)) &&
        Number.isFinite(Number(latestPrediction.predicted_high))
    ) {
        labels.push("Next Range");
        const lastPrice = prices[prices.length - 1];
        const pad = new Array(prices.length - 1).fill(null);

        const highSeries = [...pad, lastPrice, Number(latestPrediction.predicted_high)];
        const midSeries = [...pad, lastPrice, Number(latestPrediction.predicted_price)];
        const lowSeries = [...pad, lastPrice, Number(latestPrediction.predicted_low)];

        datasets.push(
            {
                label: "Predicted High",
                data: highSeries,
                borderColor: "rgba(21, 155, 135, 0.65)",
                borderDash: [4, 4],
                borderWidth: 1.5,
                pointRadius: ctx => (ctx.dataIndex === highSeries.length - 1 ? 3 : 0),
                tension: 0
            },
            {
                label: "Predicted Midpoint",
                data: midSeries,
                borderColor: "#159b87",
                backgroundColor: "#159b87",
                borderDash: [5, 4],
                borderWidth: 2,
                pointRadius: ctx => (ctx.dataIndex === midSeries.length - 1 ? 4 : 0),
                tension: 0
            },
            {
                label: "Predicted Low",
                data: lowSeries,
                borderColor: "rgba(21, 155, 135, 0.65)",
                backgroundColor: "rgba(21, 155, 135, 0.14)",
                fill: "-2",
                borderDash: [4, 4],
                borderWidth: 1.5,
                pointRadius: ctx => (ctx.dataIndex === lowSeries.length - 1 ? 3 : 0),
                tension: 0
            }
        );
    }

    const footer = document.getElementById("chartFooter");
    if (footer) {
        footer.innerHTML = `<span><i class="legend-swatch"></i> MARKET PRICE &amp; PREDICTED BAND</span><span>BINANCE · ${selectedInterval.toUpperCase()}</span><span>${latestCandles.length} CANDLES</span>`;
    }

    const context = canvas.getContext("2d");
    if (chart) {
        chart.destroy();
    }

    chart = new Chart(context, {
        type: "line",
        data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            interaction: {
                intersect: false,
                mode: "index"
            },
            scales: {
                x: {
                    ticks: {
                        color: "#7d899d",
                        maxTicksLimit: 10
                    },
                    grid: {
                        color: "rgba(148,163,184,0.12)"
                    }
                },
                y: {
                    ticks: {
                        color: "#7d899d",
                        callback: value => formatPrice(value)
                    },
                    grid: {
                        color: "rgba(148,163,184,0.12)"
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: "#59667b",
                        boxWidth: 14
                    }
                }
            }
        }
    });
}


// ============================================================
// CRYPTO FORECASTING QUOTES
// ============================================================

const CRYPTO_QUOTES = [
    "In crypto markets, price is what you pay; probability and risk are what you manage.",
    "Forecasts don't predict the future with certainty—they narrow the range of surprises.",
    "Volatility is the price of admission in digital assets; discipline is how you navigate it.",
    "Markets speak in noise at the second scale and in structure at the trend scale.",
    "Data reveals momentum; risk management turns probability into edge.",
    "Every candle encodes a battle between liquidity, sentiment, and time."
];

let quoteIndex = 0;

function startQuoteRotation() {
    const quoteEl = document.getElementById("cryptoQuote");
    if (!quoteEl) {
        return;
    }
    setInterval(() => {
        quoteIndex = (quoteIndex + 1) % CRYPTO_QUOTES.length;
        quoteEl.style.opacity = "0";
        setTimeout(() => {
            quoteEl.innerText = CRYPTO_QUOTES[quoteIndex];
            quoteEl.style.opacity = "1";
        }, 250);
    }, 8000);
}


// ============================================================
// INIT
// ============================================================

document.addEventListener("DOMContentLoaded", function () {
    const bitcoinCard = document.querySelector('.coin-card[data-symbol="BTCUSDT"]');
    if (bitcoinCard) {
        bitcoinCard.classList.add("active");
    }
    startQuoteRotation();
    loadSelectedCoin();
});
