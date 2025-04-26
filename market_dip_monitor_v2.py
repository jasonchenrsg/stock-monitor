#!/usr/bin/env python3
"""
market_dip_monitor.py — Monitor VOO & QQQM for dip-buy signals
=============================================================

*Checks every 15 min (or on demand) whether a set of 'buy-the-dip' criteria
fire, logs each trigger, and optionally sends an email alert.*

Designed for GitHub Actions **and** easy local testing.

Quick CLI
---------
15-min intraday poll (uses 15-minute bars)
$ python market_dip_monitor.py intraday

Historical poll for a date range (YYYY-MM-DD YYYY-MM-DD)
$ python market_dip_monitor.py historical 2025-04-01 2025-04-24

End-of-day summary for today
$ python market_dip_monitor.py daily_summary

Historical summary for a given date (YYYY-MM-DD)
$ python market_dip_monitor.py daily_summary 2025-04-10




Dependencies: pandas, yfinance, pytz (all pure-Python).
"""
import os
import sys
import csv
import time
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import List, Tuple
import pytz
import pandas as pd
import yfinance as yf
import smtplib
from ssl import create_default_context
from email.message import EmailMessage

# ─────────────── configuration ────────────────
TZ          = pytz.timezone(os.getenv("PYTZ_TIMEZONE", "America/New_York"))
TICKERS     = ["VOO", "QQQM"]
VIX_TICKER  = "^VIX"

EVENTS = {
    "SMA20_GAP" : "Price ≤ SMA20 −0.75%",
    "BOLLINGER" : "Touched lower Bollinger band (20-period, 1σ)",
    "RSI"       : "RSI < 30 (20-period)",
    "VIX25"     : "VIX > 25",
    "DD10"      : "Price ≥10% below 52-wk high",
    "RISK_BRAKE": "VIX > 35 risk brake (skip buys)"
}
# thresholds
GAP_PCT   = 0.0075   # 0.75 %
VIX_WARN  = 25
VIX_BRAKE = 35
DRAW_PCT  = 0.10     # 10 % drawdown

# email (all optional)
SMTP_USER      = os.getenv("SMTP_USER")
SMTP_PASS      = os.getenv("SMTP_PASS")
SMTP_SERVER    = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT      = int(os.getenv("SMTP_PORT", 465))
ALERT_TO       = os.getenv("ALERT_TO")
ALERT_TRIGGERS = set(os.getenv("ALERT_TRIGGERS", "SMA20_GAP,BOLLINGER,RSI,VIX25,DD10").split(','))

LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)

# ─────────────── util helpers ────────────────

def now() -> datetime:
    return datetime.now(TZ)

# Fetch wrapper -------------------------------------------------

def fetch_prices(start: date, end: date, interval: str = "1d", retries: int = 3) -> pd.DataFrame:
    """Download price data from *start* to *end* inclusive with caching and retries."""
    # Validate interval and adjust start date for 15m (yfinance limit: 60 days)
    if interval == "15m" and (end - start).days > 60:
        start = end - timedelta(days=60)

    # Check cache
    cache_file = LOG_DIR / f"prices_{start}_{end}_{interval}.csv"
    if cache_file.exists():
        try:
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        except Exception as e:
            print(f"Error reading cache {cache_file}: {e}")

    # Fetch data with retries
    for attempt in range(retries):
        try:
            data = yf.download(TICKERS + [VIX_TICKER], start=start, end=end + timedelta(days=1), interval=interval,
                               auto_adjust=True, progress=False, prepost=True, group_by="ticker")
            if data.empty:
                raise ValueError(f"No data returned for {TICKERS + [VIX_TICKER]} from {start} to {end}")
            # Validate tickers
            missing_tickers = [t for t in TICKERS + [VIX_TICKER] if t not in data.columns.levels[0]]
            if missing_tickers:
                print(f"Warning: Missing data for tickers: {missing_tickers}")
            # Select price column: prefer Adj Close, fallback to Close
            price_col = "Adj Close" if "Adj Close" in data.columns.levels[1] else "Close"
            if price_col == "Close":
                print(f"Warning: Using 'Close' instead of 'Adj Close' for {interval} data")
            prices = data.xs(price_col, level=1, axis=1, drop_level=True)
            # Cache data
            try:
                prices.to_csv(cache_file)
            except Exception as e:
                print(f"Error caching data to {cache_file}: {e}")
            return prices
        except Exception as e:
            if attempt < retries - 1:
                print(f"Attempt {attempt + 1} failed: {e}, retrying in 10 seconds...")
                time.sleep(10)
            else:
                print(f"Failed after {retries} attempts: {e}")
                return pd.DataFrame()

# TA helpers ----------------------------------------------------

def ta_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up    = delta.clip(lower=0)
    down  = -delta.clip(upper=0)
    ma_up   = up.rolling(length).mean()
    ma_down = down.rolling(length).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

# Indicator builder --------------------------------------------

def compute_indicators(df_daily: pd.DataFrame) -> dict:
    if df_daily.empty:
        print("Error: Empty DataFrame provided to compute_indicators")
        return {}
    ind = {}
    for t in TICKERS:
        if t not in df_daily.columns:
            print(f"Warning: No data for {t}, skipping")
            continue
        series = df_daily[t]
        if series.isna().all():
            print(f"Warning: All data for {t} is NaN, skipping")
            continue
        sma20  = series.rolling(20).mean().iloc[-1]
        std20  = series.rolling(20).std().iloc[-1]
        rsi    = ta_rsi(series, 20).iloc[-1]
        high52 = series.rolling(252).max().iloc[-1]
        ind[t] = {
            "latest"    : series.iloc[-1],
            "sma20"     : sma20,
            "lower_band": sma20 - std20,
            "rsi"       : rsi,
            "drawdown"  : (high52 - series.iloc[-1]) / high52 if not pd.isna(high52) else 0,
        }
    if VIX_TICKER in df_daily.columns and not df_daily[VIX_TICKER].isna().all():
        ind["VIX"] = df_daily[VIX_TICKER].iloc[-1]
    else:
        print(f"Warning: No valid VIX data, setting VIX to 0")
        ind["VIX"] = 0
    return ind

# Event evaluation ---------------------------------------------

def evaluate_events(ind: dict) -> List[Tuple[str, str, float]]:
    out = []
    if not ind:
        return out
    vix_val = ind.get("VIX", 0)
    for t in TICKERS:
        if t not in ind:
            continue
        i = ind[t]
        price = i["latest"]
        if pd.isna(price):
            continue
        # Mean-reversion
        if not pd.isna(i["sma20"]) and price <= i["sma20"] * (1 - GAP_PCT):
            out.append(("SMA20_GAP", t, price))
        if not pd.isna(i["lower_band"]) and price <= i["lower_band"]:
            out.append(("BOLLINGER", t, price))
        if not pd.isna(i["rsi"]) and i["rsi"] < 30:
            out.append(("RSI", t, price))
        # Drawdown
        if not pd.isna(i["drawdown"]) and i["drawdown"] >= DRAW_PCT:
            out.append(("DD10", t, price))
    # Fear gauges
    if vix_val > VIX_WARN:
        for t in TICKERS:
            if t in ind:
                out.append(("VIX25", t, ind[t]["latest"]))
    if vix_val > VIX_BRAKE:
        for t in TICKERS:
            if t in ind:
                out.append(("RISK_BRAKE", t, ind[t]["latest"]))
    return out

# Logging -------------------------------------------------------

def log_events(events: List[Tuple[str, str, float]], as_of: date) -> None:
    if not events:
        return
    fname = LOG_DIR / f"{as_of}_events.csv"
    new_file = not fname.exists()
    with fname.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "event", "ticker", "price"])
        for ev, ticker, price in events:
            w.writerow([now().isoformat(), ev, ticker, f"{price:.2f}"])

def log_daily_event_summary(as_of: date) -> None:
    """Append daily event counts to a summary CSV file."""
    counts = {ev: 0 for ev in EVENTS}
    log_file = LOG_DIR / f"{as_of}_events.csv"
    if log_file.exists():
        try:
            with log_file.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row["event"] in counts:
                        counts[row["event"]] += 1
        except Exception as e:
            print(f"Error reading log {log_file}: {e}")

    summary_file = LOG_DIR / "event_summary.csv"
    new_file = not summary_file.exists()
    with summary_file.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["date"] + list(EVENTS.keys()))
        w.writerow([as_of.isoformat()] + [counts[ev] for ev in EVENTS])

# Email ---------------------------------------------------------

def send_email(events: List[Tuple[str, str, float]]) -> None:
    """Send email alerts for triggered events if configured.

    Args:
        events: List of (event, ticker, price) tuples.
    """
    if not (SMTP_USER and SMTP_PASS and ALERT_TO):
        print("Email not configured, skipping")
        return
    to_alert = [e for e in events if e[0] in ALERT_TRIGGERS]
    if not to_alert:
        return
    try:
        body = "\n".join([f"{ev} — {tk} @ {price:.2f}" for ev, tk, price in to_alert])
        msg = EmailMessage()
        msg["Subject"] = "Dip-monitor alert"
        msg["From"] = SMTP_USER
        msg["To"] = ALERT_TO
        msg.set_content(body)
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT, context=create_default_context()) as s:
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        print("Email sent successfully")
    except Exception as e:
        print(f"Error sending email: {e}")

# Intraday poll -------------------------------------------------

def intraday_poll() -> None:
    end_dt = now()
    start_dt = (end_dt - timedelta(days=20)).date()
    end_date = end_dt.date() if hasattr(end_dt, 'date') else end_dt
    df = fetch_prices(start_dt, end_date, interval="15m")
    if df.empty:
        print("No intraday data available")
        return
    df = df.resample("1D").last()
    if df.index[-1].date() != end_date:
        print(f"Warning: Latest data is from {df.index[-1].date()}, expected {end_date}")
    ind = compute_indicators(df)
    events = evaluate_events(ind)
    log_events(events, as_of=end_date)
    log_daily_event_summary(as_of=end_date)
    send_email(events)
    for ev, tk, price in events:
        print(f"{ev} on {tk} at {price:.2f}")

# Historical poll -----------------------------------------------

def historical_poll(start_date: date, end_date: date) -> None:
    """Evaluate events for each day in the range [start_date, end_date]."""
    current_date = start_date
    while current_date <= end_date:
        # Fetch data up to current_date with 20-day lookback for indicators
        fetch_start = current_date - timedelta(days=20)
        df = fetch_prices(fetch_start, current_date, interval="15m")
        if df.empty:
            print(f"No intraday data available for {current_date}")
            current_date += timedelta(days=1)
            continue
        df = df.resample("1D").last()
        if df.index[-1].date() != current_date:
            print(f"Warning: Latest data for {current_date} is from {df.index[-1].date()}")
        ind = compute_indicators(df)
        events = evaluate_events(ind)
        log_events(events, as_of=current_date)
        log_daily_event_summary(as_of=current_date)
        # send_email(events)  # Skip emailing for historical runs
        for ev, tk, price in events:
            print(f"{current_date}: {ev} on {tk} at {price:.2f}")
        current_date += timedelta(days=1)

# Daily summary -------------------------------------------------

def daily_summary(target: date) -> None:
    start = target - timedelta(days=260)
    df_d = fetch_prices(start, target, interval="1d")
    if df_d.empty:
        print(f"No daily data available for {target}")
        return
    if df_d.index[-1].date() != target:
        print(f"Warning: Latest data is from {df.index[-1].date()}, expected {target}")
    ind = compute_indicators(df_d)
    events = evaluate_events(ind)
    log_events(events, as_of=target)

    # Count events by ticker
    counts = {ev: {t: 0 for t in TICKERS} for ev in EVENTS}
    log_file = LOG_DIR / f"{target}_events.csv"
    if log_file.exists():
        try:
            with log_file.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if row["event"] in counts and row["ticker"] in TICKERS:
                        counts[row["event"]][row["ticker"]] += 1
        except Exception as e:
            print(f"Error reading log {log_file}: {e}")

    summary_path = LOG_DIR / f"{target}_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Daily summary for {target}\n")
        f.write(f"VIX close: {ind.get('VIX', 0):.2f}\n\n")
        for ev, desc in EVENTS.items():
            f.write(f"{ev:10} — {desc}\n")
            for t in TICKERS:
                f.write(f"  {t:6} {counts[ev][t]:>3} triggers\n")
        f.write("\nEvents logged:\n")
        for ev, tk, price in events:
            f.write(f"  {ev} on {tk} at {price:.2f}\n")
    with summary_path.open("r", encoding="utf-8") as f:
        print(f.read())

# CLI entry -----------------------------------------------------

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python market_dip_monitor.py [intraday | historical YYYY-MM-DD YYYY-MM-DD | daily_summary [YYYY-MM-DD]]")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "intraday":
        intraday_poll()
    elif cmd == "historical":
        if len(sys.argv) != 4:
            print("Usage: python market_dip_monitor.py historical YYYY-MM-DD YYYY-MM-DD")
            sys.exit(1)
        try:
            start_date = date.fromisoformat(sys.argv[2])
            end_date = date.fromisoformat(sys.argv[3])
        except ValueError:
            print("Dates must be YYYY-MM-DD")
            sys.exit(1)
        historical_poll(start_date, end_date)
    elif cmd == "daily_summary":
        if len(sys.argv) == 3:
            try:
                target_date = date.fromisoformat(sys.argv[2])
            except ValueError:
                print("Date must be YYYY-MM-DD")
                sys.exit(1)
        else:
            target_date = now().date()
        daily_summary(target_date)
    else:
        print("Unknown command.")
        sys.exit(1)

if __name__ == "__main__":
    main()