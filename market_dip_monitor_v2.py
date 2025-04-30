import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/market_dip_monitor_v2.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
TICKERS = ['VOO', 'QQQM', '^VIX']
POLL_INTERVAL = 15 * 60  # 15 minutes in seconds
TRADING_HOURS_START = pd.to_datetime('09:30:00').time()
TRADING_HOURS_END = pd.to_datetime('16:00:00').time()
PRICE_HISTORY_DAYS = 20  # For SMA20 and other indicators
DATA_DIR = 'logs'
EVENTS_FILE = f'{DATA_DIR}/2025-04-30_events.csv'
SUMMARY_FILE = f'{DATA_DIR}/event_summary.csv'
PRICES_FILE = f'{DATA_DIR}/prices_2025-04-30_15m.csv'

# Ensure data directory exists
Path(DATA_DIR).mkdir(exist_ok=True)

def fetch_prices(tickers, interval='15m', retries=3):
    """
    Fetch real-time prices for given tickers with retry mechanism.
    """
    for attempt in range(retries):
        try:
            logger.info(f"Fetching prices for {tickers} at {datetime.now()}")
            data = yf.download(tickers, period='1d', interval=interval, progress=False)
            if data.empty:
                raise ValueError("Empty data returned from yfinance")
            close_prices = data['Close'].iloc[-1].to_dict()
            validated_prices = {}
            for ticker in tickers:
                price = close_prices.get(ticker)
                if pd.isna(price) or price <= 0:
                    raise ValueError(f"Invalid price for {ticker}: {price}")
                validated_prices[ticker] = price
            return validated_prices
        except Exception as e:
            logger.error(f"Error fetching prices (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(5)
            else:
                logger.error("Max retries reached. Using last known prices or exiting.")
                return None
    return None

def calculate_indicators(prices_df, ticker):
    """
    Calculate technical indicators for a given ticker.
    """
    if len(prices_df) < 20:
        logger.warning(f"Insufficient data for {ticker} indicators: {len(prices_df)} periods")
        return {
            'SMA20': np.nan,
            'Lower_BB': np.nan,
            'RSI': np.nan,
            'Drawdown': 0.0
        }

    close = prices_df[ticker]
    sma20 = close.rolling(window=20).mean().iloc[-1]
    std20 = close.rolling(window=20).std().iloc[-1]
    lower_bb = sma20 - 2 * std20 if not pd.isna(std20) else np.nan

    # RSI calculation
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean().iloc[-1]
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean().iloc[-1]
    rs = gain / loss if loss != 0 else np.inf
    rsi = 100 - (100 / (1 + rs)) if rs != np.inf else np.nan

    # Drawdown calculation
    peak = close.max()
    current = close.iloc[-1]
    drawdown = ((peak - current) / peak) * 100 if peak != 0 else 0.0

    return {
        'SMA20': sma20,
        'Lower_BB': lower_bb,
        'RSI': rsi,
        'Drawdown': drawdown
    }

def check_events(ticker, price, indicators, vix):
    """
    Check for event triggers based on price, indicators, and VIX.
    """
    events = []
    if vix > 25:
        events.append(('VIX25', ticker, price))
    if not pd.isna(indicators['SMA20']) and price < indicators['SMA20'] * 0.95:
        events.append(('SMA20_GAP', ticker, price))
    if not pd.isna(indicators['Lower_BB']) and price < indicators['Lower_BB']:
        events.append(('BOLLINGER', ticker, price))
    if not pd.isna(indicators['RSI']) and indicators['RSI'] < 30:
        events.append(('RSI', ticker, price))
    if indicators['Drawdown'] > 10:
        events.append(('DD10', ticker, price))
    if len(events) > 1:
        events.append(('RISK_BRAKE', ticker, price))
    return events

def save_events(events, timestamp):
    """
    Save triggered events to CSV.
    """
    if not events:
        return
    event_data = [
        {'timestamp': timestamp, 'event': event, 'ticker': ticker, 'price': price}
        for event, ticker, price in events
    ]
    df = pd.DataFrame(event_data)
    file_exists = os.path.exists(EVENTS_FILE)
    df.to_csv(EVENTS_FILE, mode='a', header=not file_exists, index=False)
    logger.info(f"Saved {len(event_data)} events to {EVENTS_FILE}")

def update_summary(events):
    """
    Update event summary CSV.
    """
    summary = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'SMA20_GAP': 0,
        'BOLLINGER': 0,
        'RSI': 0,
        'VIX25': 0,
        'DD10': 0,
        'RISK_BRAKE': 0
    }
    for event, _, _ in events:
        if event in summary:
            summary[event] += 1
    
    summary_df = pd.DataFrame([summary])
    file_exists = os.path.exists(SUMMARY_FILE)
    summary_df.to_csv(SUMMARY_FILE, mode='a', header=not file_exists, index=False)
    logger.info(f"Updated summary in {SUMMARY_FILE}")

def save_prices(prices, timestamp):
    """
    Save price data to CSV.
    """
    price_data = {'Datetime': timestamp}
    price_data.update(prices)
    df = pd.DataFrame([price_data])
    file_exists = os.path.exists(PRICES_FILE)
    df.to_csv(PRICES_FILE, mode='a', header=not file_exists, index=False)
    logger.info(f"Saved prices to {PRICES_FILE}")

def validate_prices(prices, previous_prices):
    """
    Validate that prices have changed since the last poll.
    """
    if not previous_prices:
        return True
    for ticker, price in prices.items():
        if ticker in previous_prices and abs(price - previous_prices[ticker]) < 1e-5:
            logger.warning(f"Price for {ticker} unchanged: {price}")
            return False
    return True

def main():
    """
    Main loop to monitor market dips.
    """
    logger.info("Starting Market Dip Monitor v2")
    previous_prices = {}
    price_history = {ticker: [] for ticker in TICKERS}

    while True:
        now = datetime.now()
        current_time = now.time()
        
        # Check if within trading hours
        if TRADING_HOURS_START <= current_time <= TRADING_HOURS_END:
            # Fetch prices
            prices = fetch_prices(TICKERS)
            if prices is None:
                logger.error("Failed to fetch prices. Skipping this poll.")
                time.sleep(60)  # Wait before retrying
                continue

            # Validate prices
            if not validate_prices(prices, previous_prices):
                logger.error("Unchanged prices detected. Forcing refetch.")
                time.sleep(5)
                prices = fetch_prices(TICKERS)
                if prices is None:
                    logger.error("Refetch failed. Skipping this poll.")
                    time.sleep(60)
                    continue

            previous_prices = prices.copy()
            timestamp = now.strftime('%Y-%m-%d %H:%M:%S')

            # Save prices
            save_prices(prices, timestamp)

            # Update price history
            for ticker in TICKERS:
                price_history[ticker].append(prices[ticker])
                if len(price_history[ticker]) > 20:  # Keep only last 20 for indicators
                    price_history[ticker].pop(0)

            # Create DataFrame for indicators
            prices_df = pd.DataFrame(price_history)

            # Process each ticker
            all_events = []
            for ticker in ['VOO', 'QQQM']:
                indicators = calculate_indicators(prices_df, ticker)
                logger.info(f"{ticker} Indicators: Close={prices[ticker]:.2f}, "
                           f"SMA20={indicators['SMA20']:.2f}, "
                           f"Lower_BB={indicators['Lower_BB']:.2f}, "
                           f"RSI={indicators['RSI']:.2f}, "
                           f"Drawdown={indicators['Drawdown']:.2%}")

                events = check_events(ticker, prices[ticker], indicators, prices['^VIX'])
                all_events.extend(events)

            # Log and save events
            if all_events:
                logger.info(f"Events triggered: {all_events}")
                save_events(all_events, timestamp)
                update_summary(all_events)

        # Wait until next polling interval
        next_poll = (now + timedelta(seconds=POLL_INTERVAL)).replace(second=0, microsecond=0)
        sleep_seconds = (next_poll - datetime.now()).total_seconds()
        if sleep_seconds > 0:
            logger.info(f"Sleeping for {sleep_seconds:.0f} seconds until {next_poll}")
            time.sleep(sleep_seconds)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Market Dip Monitor stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise