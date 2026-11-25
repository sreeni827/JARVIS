import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BASE_DIR / ".env", override=True)


class MarketDataEngine:
    def __init__(self):

        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            raise RuntimeError("Alpaca credentials missing.")

        self.client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)

    def get_latest_prices(self, symbols):

        request = StockLatestTradeRequest(symbol_or_symbols=symbols, feed=DataFeed.IEX)

        trades = self.client.get_stock_latest_trade(request)

        results = {}

        for symbol, trade in trades.items():
            results[symbol] = {
                "price": float(trade.price),
                "size": trade.size,
                "timestamp": trade.timestamp,
            }

        return results

    def get_historical_data(self, symbol, days=400):

        end = datetime.now(UTC).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        start = end - timedelta(days=days)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            feed=DataFeed.IEX,
            adjustment=Adjustment.ALL,
        )

        bars = self.client.get_stock_bars(request)

        return bars.df


if __name__ == "__main__":
    engine = MarketDataEngine()

    symbols = ["AAPL", "NVDA", "MSFT", "TSLA", "AMZN", "META", "GOOGL", "SPY", "QQQ"]

    print("=" * 60)
    print("           JARVIS MARKET DATA ENGINE")
    print("                 VERSION 0.2")
    print("=" * 60)

    try:
        prices = engine.get_latest_prices(symbols)

        print("\nLATEST AVAILABLE TRADE PRICES")
        print("-" * 60)

        for symbol, data in prices.items():
            print(f"{symbol:<8} ${data['price']:>10,.2f} Size: {data['size']}")

        print("\nNVDA HISTORICAL DATA")
        print("-" * 60)

        history = engine.get_historical_data("NVDA")

        print(history.tail())

        print("\nMARKET DATA ENGINE: SUCCESSFUL")

    except Exception as error:
        print(f"Market data error: {type(error).__name__}")

        print("Check market data permissions and connection.")

    print("=" * 60)
