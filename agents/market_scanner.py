from pathlib import Path
from datetime import datetime, timezone
import time

import pandas as pd
import numpy as np
import yaml

from agents.technical_agent import TechnicalAgent


BASE_DIR = Path(__file__).resolve().parents[1]

WATCHLIST_FILE = BASE_DIR / "config" / "watchlist.yaml"

DATA_DIR = BASE_DIR / "data"


class MarketScanner:

    def __init__(self):

        self.agent = TechnicalAgent()

        self.watchlist = self.load_watchlist()

        self.errors = []

    def load_watchlist(self):

        with open(
            WATCHLIST_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            config = yaml.safe_load(file)

        symbols = {}

        for sector, stocks in config["watchlist"].items():

            for symbol in stocks:

                symbols[symbol] = sector

        return symbols

    def scan(self):

        results = []

        self.errors = []

        total = len(self.watchlist)

        print(f"\nScanning {total} symbols...\n")

        for index, (symbol, sector) in enumerate(
            self.watchlist.items(),
            start=1
        ):

            try:

                result = self.agent.analyze(symbol)

                numeric_values = [
                    value
                    for value in result.values()
                    if isinstance(value, (int, float))
                ]

                if not all(np.isfinite(numeric_values)):
                    raise ValueError("Invalid indicator values")

                result["sector"] = sector

                results.append(result)

                print(
                    f"[{index:02d}/{total}] "
                    f"{symbol:<6} "
                    f"${result['close']:>9.2f} "
                    f"{result['trend']:<8} "
                    f"RSI: {result['rsi']:>6.2f} "
                    f"RVOL: {result['relative_volume']:.2f}x"
                )

            except Exception as error:

                self.errors.append({
                    "symbol": symbol,
                    "error": type(error).__name__
                })

                print(
                    f"[{index:02d}/{total}] "
                    f"{symbol:<6} "
                    f"FAILED: {type(error).__name__}"
                )

            time.sleep(0.3)

        return pd.DataFrame(results)

    def save_results(self, results):

        DATA_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%d_%H%M%S")

        filename = (
            DATA_DIR /
            f"market_scan_{timestamp}.csv"
        )

        results.to_csv(
            filename,
            index=False
        )

        print(f"\nResults saved to: {filename}")

    def display_summary(self, results):

        if results.empty:

            print("\nNo successful scan results.")

            return

        bullish = results[
            results["trend"] == "BULLISH"
        ]

        bearish = results[
            results["trend"] == "BEARISH"
        ]

        neutral = results[
            results["trend"] == "NEUTRAL"
        ]

        unusual_volume = results[
            results["relative_volume"] >= 2
        ]

        oversold = results[
            results["rsi"] < 30
        ]

        overbought = results[
            results["rsi"] > 70
        ]

        print("\n" + "=" * 65)
        print("                  MARKET SCAN SUMMARY")
        print("=" * 65)

        print(f"Symbols Requested : {len(self.watchlist)}")
        print(f"Successful Scans  : {len(results)}")
        print(f"Failed Scans      : {len(self.errors)}")
        print(f"Bullish           : {len(bullish)}")
        print(f"Bearish           : {len(bearish)}")
        print(f"Neutral           : {len(neutral)}")
        print(f"Unusual Volume    : {len(unusual_volume)}")
        print(f"Oversold          : {len(oversold)}")
        print(f"Overbought        : {len(overbought)}")

        columns = [
            "symbol",
            "close",
            "trend",
            "rsi",
            "relative_volume",
            "momentum"
        ]

        print("\nTOP POSITIVE MOMENTUM")
        print("-" * 65)

        print(
            results[results["momentum"] > 0]
            .sort_values(
                "momentum",
                ascending=False
            )[columns]
            .head(5)
            .round(2)
            .to_string(index=False)
        )

        print("\nTOP NEGATIVE MOMENTUM")
        print("-" * 65)

        print(
            results[results["momentum"] < 0]
            .sort_values(
                "momentum",
                ascending=True
            )[columns]
            .head(5)
            .round(2)
            .to_string(index=False)
        )

        print("\nHIGHEST RELATIVE VOLUME")
        print("-" * 65)

        print(
            results.sort_values(
                "relative_volume",
                ascending=False
            )[columns]
            .head(5)
            .round(2)
            .to_string(index=False)
        )

        if self.errors:

            print("\nFAILED SYMBOLS")
            print("-" * 65)

            for error in self.errors:

                print(
                    f"{error['symbol']}: "
                    f"{error['error']}"
                )


if __name__ == "__main__":

    print("=" * 65)
    print("                  JARVIS MARKET SCANNER")
    print("                       VERSION 0.4")
    print("=" * 65)

    start_time = time.perf_counter()

    scanner = MarketScanner()

    results = scanner.scan()

    scanner.display_summary(results)

    if not results.empty:

        scanner.save_results(results)

    elapsed = time.perf_counter() - start_time

    print(f"\nExecution Time: {elapsed:.2f} seconds")

    print("=" * 65)

    if results.empty:
        print("MARKET SCANNER: FAILED")
    elif scanner.errors:
        print("MARKET SCANNER: COMPLETED WITH ERRORS")
    else:
        print("MARKET SCANNER: SUCCESSFUL")

    print("=" * 65)