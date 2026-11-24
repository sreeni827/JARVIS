from pathlib import Path
from datetime import datetime, timezone
import json

import numpy as np
import pandas as pd
import yaml

from market.market_data import MarketDataEngine
from market.indicators import TechnicalIndicators


BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"

CONFIG_FILE = (
    BASE_DIR / "config" / "market_regime.yaml"
)


class MarketRegimeEngine:

    def __init__(self, market=None):

        with open(
            CONFIG_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            self.config = yaml.safe_load(file)

        self.settings = self.config["settings"]
        self.benchmarks = self.config["benchmarks"]
        self.weights = self.config["weights"]

        if sum(self.benchmarks.values()) != 100:

            raise ValueError(
                "Benchmark weights must total 100."
            )

        if sum(self.weights.values()) != 100:

            raise ValueError(
                "Indicator weights must total 100."
            )

        self.market = (
            market if market is not None
            else MarketDataEngine()
        )

    def calculate_score(self, row):

        required = [

            "close",
            "SMA_20",
            "SMA_50",
            "SMA_200",
            "ROC_10",
            "RSI_14",
            "ATR_14"

        ]

        values = np.array(
            [float(row[key]) for key in required]
        )

        if not np.isfinite(values).all():

            raise ValueError(
                "Invalid benchmark indicators."
            )

        close = float(row["close"])

        if (
            close <= 0
            or row["ATR_14"] < 0
            or row["SMA_20"] <= 0
            or row["SMA_50"] <= 0
            or row["SMA_200"] <= 0
            or not 0 <= row["RSI_14"] <= 100
        ):

            raise ValueError(
                "Invalid price or indicator range."
            )

        components = {

            "long_term_trend": (
                1.0
                if close > row["SMA_200"]
                else 0.0
            ),

            "medium_term_trend": (
                1.0
                if row["SMA_50"] > row["SMA_200"]
                else 0.0
            ),

            "short_term_trend": (
                1.0
                if row["SMA_20"] > row["SMA_50"]
                else 0.0
            ),

            "momentum": float(
                np.clip(
                    (row["ROC_10"] + 5) / 10,
                    0,
                    1
                )
            ),

            "rsi": float(
                np.clip(
                    (row["RSI_14"] - 30) / 40,
                    0,
                    1
                )
            )

        }

        score = sum(

            components[key] * self.weights[key]

            for key in components

        )

        atr_percent = (
            float(row["ATR_14"])
            / close
            * 100
        )

        return score, atr_percent

    def analyze_benchmark(self, symbol):

        history = self.market.get_historical_data(

            symbol,

            days=self.settings["historical_days"]

        )

        if len(history) < self.settings[
            "minimum_candles"
        ]:

            raise ValueError(
                f"Insufficient history for {symbol}."
            )

        indicators = TechnicalIndicators.calculate(
            history
        )

        latest = indicators.iloc[-1]

        score, atr_percent = self.calculate_score(
            latest
        )

        timestamp = latest.name

        if isinstance(timestamp, tuple):

            timestamp = timestamp[-1]

        timestamp = pd.Timestamp(timestamp)

        if timestamp.tzinfo is None:

            raise ValueError(
                "Benchmark timestamp has no timezone."
            )

        session_date = timestamp.tz_convert(
            "America/New_York"
        ).date()

        return {

            "symbol": symbol,

            "session_date": str(session_date),

            "close": float(latest["close"]),

            "sma_20": float(latest["SMA_20"]),

            "sma_50": float(latest["SMA_50"]),

            "sma_200": float(latest["SMA_200"]),

            "rsi": float(latest["RSI_14"]),

            "momentum": float(latest["ROC_10"]),

            "atr_percent": atr_percent,

            "score": score,

            "weight": self.benchmarks[symbol]

        }

    def classify_regime(self, score, volatility):

        if (
            not np.isfinite(score)
            or not np.isfinite(volatility)
            or not 0 <= score <= 100
            or volatility < 0
        ):

            return "UNKNOWN"

        if volatility >= self.settings[
            "high_volatility_threshold"
        ]:

            return "HIGH_VOLATILITY"

        if score >= self.settings[
            "bullish_threshold"
        ]:

            return "BULLISH"

        if score <= self.settings[
            "bearish_threshold"
        ]:

            return "BEARISH"

        return "SIDEWAYS"

    def analyze_market(self):

        results = []
        errors = {}

        for symbol in self.benchmarks:

            try:

                result = self.analyze_benchmark(
                    symbol
                )

                results.append(result)

                print(

                    f"{symbol:<5} "

                    f"Score: {result['score']:>6.2f} | "

                    f"ATR: {result['atr_percent']:>5.2f}%"

                )

            except Exception as error:

                errors[symbol] = str(error)

                print(f"{symbol}: FAILED - {error}")

        dates = {
            item["session_date"]
            for item in results
        }

        if len(results) != len(self.benchmarks):

            return {

                "regime": "UNKNOWN",
                "market_score": None,
                "market_volatility": None,
                "benchmarks": results,
                "errors": errors,
                "reason": "Incomplete benchmark data."

            }

        if len(dates) != 1:

            return {

                "regime": "UNKNOWN",
                "market_score": None,
                "market_volatility": None,
                "benchmarks": results,
                "errors": errors,
                "reason": "Benchmark dates do not match."

            }

        market_score = sum(

            item["score"] * item["weight"] / 100

            for item in results

        )

        market_volatility = sum(

            item["atr_percent"] * item["weight"] / 100

            for item in results

        )

        regime = self.classify_regime(

            market_score,

            market_volatility

        )

        return {

            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),

            "session_date": next(iter(dates)),

            "regime": regime,

            "market_score": round(
                market_score, 2
            ),

            "market_volatility": round(
                market_volatility, 2
            ),

            "benchmarks": results,

            "errors": errors

        }

    def display_summary(self, result):

        print("\n" + "=" * 75)

        print("                  MARKET REGIME SUMMARY")

        print("=" * 75)

        print(
            f"Market Regime      : {result['regime']}"
        )

        if result["regime"] == "UNKNOWN":

            print(
                f"Reason             : "
                f"{result.get('reason', 'Invalid data')}"
            )

            print("Data Status        : NOT READY")

            return

        print(

            f"Market Score       : "

            f"{result['market_score']:.2f}/100"

        )

        print(

            f"Market Volatility  : "

            f"{result['market_volatility']:.2f}%"

        )

        print(
            f"Data Session       : "
            f"{result['session_date']}"
        )

        print(
            f"Benchmarks Analyzed: "
            f"{len(result['benchmarks'])}"
        )

        print("Mode               : RESEARCH ONLY")

    def save_results(self, result):

        DATA_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%d_%H%M%S")

        json_file = (

            DATA_DIR /

            f"market_regime_{timestamp}.json"

        )

        with open(

            json_file,

            "w",

            encoding="utf-8"

        ) as file:

            json.dump(
                result,
                file,
                indent=4,
                allow_nan=False
            )

        print(f"\nSaved: {json_file.name}")

        if result["benchmarks"]:

            csv_file = (

                DATA_DIR /

                f"market_benchmarks_{timestamp}.csv"

            )

            pd.DataFrame(

                result["benchmarks"]

            ).to_csv(

                csv_file,

                index=False

            )

            print(f"Saved: {csv_file.name}")


if __name__ == "__main__":

    print("=" * 75)

    print("                JARVIS MARKET REGIME ENGINE")

    print("                        VERSION 0.8.3")

    print("=" * 75)

    print("Mode: MARKET INTELLIGENCE")

    print("Order Execution: DISABLED\n")

    engine = MarketRegimeEngine()

    result = engine.analyze_market()

    engine.display_summary(result)

    engine.save_results(result)

    print("\n" + "=" * 75)

    if result["regime"] == "UNKNOWN":

        print("MARKET REGIME ENGINE: DATA NOT READY")

        raise SystemExit(1)

    print("MARKET REGIME ENGINE: SUCCESSFUL")

    print("NO ORDERS EXECUTED")

    print("=" * 75)