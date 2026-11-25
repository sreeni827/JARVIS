from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from intelligence.market_regime import MarketRegimeEngine
from market.indicators import TechnicalIndicators

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"


class HistoricalRegimeEngine:
    def __init__(self):

        self.engine = MarketRegimeEngine()

    def prepare_benchmark(self, symbol, days):

        print(f"Downloading {symbol} historical data...")

        history = self.engine.market.get_historical_data(symbol, days=days)

        indicators = TechnicalIndicators.calculate(history)

        records = []

        for timestamp, row in indicators.iterrows():
            if isinstance(timestamp, tuple):
                timestamp = timestamp[-1]

            timestamp = pd.Timestamp(timestamp)

            if timestamp.tzinfo is None:
                raise ValueError("Historical timestamp has no timezone.")

            session_date = timestamp.tz_convert("America/New_York").date()

            try:
                score, volatility = self.engine.calculate_score(row)

            except ValueError:
                continue

            records.append(
                {
                    "session_date": pd.Timestamp(session_date),
                    f"{symbol}_score": score,
                    f"{symbol}_volatility": volatility,
                }
            )

        if not records:
            raise ValueError(f"No valid indicators for {symbol}.")

        result = pd.DataFrame(records)

        if result["session_date"].duplicated().any():
            raise ValueError(f"Duplicate sessions for {symbol}.")

        return result.set_index("session_date").sort_index()

    def build_history(self, days=1500):

        benchmarks = []

        for symbol in self.engine.benchmarks:
            benchmark = self.prepare_benchmark(symbol, days)

            benchmarks.append(benchmark)

        combined = pd.concat(benchmarks, axis=1, join="outer").sort_index()

        if combined.empty:
            raise ValueError("No benchmark history available.")

        results = []

        for session_date, row in combined.iterrows():
            if not np.isfinite(row.to_numpy(dtype=float)).all():
                results.append(
                    {
                        "session_date": session_date,
                        "market_score": np.nan,
                        "market_volatility": np.nan,
                        "regime": "UNKNOWN",
                    }
                )

                continue

            score = 0.0

            volatility = 0.0

            for symbol, weight in self.engine.benchmarks.items():
                score += row[f"{symbol}_score"] * weight / 100

                volatility += row[f"{symbol}_volatility"] * weight / 100

            regime = self.engine.classify_regime(score, volatility)

            results.append(
                {
                    "session_date": session_date,
                    "market_score": score,
                    "market_volatility": volatility,
                    "regime": regime,
                }
            )

        return pd.DataFrame(results)

    def save_results(self, results):

        DATA_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        filename = DATA_DIR / f"historical_regimes_{timestamp}.csv"

        results.to_csv(filename, index=False)

        print(f"\nSaved: {filename.name}")

    def run(self):

        results = self.build_history()

        print("\nHISTORICAL MARKET REGIMES")
        print("-" * 65)

        distribution = (
            results["regime"]
            .value_counts()
            .rename_axis("regime")
            .reset_index(name="sessions")
        )

        distribution["percentage"] = distribution["sessions"] / len(results) * 100

        print(distribution.round(2).to_string(index=False))

        print(f"\nTotal Sessions: {len(results)}")

        print(f"Start Date: {results['session_date'].min().date()}")

        print(f"End Date: {results['session_date'].max().date()}")

        self.save_results(results)

        return results


if __name__ == "__main__":
    print("=" * 75)

    print("        JARVIS HISTORICAL MARKET REGIME ENGINE")

    print("                     VERSION 0.8.4")

    print("=" * 75)

    print("Mode: HISTORICAL RESEARCH")

    print("Order Execution: DISABLED\n")

    engine = HistoricalRegimeEngine()

    engine.run()

    print("\nHISTORICAL REGIME ANALYSIS COMPLETE")
